import torch.utils.data
import torch
from tqdm import tqdm
import os
import numpy as np
from models.model_single import ModelEmb as ModelEmb
from models.model_single_rgbd import ModelEmb as ModelEmbRGBD

from segment_anything_1 import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from dataset.davsod_video import get_davsod_dataset_test
from dataset.ViDSOD100 import get_vidsod_dataset_test
from segment_anything_1.utils.transforms import ResizeLongestSide as ResizeLongestSide_sam1
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from utils import save_image
import torch.nn.functional as F
from train_video_rgbd import get_input_dict, norm_batch, unpad


def get_dice_ji(predict_scores, target_in, smooth=1e-8):
    target = target_in + 1
    thresholds = np.linspace(0, 1, 21)
    # thresholds = [2/3]
    f_betas = []
    for thresh in thresholds:
        predict = predict_scores.copy()
        predict[predict_scores > thresh] = 1
        predict[predict_scores <= thresh] = 0
        predict = predict + 1
        tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
        fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
        fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
        ji = float(np.nan_to_num(tp / (tp + fp + fn + smooth)))
        dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn + smooth)))

        precision = float(np.nan_to_num(tp / (tp + fp))) if (tp + fp) > 0 else 0.0
        recall = float(np.nan_to_num(tp / (tp + fn))) if (tp + fn) > 0 else 0.0
        beta_sq = 0.3
        f_beta = float(np.nan_to_num((1 + beta_sq) * precision * recall / (beta_sq * precision + recall + smooth))) if (precision + recall) > 0 else 0.0
        f_betas.append(f_beta)
    return dice, ji, f_betas


def get_mae(predict, target):
    mae = np.mean(np.abs(predict - target))
    return mae


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return (path + '/gpu' + str(len(a)))


#
#
# def postprocess_masks(masks_dict):
#     masks = torch.zeros((len(masks_dict), *masks_dict[0]['low_res_logits'].squeeze().shape)).unsqueeze(dim=1).cuda()
#     ious = torch.zeros(len(masks_dict)).cuda()
#     for i in range(len(masks_dict)):
#         cur_mask = masks_dict[i]['low_res_logits'].squeeze()
#         cur_mask = (cur_mask - cur_mask.min()) / (cur_mask.max() - cur_mask.min())
#         masks[i, 0] = cur_mask.squeeze()
#         ious[i] = masks_dict[i]['iou_predictions'].squeeze()
#     return masks, ious


def call_model(model, model_input_rgb, model_input_depth, device, use_depth):
    # Define pixel mean and std as tensors
    pixel_mean = torch.tensor([123.675, 116.28, 103.53], device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375], device=device).view(1, 3, 1, 1)
    # Normalize the input
    normalized_input = (model_input_rgb - pixel_mean) / pixel_std

    num_frames = model_input_rgb.shape[1]
    outputs = []
    for idx_frame in range(num_frames):
        normalized_input_frame = normalized_input[:, idx_frame]
        depth_frame= model_input_depth[:,idx_frame]
        if use_depth:
            output = model(normalized_input_frame, depth_frame)
        else:
            output = model(normalized_input_frame)
        outputs.append(output.cpu())
    outputs = torch.stack(outputs, dim=1)

    return outputs


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, args, test_run, device, use_depth=True):
        self.eval_root = args['root_images_eval']
        self.Idim = int(args['Idim'])
        self.test_run = test_run
        self.num_outputs = 5
        self.device=device
        self.use_depth = use_depth

    @torch.inference_mode()
    def inference_ds(self, ds, model, sam, transform, epoch, device):
        num_images = len(ds)
        denom = num_images // self.num_outputs
        pbar = tqdm(ds)
        model.eval()
        iou_list = []
        dice_list = []
        f_beta_list = []
        f_betas_list = []
        mae_list = []
        eval_dir = os.path.join(self.eval_root, str(epoch))
        if not os.path.isdir(eval_dir):
            os.mkdir(eval_dir)
        for ii, (imgs, gts, depth, original_szs, img_szs) in enumerate(pbar):
            batch_size, seq_len, c, h, w = imgs.shape  # images have shape [B, T, C, H, W]

            assert torch.all(original_szs == original_szs[0, 0])
            assert torch.all(img_szs == img_szs[0, 0])
            img_sz = img_szs[:, 0]
            original_sz = original_szs[:, 0]

            orig_imgs = imgs.to(device)
            gts = gts.to(device)
            depth_imgs = depth.to(device)
            orig_imgs_small = F.interpolate(orig_imgs.view(-1, c, h, w), (self.Idim, self.Idim), mode='bilinear', align_corners=True)
            orig_imgs_small = orig_imgs_small.view(batch_size, seq_len, c, self.Idim, self.Idim)
            depth_imgs_small = F.interpolate(depth_imgs.view(-1, 1, h, w), (self.Idim, self.Idim), mode='bilinear',
                                            align_corners=True)
            depth_imgs_small = depth_imgs_small.view(batch_size, seq_len, 1, self.Idim, self.Idim)

            dense_embeddings = call_model(model, orig_imgs_small, depth_imgs_small, device=device, use_depth=self.use_depth)
            batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
            masks, _ = sam_call(batched_input, sam, dense_embeddings, device)
            masks = norm_batch(masks)

            masks = unpad(masks, original_sz)
            gts = unpad(gts, original_sz)

            masks_score = masks.clone()

            mae = get_mae(
                   masks_score.squeeze().detach().cpu().numpy(),
                   gts.squeeze().detach().cpu().numpy().astype(np.float32)
            )

            dice, ji, f_betas = get_dice_ji(masks_score.squeeze().detach().cpu().numpy(),
                                   gts.squeeze().detach().cpu().numpy())
            f_betas = np.array(f_betas)
            f_beta = f_betas[15]
            iou_list.append(ji)
            dice_list.append(dice)
            f_beta_list.append(f_beta)
            f_betas_list.append(f_betas)
            mae_list.append(mae)
            pbar.set_description(
                '(Inference | {task}) Epoch {epoch} :: Dice {dice:.4f} :: MAE {mae:.4f} :: F_beta {f_beta:.4f}'.format(
                    task=args['task'],
                    epoch=epoch,
                    dice=np.mean(dice_list),
                    mae=np.mean(mae_list),
                    f_beta=np.mean(f_beta_list)))

            for idx_frame in range(seq_len):
                if idx_frame % denom == 0:
                    save_image(unpad(orig_imgs[0, idx_frame], original_sz), f'{eval_dir}/{ii}_{idx_frame}_image_in.png', is_mask=False)
                    save_image(gts[0, idx_frame], f'{eval_dir}/{ii}_{idx_frame}_gt_mask.png', is_mask=True)
                    masks[masks_score > 0.75] = 1
                    masks[masks_score <= 0.75] = 0
                    save_image(masks[0, idx_frame], f'{eval_dir}/{ii}_{idx_frame}_pred_mask.png', is_mask=True)

            if self.test_run:
                break
        f_betas_all = np.array(f_betas_list)
        f_betas_mean = f_betas_all.mean(axis=0)
        idx_f_beta_max = np.argmax(f_betas_mean)
        f_beta_max = f_betas_mean[idx_f_beta_max]
        print(f'maximum f score: {f_beta_max}')
        print(f'maximum f score index: {idx_f_beta_max}')
        model.train()
        return f_beta_max, f_betas_all


def sam_call(batched_input, sam, dense_embeddings, device):
    with torch.no_grad():
        input_images = torch.stack([x["image"] / 255 for x in batched_input], dim=0)
        bs, num_frames, c, H, W = input_images.shape
        inference_state = sam.init_state(
            images_in=input_images.permute(1, 0, 2, 3, 4),
            offload_video_to_cpu=False,
            offload_state_to_cpu=False,
        )
    input_points = None
    input_labels = None
    dense_embeddings_frame = dense_embeddings[:, 0].to(device=device)
    _, out_objs_ids_frame, _ = sam.add_new_points_or_box(
        inference_state=inference_state,
        frame_idx=0,
        obj_id=0,
        points=input_points,
        labels=input_labels,
        box=None,
        dense_embeddings_pred=dense_embeddings_frame,
    )
    out_mask_logits_cond = []
    for out_frame_idx, out_obj_ids, out_mask_logits_frame in sam.propagate_in_video(inference_state):
        out_mask_logits_frame = torch.clamp(out_mask_logits_frame, -32.0, 32.0)
        assert out_objs_ids_frame == [0]
        # out_mask_logits_cond.append(out_mask_logits_frame)
    # out_mask_logits_points = [out_mask_logits_frame_0]
    for frame_idx in range(1, num_frames):
        dense_embeddings_frame = dense_embeddings[:, frame_idx].to(device=device)
        # dense_embeddings_frame = None
        # input_points = np.array([[[(W // 2 + 100), (H * (360 / 640)) // 2 - 100]] for _ in range(bs)]) #  cat batch_size
        # input_labels = np.array([[1] for _ in range(bs)]) #  cat batch_size
        with torch.no_grad():
            _, out_objs_ids_frame, out_mask_logits_frame = sam.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=0,
            points=input_points,
            labels=input_labels,
            box=None,
            dense_embeddings_pred=dense_embeddings_frame,
            )
    out_mask_logits_final = []
    for out_frame_idx, out_obj_ids, out_mask_logits_frame in sam.propagate_in_video(inference_state):
        out_mask_logits_frame = torch.clamp(out_mask_logits_frame, -32.0, 32.0)
        assert out_objs_ids_frame == [0]
        out_mask_logits_final.append(out_mask_logits_frame)
    out_mask_logits_final = torch.stack(out_mask_logits_final, dim=1)
    return out_mask_logits_final, None


def main(args=None, sam_args=None, test_run=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args['use_depth']:
        model = ModelEmbRGBD(args=args, size_out=64, train_decoder_only=True).to(device)
    else:
        model = ModelEmb(args=args, size_out=64, train_decoder_only=True).to(device)

    model1 = torch.load(args['path_best'], weights_only=False)
    model.load_state_dict(model1.state_dict())

    model_cfg = sam_args['fp_config']
    sam = build_sam2_video_predictor(model_cfg, sam_args['checkpoint'], device=device)
    sam.fill_hole_area = 0
    assert not sam.training
    # sam = SAM2ImagePredictor(build_sam2(model_cfg, sam_args['checkpoint'], device=device))
    transform = None

    if args['task'] == 'davsod':
        testset = get_davsod_dataset_test(args['root_data_dir'], sam_trans=transform, cutoff_eval=args['cutoff_eval'], dataset=args['dataset'], add_depth=True)
    elif args['task'] == 'VIDSOD':
        testset = get_vidsod_dataset_test(args['root_data_dir'], sam_trans=transform,
                                               cutoff_eval=args['cutoff_eval'])
    else:
        raise Exception('unsupported task')
    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=int(args['nW_eval']), drop_last=False)

    inference_ds = InferenceDataset(args, test_run, device, use_depth=args['use_depth'])

    with torch.no_grad():
        f_beta_max, f_beta_all = inference_ds.inference_ds(ds_val, model.eval(), sam, transform, 0, device)
        eval_dir = os.path.join(args['root_images_eval'], '0')
        np.savetxt(os.path.join(eval_dir, "f_betas.csv"), f_beta_all, delimiter=",")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--root_data_dir', required=True, help='root data directory')
    parser.add_argument('--sam2_size', default='large', help='root data directory')
    # parser.add_argument('--eval_dir_root', default='eval', help='root eval image directory')
    parser.add_argument('-nW', '--nW', default=0, help='num workers train', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='num workers eval', required=False)
    parser.add_argument('-WD', '--WD', default=0, help='weight decay', required=False)  # 1e-4
    parser.add_argument('-task', '--task', default='VIDSOD', help='segmenatation task type', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='unkown effect, model_single.py', required=False)
    parser.add_argument('-order', '--order', default=85, help='unkown effect, model_single.py', required=False)
    parser.add_argument('-Idim', '--Idim', default=512, help='image size', required=False)
    parser.add_argument('--test_run', default=False, type=bool, help='if True, stops all train / eval loops after single iteration / input', required=False)
    parser.add_argument('--cutoff_eval', default=None, type=int, help='sets max length for eval datasets.', required=False)
    parser.add_argument('-folder', '--folder', help='image size', required=True)
    parser.add_argument('--dataset', default='easy', help='test dataset. easy, normal, hard, vidsod')
    parser.add_argument('--use_depth', default=True, type=bool, help='If true, uses RGBD backbone for the prompt encoder')
    args = vars(parser.parse_args())

    os.makedirs('results_test', exist_ok=True)
    folder_load = args['folder']
    args['results_root'] = open_folder('results_test')
    args['path_best'] = os.path.join('results',
                                     'gpu' + str(args['folder']),
                                     'net_best.pth')
    args['root_images_eval'] = os.path.join(args['results_root'],
                                     'eval_images')
    args['root_images_train'] = os.path.join(args['results_root'],
                                     'train_images')
    os.mkdir(args['root_images_eval'])
    os.mkdir(args['root_images_train'])
    if args['sam2_size'] == 'large':
        checkpoint = "cp_sam2/sam2.1_hiera_large.pt"
        fp_config = "configs/sam2.1/sam2.1_hiera_l.yaml"
    elif args['sam2_size'] == 'base_plus':
        checkpoint = "cp_sam2/sam2.1_hiera_base_plus.pt"
        fp_config = "configs/sam2.1/sam2.1_hiera_b+.yaml"
    elif args['sam2_size'] == 'tiny':
        checkpoint = "cp_sam2/sam2.1_hiera_tiny.pt"
        fp_config = "configs/sam2.1/sam2.1_hiera_t.yaml"
    else:
        raise Exception("Invalid sam size")

    sam_args = {
        'fp_config': fp_config,
        'checkpoint': checkpoint,
    }
    if args['test_run']:
        args['cutoff_eval'] = 5
    main(args=args, sam_args=sam_args, test_run=args['test_run'])
