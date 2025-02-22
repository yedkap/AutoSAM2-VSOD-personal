import torch.optim as optim
import torch.utils.data
import torch
import torch.nn as nn
from tqdm import tqdm
import os
import numpy as np
from models.model_single import ModelEmb
from segment_anything_1 import SamPredictor, sam_model_registry, SamAutomaticMaskGenerator
from dataset.davsod import get_davsod_dataset
from segment_anything_1.utils.transforms import ResizeLongestSide as ResizeLongestSide_sam1
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2
import torch.nn.functional as F
from utils import save_image

SAM_VERSION = 2

def norm_batch(x):
    bs = x.shape[0]
    H, W = x.shape[-2], x.shape[-1]
    min_value = x.view(bs, -1).min(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, H, W)
    max_value = x.view(bs, -1).max(dim=1)[0].repeat(1, 1, 1, 1).permute(3, 2, 1, 0).repeat(1, 1, H, W)
    x = (x - min_value) / (max_value - min_value + 1e-6)
    # x = torch.sigmoid(x)
    return x


def Dice_loss(y_true, y_pred, smooth=1):
    alpha = 0.5
    beta = 0.5
    tp = torch.sum(y_true * y_pred, dim=(1, 2, 3))
    fn = torch.sum(y_true * (1 - y_pred), dim=(1, 2, 3))
    fp = torch.sum((1 - y_true) * y_pred, dim=(1, 2, 3))
    tversky_class = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - torch.mean(tversky_class)


def get_dice_ji(predict, target):
    predict = predict + 1
    target = target + 1
    tp = np.sum(((predict == 2) * (target == 2)) * (target > 0))
    fp = np.sum(((predict == 2) * (target == 1)) * (target > 0))
    fn = np.sum(((predict == 1) * (target == 2)) * (target > 0))
    ji = float(np.nan_to_num(tp / (tp + fp + fn)))
    dice = float(np.nan_to_num(2 * tp / (2 * tp + fp + fn)))
    return dice, ji


def open_folder(path):
    if not os.path.exists(path):
        os.mkdir(path)
    a = os.listdir(path)
    os.mkdir(path + '/gpu' + str(len(a)))
    return str(len(a))


def gen_step(optimizer, gts, masks, criterion, accumulation_steps, step):
    size = masks.shape[2:]
    if len(masks.shape) == 3:
        masks = masks.unsqueeze(dim=1)
    if len(gts.shape) == 3:
        gts = gts.unsqueeze(dim=1)
    gts_sized = F.interpolate(gts, size, mode='nearest')
    loss = criterion(masks, gts_sized) + Dice_loss(masks, gts_sized)
    loss.backward()
    if (step + 1) % accumulation_steps == 0:  # Wait for several backward steps
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def get_input_dict(imgs, original_sz, img_sz):
    batched_input = []
    for i, img in enumerate(imgs):
        input_size = tuple([int(x) for x in img_sz[i].squeeze().tolist()])
        original_size = tuple([int(x) for x in original_sz[i].squeeze().tolist()])
        singel_input = {
            'image': img,
            'original_size': original_size,
            'image_size': input_size,
            'point_coords': None,
            'point_labels': None,
        }
        batched_input.append(singel_input)
    return batched_input


def postprocess_masks(masks_dict):
    masks = torch.zeros((len(masks_dict), *masks_dict[0]['low_res_logits'].squeeze().shape)).unsqueeze(dim=1).cuda()
    ious = torch.zeros(len(masks_dict)).cuda()
    for i in range(len(masks_dict)):
        cur_mask = masks_dict[i]['low_res_logits'].squeeze()
        cur_mask = (cur_mask - cur_mask.min()) / (cur_mask.max() - cur_mask.min())
        masks[i, 0] = cur_mask.squeeze()
        ious[i] = masks_dict[i]['iou_predictions'].squeeze()
    return masks, ious


def call_model(model, model_input):
    # Define pixel mean and std as tensors
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

    # Normalize the input
    normalized_input = (model_input - pixel_mean) / pixel_std
    output = model(normalized_input)
    return output

class Trainer(torch.utils.data.Dataset):
    def __init__(self, args, test_run=False):
        self.train_root = args['root_images_train']
        self.Idim = int(args['Idim'])
        self.test_run = test_run
        self.num_outputs = 80

    def train_single_epoch(self, ds, model, sam, optimizer, transform, epoch, device, accumulation_steps, test_run=False):
        num_images = len(ds)
        denom = num_images // self.num_outputs
        loss_list = []
        pbar = tqdm(ds)
        criterion = nn.BCELoss()
        optimizer.zero_grad()
        train_dir = os.path.join(self.train_root, str(epoch))
        if not os.path.isdir(train_dir):
            os.mkdir(train_dir)
        for ii, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
            orig_imgs = imgs.to(device)
            gts = gts.to(device)
            orig_imgs_small = F.interpolate(orig_imgs, (self.Idim, self.Idim), mode='bilinear', align_corners=True)
            if SAM_VERSION == 1:
                dense_embeddings = model(orig_imgs_small)
            else:
                dense_embeddings = call_model(model, orig_imgs_small)
            batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
            masks, masks_gt = sam_call(batched_input, sam, dense_embeddings)
            masks = norm_batch(masks)
            if masks_gt is not None:
                # gts = norm_batch(masks_gt)
                masks_gt[masks_gt > 0.] = 1
                masks_gt[masks_gt <= 0.] = 0
                gts = masks_gt
            loss = gen_step(optimizer, gts, masks, criterion, accumulation_steps=accumulation_steps, step=ii)
            loss_list.append(loss)
            pbar.set_description(
                '(train | {}) epoch {epoch} ::'
                ' loss {loss:.4f}'.format(
                    'Medical',
                    epoch=epoch,
                    loss=np.mean(loss_list)
                ))
            if ii % denom == 0:
                save_image(orig_imgs[0], f'{train_dir}/{ii}_image_in.png', is_mask=False)
                save_image(gts[0], f'{train_dir}/{ii}_gt_mask.png', is_mask=True)
                save_image(masks[0], f'{train_dir}/{ii}_pred_mask.png', is_mask=True)

            if test_run:
                break
        return np.mean(loss_list)


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, args, test_run=False):
        self.eval_root = args['root_images_eval']
        self.Idim = int(args['Idim'])
        self.test_run = test_run
        self.num_outputs = 120

    def inference_ds(self, ds, model, sam, transform, epoch, device):
        num_images = len(ds)
        denom = num_images // self.num_outputs
        pbar = tqdm(ds)
        model.eval()
        iou_list = []
        dice_list = []
        eval_dir = os.path.join(self.eval_root, str(epoch))
        if not os.path.isdir(eval_dir):
            os.mkdir(eval_dir)
        for ii, (imgs, gts, original_sz, img_sz) in enumerate(pbar):
            orig_imgs = imgs.to(device)
            gts = gts.to(device).unsqueeze(dim=0)
            orig_imgs_small = F.interpolate(orig_imgs, (self.Idim, self.Idim), mode='bilinear', align_corners=True)
            if SAM_VERSION == 1:
                dense_embeddings = model(orig_imgs_small)
            else:
                dense_embeddings = call_model(model, orig_imgs_small)
            batched_input = get_input_dict(orig_imgs, original_sz, img_sz)
            masks, _ = sam_call(batched_input, sam, dense_embeddings)
            masks = norm_batch(masks)
            # input_size = tuple([int(x) for x in img_sz[0].squeeze().tolist()])
            # original_size = tuple([int(x) for x in original_sz[0].squeeze().tolist()])
            # masks = sam.postprocess_masks(masks, input_size=input_size, original_size=original_size)
            # gts = sam.postprocess_masks(gts, input_size=input_size, original_size=original_size)
            masks = F.interpolate(masks, (self.Idim, self.Idim), mode='bilinear', align_corners=True)
            gts = F.interpolate(gts, (self.Idim, self.Idim), mode='nearest')
            masks[masks > 0.5] = 1
            masks[masks <= 0.5] = 0
            dice, ji = get_dice_ji(masks.squeeze().detach().cpu().numpy(),
                                   gts.squeeze().detach().cpu().numpy())
            iou_list.append(ji)
            dice_list.append(dice)
            pbar.set_description(
                '(Inference | {task}) Epoch {epoch} :: Dice {dice:.4f} :: IoU {iou:.4f}'.format(
                    task=args['task'],
                    epoch=epoch,
                    dice=np.mean(dice_list),
                    iou=np.mean(iou_list)))

            if ii % denom == 0:
                save_image(orig_imgs, f'{eval_dir}/image_in_{ii}.png', is_mask=False)
                save_image(gts, f'{eval_dir}/gt_mask_{ii}.png', is_mask=True)
                save_image(masks, f'{eval_dir}/pred_mask_{ii}.png', is_mask=True)

            if self.test_run:
                break
        model.train()
        return np.mean(iou_list)


def sam_call(batched_input, sam, dense_embeddings):
    if SAM_VERSION == 1:
        low_res_masks, low_res_masks_gt = sam_call_v1(batched_input, sam, dense_embeddings)
    else:
        low_res_masks, low_res_masks_gt = sam_call_v2(batched_input, sam, dense_embeddings)
    return low_res_masks, low_res_masks_gt

def sam_call_v1(batched_input, sam, dense_embeddings):
    with torch.no_grad():
        input_images = torch.stack([sam.preprocess(x["image"]) for x in batched_input], dim=0)
        image_embeddings = sam.image_encoder(input_images)
        sparse_embeddings_none, dense_embeddings_none = sam.prompt_encoder(points=None, boxes=None, masks=None)
    low_res_masks, iou_predictions = sam.mask_decoder(
        image_embeddings=image_embeddings,
        image_pe=sam.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings_none,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    return low_res_masks, None


def sam_call_v2(batched_input, sam, dense_embeddings):
    # with torch.no_grad():
    with torch.no_grad():
        input_images = [x["image"].permute((1, 2, 0)).cpu().numpy() / 255 for x in batched_input]
        sam.set_image_batch(input_images)

    fabricate_masks = False
    if fabricate_masks:
        with torch.no_grad():
            H, W = batched_input[0]['image'].shape[-2:]
            batch_size = len(batched_input)
            input_points = np.array([[[(H * (360 / 640)) // 2, W // 2]] for _ in range(batch_size)]) #  cat batch_size
            input_labels = np.array([[1] for _ in range(batch_size)]) #  cat batch_size
            low_res_masks_gt, iou_predictions, _ = sam.predict_batch(
                point_coords_batch=input_points, point_labels_batch=input_labels,
                prompt_embedding_batch=None, multimask_output=False, return_logits=True
            )
            low_res_masks_gt = torch.stack(low_res_masks_gt, dim=0)
    else:
        low_res_masks_gt = None
    low_res_masks, iou_predictions, _ = sam.predict_batch(prompt_embedding_batch=dense_embeddings,
                                                          multimask_output=False, return_logits=True)
    low_res_masks = torch.stack(low_res_masks, dim=0)
    return low_res_masks, low_res_masks_gt


def main(args=None, sam_args=None, test_run=False):
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = ModelEmb(args=args).to(device)
    if SAM_VERSION == 1:
        sam = sam_model_registry[sam_args['model_type']](checkpoint=sam_args['sam_checkpoint'])
        transform = ResizeLongestSide_sam1(sam.image_encoder.img_size)
    else:
        model_cfg = sam_args['fp_config']
        sam = SAM2ImagePredictor(build_sam2(model_cfg, sam_args['checkpoint'], device=device))
        transform = None
        assert not sam.model.training
    optimizer = optim.Adam(model.parameters(),
                           lr=float(args['learning_rate']),
                           weight_decay=float(args['WD']))
    if args['task'] == 'davsod':
        trainset, testset = get_davsod_dataset(args['root_data_dir'], sam_trans=transform)
    else:
        raise Exception('unsupported task')
    ds = torch.utils.data.DataLoader(trainset, batch_size=int(args['Batch_size']), shuffle=True,
                                     num_workers=int(args['nW']), drop_last=True)
    ds_val = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False,
                                         num_workers=int(args['nW_eval']), drop_last=False)
    best = 0
    path_best = 'results/gpu' + str(args['folder']) + '/best.csv'
    f_best = open(path_best, 'w')

    trainer = Trainer(args, test_run)
    inference_ds = InferenceDataset(args, test_run)

    for epoch in range(int(args['epoches'])):
        trainer.train_single_epoch(ds, model.train(), sam, optimizer, transform, epoch, device, accumulation_steps=args['accumulation_steps'], test_run=test_run)
        with torch.no_grad():
            IoU_val = inference_ds.inference_ds(ds_val, model.eval(), sam, transform, epoch, device)
            if IoU_val > best:
                torch.save(model, args['path_best'])
                best = IoU_val
                print('best results: ' + str(best))
                f_best.write(str(epoch) + ',' + str(best) + '\n')
                f_best.flush()
                if test_run:
                    break


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('--root_data_dir', required=True, help='root data directory')
    parser.add_argument('--sam2_size', default='large', help='root data directory')
    # parser.add_argument('--eval_dir_root', default='eval', help='root eval image directory')
    parser.add_argument('-lr', '--learning_rate', default=0.0003, help='learning_rate', required=False)
    parser.add_argument('-bs', '--Batch_size', default=3, help='batch_size', required=False)
    parser.add_argument('-epoches', '--epoches', default=200, help='number of epoches', required=False)
    parser.add_argument('-nW', '--nW', default=0, help='num workers train', required=False)
    parser.add_argument('-nW_eval', '--nW_eval', default=0, help='num workers eval', required=False)
    parser.add_argument('-WD', '--WD', default=0, help='weight decay', required=False)  # 1e-4
    parser.add_argument('-task', '--task', default='davsod', help='segmenatation task type', required=False)
    parser.add_argument('-depth_wise', '--depth_wise', default=False, help='unkown effect, model_single.py', required=False)
    parser.add_argument('-order', '--order', default=85, help='unkown effect, model_single.py', required=False)
    parser.add_argument('-Idim', '--Idim', default=256, help='image size', required=False)
    parser.add_argument('--test_run', default=True, type=bool, help='if True, stops all train / eval loops after single iteration / input', required=False)
    parser.add_argument('--accumulation_steps', default=4, type=int, help='number of accumulation steps for backwards pass', required=False)
    args = vars(parser.parse_args())

    os.makedirs('results', exist_ok=True)
    folder = open_folder('results')
    args['folder'] = folder
    args['results_root'] = os.path.join('results',
                                'gpu' + folder,
                           )
    args['path'] = os.path.join(args['results_root'],
                                'net_last.pth')
    args['path_best'] = os.path.join(args['results_root'],
                                     'net_best.pth')
    args['root_images_eval'] = os.path.join(args['results_root'],
                                     'eval_images')
    args['root_images_train'] = os.path.join(args['results_root'],
                                     'train_images')
    args['vis_folder'] = os.path.join('results', 'gpu' + args['folder'], 'vis')
    os.mkdir(args['vis_folder'])
    os.mkdir(args['root_images_eval'])
    os.mkdir(args['root_images_train'])
    if SAM_VERSION == 1:
        sam_args = {
            'sam_checkpoint': "cp_sam1/sam_vit_h.pth",
            'model_type': "vit_h",
            'generator_args': {
                'points_per_side': 8,
                'pred_iou_thresh': 0.95,
                'stability_score_thresh': 0.7,
                'crop_n_layers': 0,
                'crop_n_points_downscale_factor': 2,
                'min_mask_region_area': 0,
                'point_grids': None,
                'box_nms_thresh': 0.7,
            },
            'gpu_id': 0,
        }
    else:
        if args['sam2_size'] == 'large':
            checkpoint = "cp_   sam2/sam2.1_hiera_large.pt"
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
        # raise Exception('sam args not yet implemented for sam version 2')
    if args['test_run']:
        args['Batch_size'] = 1
    main(args=args, sam_args=sam_args, test_run=args['test_run'])
