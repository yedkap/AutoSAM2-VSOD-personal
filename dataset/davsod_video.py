import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2
from dataset.tfs import DAVSODTransformVideo


class DAVSODDataset(data.Dataset):
    """
    DataLoader for DAVSOD dataset for video salient object detection (VSOD)
    """

    def __init__(self, dir_root, train=True, sam_trans=None, cutoff=None, len_seq=4, is_eval=False, frame_skip=1):
        self.dir_root = dir_root
        self.len_seq = len_seq
        # self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(('.jpg', '.png'))]
        # self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.video_dirs = [os.path.join(dir_root, d) for d in os.listdir(dir_root) if os.path.isdir(os.path.join(dir_root, d))]
        self.frame_skip = frame_skip

        self.video_seqs = []
        for video_dir in self.video_dirs:
            img_dir = os.path.join(video_dir, "Imgs")
            gt_dir = os.path.join(video_dir, "GT_object_level")
            img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
            mask_files = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.png')])
            for img_file, gt_file in zip(img_files, mask_files):
                assert os.path.basename(img_file) == os.path.basename(gt_file)

            if cutoff is not None:
                img_files = img_files[:(cutoff * self.frame_skip)]
                mask_files = mask_files[:(cutoff * self.frame_skip)]
            self.video_seqs.append({'imgs': img_files, 'masks': mask_files})

        # self.filter_files()
        self.size = len(self.video_seqs)
        self.train = train
        self.sam2_trans = sam_trans
        self.im_h = 360
        self.im_w = 640
        self.is_eval = is_eval
        self.augmentations = DAVSODTransformVideo(is_eval=is_eval)

    def pad_to_square(self, x: torch.Tensor, is_mask: bool = False) -> torch.Tensor:
        """pad to a square input."""
        h, w = x.shape[-2:]
        assert h == self.im_h and w == self.im_w
        if self.im_h > self.im_w:
            padw = self.im_h - w
            padh = 0
        else:
            padw = 0
            padh = self.im_w - h
        x = torch.nn.functional.pad(x, (0, padw, 0, padh), )
        return x

    def __getitem__(self, index):
        video = self.video_seqs[index]
        len_video = len(video['imgs'])
        if self.len_seq and self.len_seq < np.inf:
            assert len_video >= self.len_seq
            len_seq = self.len_seq
            idx_start = np.random.randint(0, len_video - (len_seq * self.frame_skip) + 1)
        else:
            len_seq = len_video // self.frame_skip
            idx_start = 0

        imgs, masks = [], []
        original_sizes, image_sizes = [], []
        self.augmentations.set_rand_params()
        for ii in range(len_seq):
            img_path, gt_path = video['imgs'][idx_start + (ii * self.frame_skip)], video['masks'][idx_start + (ii * self.frame_skip)]
            image = self.cv2_loader(img_path, is_mask=False)
            mask = self.cv2_loader(gt_path, is_mask=True)

            img = self.augmentations.transform(image, is_mask=False) * 255
            mask = self.augmentations.transform(mask * 255, is_mask=True)

            original_sizes.append(img.shape[-2:])
            if self.sam2_trans is not None:
                img, mask = self.sam2_trans.apply_image_torch(img), self.sam2_trans.apply_image_torch(mask)
            mask[mask > 0.5] = 1
            mask[mask <= 0.5] = 0
            image_size = img.shape[-2:]
            image_sizes.append(image_size)
            if self.sam2_trans is not None:
                imgs.append(self.sam2_trans.preprocess(img))
                masks.append(self.sam2_trans.preprocess(mask))
            else:
                imgs.append(self.pad_to_square(img))
                masks.append(self.pad_to_square(mask, is_mask=True))
        imgs = torch.stack(imgs, dim=0)
        masks = torch.stack(masks, dim=0)
        original_sizes = torch.tensor(original_sizes)
        image_sizes = torch.tensor(image_sizes)

        assert torch.all(original_sizes == original_sizes[0])
        assert torch.all(image_sizes == image_sizes[0])

        return imgs, masks, original_sizes, image_sizes

    @staticmethod
    def cv2_loader(path, is_mask):
        """Load images and masks using OpenCV"""
        if is_mask:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img[img > 0] = 1  # Binarize mask
        else:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img

    def __len__(self):
        return self.size


def get_davsod_dataset(root_dir, sam_trans=None, cutoff_eval=None, len_seq=4, frame_skip_train=4, frame_skip_eval=4):
    """Load training and testing datasets for DAVSOD as sequences"""
    dir_root_train = os.path.join(root_dir, 'DAVSOD/Training Set/')
    ds_train = DAVSODDataset(dir_root_train, sam_trans=sam_trans, len_seq=len_seq, is_eval=False, frame_skip=frame_skip_train)
    dir_root_val = os.path.join(root_dir, 'DAVSOD/Validation Set/')
    ds_val = DAVSODDataset(dir_root_val, train=False, sam_trans=sam_trans, cutoff=cutoff_eval, len_seq=np.inf, is_eval=True, frame_skip=frame_skip_eval)
    return ds_train, ds_val


def get_davsod_dataset_test(root_dir, sam_trans=None, cutoff_eval=None):
    """Load training and testing datasets for DAVSOD as sequences"""
    dir_root_test = os.path.join(root_dir, 'DAVSOD/Easy-35/')
    # dir_root_test = os.path.join(root_dir, 'DAVSOD/Difficult-20/')
    # dir_root_test = os.path.join(root_dir, 'DAVSOD/Normal-25/')
    ds_test = DAVSODDataset(dir_root_test, train=False, sam_trans=sam_trans, cutoff=cutoff_eval, len_seq=np.inf, is_eval=True)
    return ds_test
#
#
# def get_tests_davsod_dataset(sam_trans):
#     """Load DAVSOD test datasets from different sources"""
#     transform_train, transform_test = get_davsod_transform()
#
#     datasets = {}
#     test_sets = ['DAVSOD-Easy', 'DAVSOD-Medium', 'DAVSOD-Hard']
#     for test_set in test_sets:
#         image_root = f'DAVSOD/TestDataset/{test_set}/images/'
#         gt_root = f'DAVSOD/TestDataset/{test_set}/masks/'
#         datasets[test_set] = DAVSODDataset(image_root, gt_root, augmentations=transform_test, train=False,
#                                            sam_trans=sam_trans)
#
#     return datasets
