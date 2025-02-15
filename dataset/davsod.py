import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
import torch
import cv2


class DAVSODDataset(data.Dataset):
    """
    DataLoader for DAVSOD dataset for video salient object detection (VSOD)
    """

    def __init__(self, dir_root, trainsize=352, augmentations=None, train=True, sam_trans=None):
        self.trainsize = trainsize
        self.augmentations = augmentations
        # self.images = [os.path.join(image_root, f) for f in os.listdir(image_root) if f.endswith(('.jpg', '.png'))]
        # self.gts = [os.path.join(gt_root, f) for f in os.listdir(gt_root) if f.endswith('.png')]
        self.video_dirs = [os.path.join(dir_root, d) for d in os.listdir(dir_root) if os.path.isdir(os.path.join(dir_root, d))]

        self.images = []
        self.gts = []
        for video_dir in self.video_dirs:
            img_dir = os.path.join(video_dir, "Imgs")
            gt_dir = os.path.join(video_dir, "GT_object_level")
            img_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.png')])
            gt_files = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.png')])
            self.images.extend(img_files)
            self.gts.extend(gt_files)

        self.images.sort()
        self.gts.sort()
        self.filter_files()
        self.size = len(self.images)
        self.train = train
        self.sam2_trans = sam_trans

    def __getitem__(self, index):
        image = self.cv2_loader(self.images[index], is_mask=False)
        gt = self.cv2_loader(self.gts[index], is_mask=True)
        img, mask = self.augmentations(image, gt)
        original_size = tuple(img.shape[1:3])
        img, mask = self.sam2_trans.apply_image_torch(img), self.sam2_trans.apply_image_torch(mask)
        mask[mask > 0.5] = 1
        mask[mask <= 0.5] = 0
        image_size = tuple(img.shape[1:3])
        return self.sam2_trans.preprocess(img), self.sam2_trans.preprocess(mask), torch.Tensor(
            original_size), torch.Tensor(image_size)

    def filter_files(self):
        """Ensure that only images with corresponding ground truth masks are used"""
        assert len(self.images) == len(self.gts), "Mismatch between images and ground truths"
        valid_images, valid_gts = [], []
        for img_path, gt_path in zip(self.images, self.gts):
            img = Image.open(img_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                valid_images.append(img_path)
                valid_gts.append(gt_path)
        self.images, self.gts = valid_images, valid_gts

    def cv2_loader(self, path, is_mask):
        """Load images and masks using OpenCV"""
        if is_mask:
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            img[img > 0] = 1  # Binarize mask
        else:
            img = cv2.cvtColor(cv2.imread(path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return img

    def __len__(self):
        return self.size


def get_davsod_dataset(root_dir, sam_trans=None):
    """Load training and testing datasets for DAVSOD"""
    transform_train, transform_test = get_davsod_transform()
    dir_root_train = os.path.join(root_dir, 'DAVSOD/Training Set/images/')
    ds_train = DAVSODDataset(dir_root_train, augmentations=transform_train, sam_trans=sam_trans)
    dir_root_val = os.path.join(root_dir, 'DAVSOD/Validation Set/masks/')
    ds_test = DAVSODDataset(dir_root_val, train=False, augmentations=transform_test, sam_trans=sam_trans)
    return ds_train, ds_test
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