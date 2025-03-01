from dataset import transforms_shir as transforms
import numpy as np
from PIL import Image
import torch
import torchvision.transforms.functional as F

# from utils import *


def get_cub_transform():
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(22, scale=(0.75, 1.25)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[255*0.485, 255*0.456, 255*0.406], std=[255*0.229, 255*0.224, 255*0.225])
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[255*0.485, 255*0.456, 255*0.406], std=[255*0.229, 255*0.224, 255*0.225])
    ])
    return transform_train, transform_test


def get_glas_transform():
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((256, 256)),
        transforms.ColorJitter(brightness=0.2,
                               contrast=0.2,
                               saturation=0.2,
                               hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(5, scale=(0.75, 1.25)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[255*0.485, 255*0.456, 255*0.406], std=[255*0.229, 255*0.224, 255*0.225])
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((256, 256)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[255*0.485, 255*0.456, 255*0.406], std=[255*0.229, 255*0.224, 255*0.225])
    ])
    return transform_train, transform_test

# def get_glas_transform():
#     transform_train = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((256, 256)),
#         transforms.ColorJitter(brightness=0.2,
#                                contrast=0.2,
#                                saturation=0.2,
#                                hue=0.1),
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomAffine(5, scale=(0.75, 1.25)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[255*0.485, 255*0.456, 255*0.406], std=[255*0.229, 255*0.224, 255*0.225])
#     ])
#     transform_test = transforms.Compose([
#         transforms.ToPILImage(),
#         transforms.Resize((256, 256)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])
#     ])
#     return transform_train, transform_test


def get_monu_transform(args):
    Idim = int(args['Idim'])
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((Idim, Idim)),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(int(args['rotate']), scale=(float(args['scale1']), float(args['scale2']))),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[142.07, 98.48, 132.96], std=[65.78, 57.05, 57.78])
    ])
    transform_test = transforms.Compose([
        transforms.ToPILImage(),
        # transforms.Resize((Idim, Idim)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[142.07, 98.48, 132.96], std=[65.78, 57.05, 57.78])
    ])
    return transform_train, transform_test


def get_polyp_transform():
    transform_train = transforms.Compose([
        # transforms.Resize((352, 352)),
        transforms.ToPILImage(),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        transforms.RandomVerticalFlip(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(90, scale=(0.75, 1.25)),
        transforms.ToTensor(),
        # transforms.Normalize([105.61, 63.69, 45.67],
        #                      [83.08, 55.86, 42.59])
    ])
    transform_test = transforms.Compose([
        # transforms.Resize((352, 352)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize([105.61, 63.69, 45.67],
        #                      [83.08, 55.86, 42.59])
    ])
    return transform_train, transform_test


def add_salt_pepper_noise(img, prob=0.02):
    """
    Adds salt and pepper noise to an image.

    Args:
        img (Tensor): Image tensor (C, H, W) with values in [0, 1].
        prob (float): Probability of noise for each pixel.

    Returns:
        Tensor: Noisy image.
    """
    img_np = np.array(img)  # Convert to NumPy array
    c, h, w = img_np.shape
    mask = np.random.rand(h, w)  # Generate random mask

    salt_mask = mask < prob / 2
    pepper_mask = mask > 1 - prob / 2

    img_np[:, salt_mask] = 1  # Set salt pixels to 1
    img_np[:, pepper_mask] = 0  # Set pepper pixels to 0

    return torch.tensor(img_np, dtype=torch.float32)


def salt_and_pepper_noise(img, prob=0.02):
    """Applies salt and pepper noise."""
    np_img = np.array(img)
    noise = np.random.rand(*np_img.shape[:2])

    # Set pixels to black (pepper)
    np_img[noise < prob / 2] = 0

    # Set pixels to white (salt)
    np_img[noise > 1 - prob / 2] = 255

    return Image.fromarray(np_img)


def add_gaussian_noise(img, mean=0.0, std=0.05):
    """Applies Gaussian noise to a tensor."""
    img = np.array(img).astype(np.float32) / 255.
    noise = np.random.randn(*img.shape) * std + mean
    img_noisy = np.clip(img + noise, a_min=.0, a_max=.1)
    img_noisy = (img_noisy * 255).astype(np.uint8)
    return Image.fromarray(img_noisy)


class DAVSODTransform:
    def __init__(self):
        self.params = None  # Holds the randomly sampled parameters

    def rand_uniform(self, min=0., max=1.):
        start = float(torch.rand((1,))[0])
        rand_out = start * (max - min) + min
        return rand_out

    def set_rand_params(self):
        """Set random transformation parameters once per scene."""
        self.params = {
            "brightness_factor": self.rand_uniform(0.6, 1.4),
            "contrast_factor": self.rand_uniform(0.6, 1.4),
            "saturation_factor": self.rand_uniform(0.6, 1.4),
            "hue_factor": self.rand_uniform(-0.1, 0.1),
            "flip": self.rand_uniform() > 0.5,
            "angle": self.rand_uniform(-20, 20),
            "scale": self.rand_uniform(0.75, 1.25),
        }

    def transform(self, frame, is_mask=False):
        """
        Apply the stored transformation parameters to a frame.

        Args:
            frame: A tensor or PIL image.
            is_mask (bool): If True, skip color adjustments.
        """
        if self.params is None:
            raise ValueError("Call set_rand_params() before using transform().")
        frame = F.to_pil_image(frame)

        if not is_mask:
            frame = F.adjust_brightness(frame, self.params["brightness_factor"])
            frame = F.adjust_contrast(frame, self.params["contrast_factor"])
            frame = F.adjust_saturation(frame, self.params["saturation_factor"])
            frame = F.adjust_hue(frame, self.params["hue_factor"])

        if self.params["flip"]:
            frame = F.hflip(frame)

        frame = F.affine(frame, angle=self.params["angle"], translate=[0, 0], scale=self.params["scale"], shear=[0])
        frame = F.to_tensor(frame)

        return frame


def get_davsod_transform():
    transform_train = transforms.Compose([
        # transforms.Resize((352, 352)),
        transforms.ToPILImage(),
        # transforms.Lambda(lambda x: add_gaussian_noise(x, mean=0.0, std=0.05)),
        transforms.ColorJitter(brightness=0.4,
                               contrast=0.4,
                               saturation=0.4,
                               hue=0.1),
        # transforms.Lambda(lambda img: salt_and_pepper_noise(img, prob=0.02)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=20, scale=(0.75, 1.25)),
        transforms.ToTensor(),
        # transforms.Normalize([105.61, 63.69, 45.67],
        #                      [83.08, 55.86, 42.59])
    ])
    transform_test = transforms.Compose([
        # transforms.Resize((352, 352)),
        transforms.ToPILImage(),
        transforms.ToTensor(),
        # transforms.Normalize([105.61, 63.69, 45.67],
        #                      [83.08, 55.86, 42.59])
    ])
    return transform_train, transform_test