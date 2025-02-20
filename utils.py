import matplotlib.pyplot as plt
import numpy as np

def save_image(tensor, path, is_mask=False):
    """Utility function to save a tensor as an image."""
    img = tensor.squeeze().detach().cpu().numpy() # Convert to float
    if is_mask:
        plt.imsave(path, img.astype(np.float32), cmap='gray')
    else:
        if img.ndim == 3 and img.shape[0] in [1, 3]:
            img = np.transpose(img, (1, 2, 0)) / 255  # Convert from (C, H, W) to (H, W, C)
        plt.imsave(path, img)
