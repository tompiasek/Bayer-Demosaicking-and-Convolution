import numpy as np
import skimage as sk


def convolve2d(img, kernel):
    result = np.zeros_like(img)
    krn_height, krn_width = kernel.shape
    pad_height, pad_width = krn_height // 2, krn_width // 2
    padded_image = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    for i in range(result.shape[0]):
        for j in range(result.shape[1]):
            result[i, j] = np.sum(padded_image[i:i+krn_height, j:j+krn_width] * kernel)

    return result


def normalize_image(img):
    min_val = np.min(img)
    max_val = np.max(img)
    return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
