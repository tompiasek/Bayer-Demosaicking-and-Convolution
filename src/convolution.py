import matplotlib.pyplot as plt
import numpy as np
import skimage as sk


def convolve2d(img, kernel, pad_val=0):
    result = np.zeros_like(img)
    krn_height, krn_width = kernel.shape
    pad_height, pad_width = krn_height // 2, krn_width // 2
    if len(img.shape) == 2:
        padded_image = np.pad(img, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=pad_val)

        for i in range(result.shape[0]):
            for j in range(result.shape[1]):
                result[i, j] = np.sum(padded_image[i:i+krn_height, j:j+krn_width] * kernel)

    elif len(img.shape) == 3:
        padded_red = np.pad(img[:, :, 0], ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=pad_val)
        padded_green = np.pad(img[:, :, 1], ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=pad_val)
        padded_blue = np.pad(img[:, :, 2], ((pad_height, pad_height), (pad_width, pad_width)), mode='constant', constant_values=pad_val)
        padded_image = np.dstack([padded_red, padded_green, padded_blue])
        for ch in range(result.shape[2]):
            for i in range(result.shape[0]):
                for j in range(result.shape[1]):
                    if np.sum(kernel) != 0:
                        result[i, j, ch] = np.sum(padded_image[i:i+krn_height, j:j+krn_width, ch] * kernel) / np.sum(kernel)
                    else:
                        result[i, j, ch] = np.sum(padded_image[i:i+krn_height, j:j+krn_width, ch] * kernel)
                        print("The sum of kernel: " + str(np.sum(kernel)) + "\n" +
                              "Channel: " + str(ch) + "\n")

    else:
        print("Err: Wrong image format!" + "\n" +
              "Image type: " + str(type(img)) + "\n" +
              "Image shape: " + str(img.shape))
        return img

    return result


def normalize_image(img):
    if len(img.shape) == 2:
        min_val = np.min(img)
        max_val = np.max(img)
        if max_val > 1:
            return ((img - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        else:
            return ((img - min_val) / (max_val - min_val)).astype(np.float32)
    elif len(img.shape) == 3:
        for ch in range(img.shape[2]):
            min_val = np.min(img[:, :, ch])
            max_val = np.max(img[:, :, ch])
            if max_val > 1:
                img[:, :, ch] = ((img[:, :, ch] - min_val) / (max_val - min_val) * 255).astype(np.uint8)
            else:
                img[:, :, ch] = ((img[:, :, ch] - min_val) / (max_val - min_val)).astype(np.float32)

        return img
    else:
        print("Err: Wrong image format!" + "\n" +
              "Image type: " + str(type(img)) + "\n" +
              "Image shape: " + str(img.shape))
        return img
