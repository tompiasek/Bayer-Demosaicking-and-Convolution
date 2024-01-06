import numpy as np
import skimage as sk


def print_channels_data(r, g, b):
    """
    Print data about the channels
    :param r: Red channel
    :param g: Green channel
    :param b: Blue channel
    :return: none
    """
    print("Red channel: " + str(r.shape))
    print("Red min: " + str(np.min(r)))
    print("Red max: " + str(np.max(r)))
    print("Green channel: " + str(g.shape))
    print("Green min: " + str(np.min(g)))
    print("Green max: " + str(np.max(g)))
    print("Blue channel: " + str(b.shape))
    print("Blue min: " + str(np.min(b)))
    print("Blue max: " + str(np.max(b)))


def compare_with_original(original, compare):
    """
    Calculates PSNR between two images
    :param original: Original image
    :param compare: Image to compare
    :return: PSNR and SSIM value
    """
    # Check if images are not empty
    if len(original) < 1 or len(compare) < 1:
        print("Err: Empty image sent to the PSNR func!")
        return 0

    # Convert from gray-scale to RGB if needed
    if len(original.shape) == 2:
        original = sk.color.gray2rgb(original)
    if len(compare.shape) == 2:
        reconstructed = sk.color.gray2rgb(compare)

    # Check if images have the same shape
    if original.shape < compare.shape:
        print("Err: Original image can't be lower res than compare image!")
        return 0
    elif original.shape > compare.shape:
        original = sk.transform.resize(original, compare.shape)

    # Calculate PSNR and SSIM
    if np.max(original) > 1:
        psnr = sk.metrics.peak_signal_noise_ratio(original, compare, data_range=255)
        ssim = sk.metrics.structural_similarity(original, compare, multichannel=True, channel_axis=2, data_range=255)
    else:
        psnr = sk.metrics.peak_signal_noise_ratio(original, compare, data_range=1)
        ssim = sk.metrics.structural_similarity(original, compare, multichannel=True, channel_axis=2, data_range=1)

    return psnr, ssim


def invert(x):
    return np.sin(np.power(x, -1))


def signum(x):
    return np.sign(np.sin(8*x))


def find_closest_indexes(point, arr, amount=1):
    if len(arr) < 1:
        print("Err: array in find_closest() function can't be empty!")
        return 0

    if amount > len(arr):
        print("Err: Array given in find_closest() function is smaller than amount of searched points!")

    left_offset = 0
    right_offset = 0
    left_index = 0
    right_index = 0

    result_arr = []

    if point < arr[0]:
        for i in range(amount):
            if i <= (len(arr) - 1):
                result_arr.append(i)
        return result_arr

    if point > arr[-1]:
        for i in np.flip(range(amount)):
            if len(arr) - i <= len(arr):
                result_arr.append(len(arr) - 1 - i)
        return result_arr

    for i in range(len(arr)):
        if arr[i] == point:
            for j in np.flip(range(amount)):
                if 0 <= i - j - 1 < len(arr):
                    result_arr.append(i - j - 1)

            if arr[i] == point:
                result_arr.append(i)

            for j in range(amount):
                if i + j + 1 < len(arr):
                    if arr[i + j + 1] == point:
                        continue
                    result_arr.append(i + j + 1)

            return result_arr

        elif arr[i] > point:
            for j in np.flip(range(amount)):
                if 0 <= i - j - 1 < len(arr):
                    result_arr.append(i - j - 1)

            for j in range(amount):
                if i + j < len(arr):
                    if arr[i + j] == point:
                        continue
                    result_arr.append(i + j)

            return result_arr


def find_closest(point, arr, amount=1):
    if len(arr) < 1:
        print("Err: array in find_closest() function can't be empty!")
        return 0

    if amount > len(arr):
        print("Err: Array given in find_closest() function is smaller than amount of searched points!")

    result_arr = []

    if point < arr[0]:
        for i in range(amount):
            if i <= (len(arr) - 1):
                result_arr.append(arr[i])
        return result_arr

    if point > arr[-1]:
        for i in np.flip(range(amount)):
            if len(arr) - i <= len(arr):
                result_arr.append(arr[len(arr) - 1 - i])
        return result_arr

    for i in range(len(arr)):
        if arr[i] == point:
            for j in np.flip(range(amount)):
                if 0 <= i - j - 1 < len(arr):
                    result_arr.append(arr[i - j - 1])

            if arr[i] == point:
                result_arr.append(point)

            for j in range(amount):
                if i + j + 1 < len(arr):
                    if arr[i + j + 1] == point:
                        continue
                    result_arr.append(arr[i + j + 1])

            return result_arr

        elif arr[i] > point:
            for j in np.flip(range(amount)):
                if 0 <= i - j - 1 < len(arr):
                    result_arr.append(arr[i - j - 1])

            for j in range(amount):
                if i + j < len(arr):
                    if arr[i + j] == point:
                        continue
                    result_arr.append(arr[i + j])

            return result_arr
