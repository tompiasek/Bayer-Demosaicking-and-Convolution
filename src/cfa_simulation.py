import numpy as np
import skimage as sk


def simulate_cfa(input_img, cfa_pattern='GRBG'):
    """
    Simulates the CFA (Color Filter Array) pattern on the input image

    :param input_img: Original image
    :param cfa_pattern: Pattern of the CFA
    :return: Image with simulated CFA
    """
    if len(input_img) < 1:
        print("Err: Empty image sent to the simulate_cfa func!")
        return 0

    gray_img = sk.color.rgb2gray(input_img)

    cfa_raw = np.zeros_like(gray_img)

    if cfa_pattern == 'GRBG':
        cfa_raw[::2, ::2] = input_img[::2, ::2, 1]  # Green
        cfa_raw[::2, 1::2] = input_img[::2, 1::2, 0]  # Red
        cfa_raw[1::2, ::2] = input_img[1::2, ::2, 2]  # Blue
        cfa_raw[1::2, 1::2] = input_img[1::2, 1::2, 1]  # Green
    elif cfa_pattern == 'RGGB':
        cfa_raw[::2, ::2] = input_img[::2, ::2, 0]  # Red
        cfa_raw[1::2, ::2] = input_img[1::2, ::2, 1]  # Green
        cfa_raw[::2, 1::2] = input_img[::2, 1::2, 1]  # Green
        cfa_raw[1::2, 1::2] = input_img[1::2, 1::2, 2]  # Blue
    else:
        print("Err: Unsupported CFA pattern!")
        raise ValueError("Unsupported CFA pattern")

    return cfa_raw


def simulate_cfa_3d(input_img, cfa_pattern='GRBG'):
    """
    Simulates the CFA (Color Filter Array) pattern on the input image

    :param input_img: Original image
    :param cfa_pattern: Pattern of the CFA
    :return: Image with simulated CFA
    """
    if len(input_img) < 1:
        print("Err: Empty image sent to the simulate_cfa func!")
        return 0

    # gray_img = sk.color.rgb2gray(input_img)

    cfa_raw = np.zeros_like(input_img)

    if cfa_pattern == 'GRBG':
        cfa_raw[::2, ::2, 1] = input_img[::2, ::2, 1]  # Green
        cfa_raw[::2, 1::2, 0] = input_img[::2, 1::2, 0]  # Red
        cfa_raw[1::2, ::2, 2] = input_img[1::2, ::2, 2]  # Blue
        cfa_raw[1::2, 1::2, 1] = input_img[1::2, 1::2, 1]  # Green
    elif cfa_pattern == 'RGGB':
        cfa_raw[::2, ::2, 0] = input_img[::2, ::2, 0]  # Red
        cfa_raw[1::2, ::2, 1] = input_img[1::2, ::2, 1]  # Green
        cfa_raw[::2, 1::2, 1] = input_img[::2, 1::2, 1]  # Green
        cfa_raw[1::2, 1::2, 2] = input_img[1::2, 1::2, 2]  # Blue
    else:
        print("Err: Unsupported CFA pattern!")
        raise ValueError("Unsupported CFA pattern")

    return cfa_raw
