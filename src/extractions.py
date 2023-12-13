import numpy as np


def extract_green(img):
    """
    Extracts green channel from the image

    :param img: Original image
    :return: Green channel of the image
    """
    if len(img) < 1:
        print("Err: Empty image sent to the extract_green func!")
        return 0

    # green_img = img[::2, ::2] + img[1::2, 1::2]

    green_img = np.zeros((len(img), len(img[0])))

    for row in range(len(img)):
        for col in range(len(img[row])):
            if (col % 2 == 0 and row % 2 == 0) or (col % 2 == 1 and row % 2 == 1):
                green_img[row][col] = img[row][col]
            else:
                green_img[row][col] = 0

    # green_img = np.zeros((len(img), len(img[0])))
    #
    # for row in range(len(img)):
    #     for col in range(len(img[row])):
    #         if (col % 2 == 0 and row % 2 == 0) or (col % 2 == 1 and row % 2 == 1):
    #             green_img[row][col] = img[row][col]
    #         else:
    #             green_img[row][col] = 0

    return green_img


def extract_red(img):
    """
    Extracts red channel from the image

    :param img: Original image
    :return: Red channel of the image
    """
    if len(img) < 1:
        print("Err: Empty image sent to the extract_red func!")
        return 0

    # red_img = img[::2, 1::2]

    red_img = np.zeros((len(img), len(img[0])))

    for row in range(len(img)):
        for col in range(len(img[row])):
            if col % 2 == 1 and row % 2 == 0:
                red_img[row][col] = img[row][col]
            else:
                red_img[row][col] = 0

    return red_img


def extract_blue(img):
    """
    Extracts red channel from the image

    :param img: Original image
    :return: Red channel of the image
    """
    if len(img) < 1:
        print("Err: Empty image sent to the extract_red func!")
        return 0

    # blue_img = img[1::2, 1::2]

    blue_img = np.zeros((len(img), len(img[0])))

    for row in range(len(img)):
        for col in range(len(img[row])):
            if col % 2 == 0 and row % 2 == 1:
                blue_img[row][col] = img[row][col]
            else:
                blue_img[row][col] = 0

    return blue_img


def merge_channels(red, green, blue):
    """
    Merges three channels into one image

    :param red: Red channel
    :param green: Green channel
    :param blue: Blue channel
    :return: Image with three channels
    """
    if len(red) < 1 or len(green) < 1 or len(blue) < 1:
        print("Err: One of the channels is empty!")
        return 0

    if len(red) != len(green) or len(red) != len(blue):
        print("Err: Channels have different sizes!")
        return 0

    result_img = np.zeros((len(red), len(red[0]), 3))

    for row in range(len(red)):
        for col in range(len(red[row])):
            result_img[row][col][0] = red[row][col]
            result_img[row][col][1] = green[row][col]
            result_img[row][col][2] = blue[row][col]

    return result_img
