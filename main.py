import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import skimage as sk
from src.extractions import extract_red, extract_green, extract_blue, merge_channels
import src.kernels as krn
import src.interpolation as interp


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
        cfa_raw[::2, ::2] = input_img[::2, 1::2, 1]  # Green
        cfa_raw[::2, 1::2] = input_img[::2, ::2, 0]  # Red
        cfa_raw[1::2, ::2] = input_img[1::2, 1::2, 2]  # Blue
        cfa_raw[1::2, 1::2] = input_img[1::2, ::2, 1]  # Green
    elif cfa_pattern == 'RGGB':
        cfa_raw[::2, ::2] = gray_img[::2, ::2]  # Red
        cfa_raw[1::2, ::2] = gray_img[1::2, ::2]  # Green
        cfa_raw[::2, 1::2] = gray_img[::2, 1::2]  # Green
        cfa_raw[1::2, 1::2] = gray_img[1::2, 1::2]  # Blue
    else:
        print("Err: Unsupported CFA pattern!")
        raise ValueError("Unsupported CFA pattern")

    return cfa_raw


if __name__ == '__main__':

    fig, ax = plt.subplots(2, 2, figsize=(12, 8))

    ax_base = ax[0][0]
    ax_red = ax[0][1]
    ax_green = ax[1][0]
    ax_blue = ax[1][1]

    image = sk.io.imread("img/namib.jpg")  # Load image
    image = image[:, :, :3]  # Remove alpha channel if exists

    print(image.shape)

    print("Blue min: " + str(np.min(image[:, :, 2])))
    print("Blue max: " + str(np.max(image[:, :, 2])))

    image = simulate_cfa(image)  # Simulate CFA


    print(image.shape)

    red = extract_red(image)
    green = extract_green(image)
    blue = extract_blue(image)

    red = interp.interpolate_red(red)
    green = interp.interpolate_green(green)
    blue = interp.interpolate_blue(blue)

    # Print channel data
    print("Red channel: " + str(red.shape))
    print("Red min: " + str(np.min(red)))
    print("Red max: " + str(np.max(red)))
    print("Green channel: " + str(green.shape))
    print("Green min: " + str(np.min(green)))
    print("Green max: " + str(np.max(green)))
    print("Blue channel: " + str(blue.shape))
    print("Blue min: " + str(np.min(blue)))
    print("Blue max: " + str(np.max(blue)))

    # Custom colormaps
    reds = [(0, 0, 0), (1, 0, 0)]
    cm_r = LinearSegmentedColormap.from_list('red', reds, N=20)
    greens = [(0, 0, 0), (0, 1, 0)]
    cm_g = LinearSegmentedColormap.from_list('green', greens, N=20)
    blues = [(0, 0, 0), (0, 0, 1)]
    cm_b = LinearSegmentedColormap.from_list('blue', blues, N=20)

    final_img = np.dstack([red, green, blue])
    print("Final image: " + str(final_img.shape))

    ax_base.imshow(image)
    ax_red.imshow(red, cmap=cm_r)
    ax_green.imshow(green, cmap=cm_g)
    ax_blue.imshow(final_img)

    plt.show()
