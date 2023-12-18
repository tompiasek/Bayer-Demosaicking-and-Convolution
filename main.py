import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import skimage as sk
from src.utilities import print_channels_data
from src.extractions import extract_red, extract_green, extract_blue
from src.cfa_simulation import simulate_cfa, simulate_cfa_3d
import src.interpolation as interp
from src.convolution import convolve2d, normalize_image


def demosaic_bayer_interp(img) -> np.ndarray:
    """
    Demosaic the image using bilinear interpolation
    :param img: Image to demosaic
    :return: Demosaiced image -> ndarray[H, W, 3]
    """
    # Extract channels
    red = extract_red(img)
    green = extract_green(img)
    blue = extract_blue(img)

    # Interpolate channels
    red = interp.interpolate_red(red)
    green = interp.interpolate_green(green)
    blue = interp.interpolate_blue(blue)

    print_channels_data(red, green, blue)  # Print data about the channels

    result_img = np.dstack([red, green, blue])  # Merge channels
    print("Result image: " + str(result_img.shape))  # Print data about the result image

    return result_img


def demosaic_bayer_conv(img, kernel):
    """
    Demosaic the image using convolution
    :param img: Image to demosaic
    :param kernel: Kernel array for convolution
    :return: Demosaiced image -> ndarray[H, W, 3]
    """
    # Separate color channels
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]

    # Convolve each channel with the demosaicing kernel
    red_convolved = convolve2d(red, kernel)
    green_convolved = convolve2d(green, kernel/2)
    blue_convolved = convolve2d(blue, kernel)

    # Stack the convolved channels to form the demosaiced image
    result_img = np.dstack([red_convolved, green_convolved, blue_convolved])

    # Normalize values to the valid range (0-255)
    result_img = normalize_image(result_img)

    return result_img


if __name__ == '__main__':

    """ DEMOSAICING - INTERPOLATION """

    image = sk.io.imread("img/test_min.jpg")  # Load image
    image = image[:, :, :3]  # Remove alpha channel if exists

    original_img = image
    image_interp = simulate_cfa(image)  # Simulate CFA

    image_interp = demosaic_bayer_interp(image_interp)  # Demosaic image

    """ DEMOSAICING - CONVOLUTION """

    image_conv = simulate_cfa_3d(original_img)  # Simulate CFA

    # Demosaicing kernel (e.g., bilinear)
    kernel = np.array([[0.25, 0.5, 0.25],
                       [0.5, 1, 0.5],
                       [0.25, 0.5, 0.25]])

    # Normalize the kernel
    kernel /= np.sum(kernel)

    # Perform demosaicing
    final_conv_img = demosaic_bayer_conv(image_conv, kernel)

    """ PLOTTING """

    # Custom color-maps for plotting the channels
    reds = np.array([(0, 0, 0), (1, 0, 0)])
    greens = np.array([(0, 0, 0), (0, 1, 0)])
    blues = np.array([(0, 0, 0), (0, 0, 1)])
    cm_r = LinearSegmentedColormap.from_list('red', reds, N=20)
    cm_g = LinearSegmentedColormap.from_list('green', greens, N=20)
    cm_b = LinearSegmentedColormap.from_list('blue', blues, N=20)

    # Plotting the channels
    _, ax = plt.subplots(2, 2, figsize=(14, 10))

    ax[0][0].imshow(image_interp)  # Demosaiced image
    ax[0][0].set_title('Demosaiced image')
    ax[0][1].imshow(image_interp[:, :, 0], cmap=cm_r)  # Red channel
    ax[0][1].set_title('Red channel')
    ax[1][0].imshow(image_interp[:, :, 1], cmap=cm_g)  # Green channel
    ax[1][0].set_title('Green channel')
    ax[1][1].imshow(image_interp[:, :, 2], cmap=cm_b)  # Blue channel
    ax[1][1].set_title('Blue channel')

    # Plotting the images for interpolation
    _, ax2 = plt.subplots(1, 2, figsize=(12, 6))

    ax2[0].imshow(image_conv)
    ax2[0].set_title('Original Bayer Image')
    ax2[1].imshow(image_interp)
    ax2[1].set_title('Demosaiced Image (Interpolation)')

    # Plotting the images for convolution
    _, ax3 = plt.subplots(1, 2, figsize=(12, 6))

    ax3[0].imshow(image_conv)
    ax3[0].set_title('Original Bayer Image')
    ax3[1].imshow(final_conv_img)
    ax3[1].set_title('Demosaiced Image (Convolution)')

    plt.show()
