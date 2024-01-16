import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import skimage as sk
from src.utilities import print_channels_data, compare_with_original
from src.extractions import extract_red, extract_green, extract_blue
from src.cfa_simulation import simulate_cfa, simulate_cfa_3d
import src.interpolation as interp
from src.convolution import convolve2d, normalize_image
import cv2


def demosaic_bayer_interp(img) -> np.ndarray:
    """
    Demosaic the image using bi-linear interpolation
    :param img: Image to demosaic -> ndarray[H, W, 3]
    :return: Demosaiced image -> ndarray[H, W, 3]
    """
    # Separate color channels
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
        red = extract_red(img)
        green = extract_green(img)
        blue = extract_blue(img)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
    else:
        print("Err: Wrong image format!" + "\n" +
              "Image type: " + str(type(img)) + "\n" +
              "Image shape: " + str(img.shape))
        return img

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
    :param img: Image to demosaic -> ndarray[H, W, 3], ndarray[H, W]
    :param kernel: Kernel array for convolution
    :return: Demosaiced image -> ndarray[H, W, 3]
    """
    # Separate color channels
    if len(img.shape) == 2 or len(img.shape) == 3 and img.shape[2] == 1:
        red = extract_red(img)
        green = extract_green(img)
        blue = extract_blue(img)
    elif len(img.shape) == 3 and img.shape[2] == 3:
        red = img[:, :, 0]
        green = img[:, :, 1]
        blue = img[:, :, 2]
    else:
        print("Err: Wrong image format!" + "\n" +
              "Image type: " + str(type(img)) + "\n" +
              "Image shape: " + str(img.shape))
        return img

    # Convolve each channel with the demosaicing kernel
    red_convolved = convolve2d(red, kernel)
    if np.max(green) > 1:
        green_convolved = convolve2d(green, kernel/2)
    else:
        green_convolved = convolve2d(green, kernel)
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

    image_test = [[255, 255, 255, 255, 255, 255, 255, 255, 255],
                  [255, 255, 255, 255, 128, 255, 255, 255, 255],
                  [255, 255, 255, 48 , 64 , 48 , 255, 255, 255],
                  [255, 255, 48 , 32 , 0  , 32 , 48 , 255, 255],
                  [255, 128, 64 , 0  , 0  , 0  , 64 , 128, 255],
                  [255, 255, 48 , 32 , 0  , 32 , 48 , 255, 255],
                  [255, 255, 255, 48 , 64 , 48 , 255, 255, 255],
                  [255, 255, 255, 255, 128, 255, 255, 255, 255],
                  [255, 255, 255, 255, 255, 255, 255, 255, 255]]

    # original_img = image
    # image_interp = simulate_cfa_3d(image)  # Simulate CFA

    # image_interp = demosaic_bayer_interp(image_interp)  # Demosaic image
    image_interp = sk.io.imread("img/saved/Demosaicing/interp_demosaic_min.jpg")

    """ DEMOSAICING - CONVOLUTION """

    image_conv = simulate_cfa_3d(image)  # Simulate CFA

    # Demosaicing kernel (e.g., bi-linear)
    kernel = np.array([[0.25, 0.5, 0.25],
                       [0.5, 1, 0.5],
                       [0.25, 0.5, 0.25]], dtype='float64')

    kernel_d = np.array([[1, 1], [1, 1]]).astype('float64')
    kernel /= np.sum(kernel)

    # Perform demosaicing
    # final_conv_img = demosaic_bayer_conv(image_conv, kernel)
    # final_conv_img_d = demosaic_bayer_conv(image_conv, kernel_d)

    # Read demosaiced images from files to save time
    final_conv_img = sk.io.imread("img/saved/Demosaicing/conv_demosaic_bilinear_min.jpg")
    final_conv_img_d = sk.io.imread("img/saved/Demosaicing/conv_demosaic_midkernel_min.jpg")

    """ COMPARE DIFFERENT DEMOSAICING TECHNIQUES

    full_res_img = sk.io.imread("img/test_min.jpg")
    # full_res_img = sk.transform.resize(full_res_img, (len(image_conv), len(image_conv[0])))


    print("Interpolation MSE: " + str(np.round(sk.metrics.mean_squared_error(full_res_img, image_interp), 4)) + "\n" +
          "Convolution (bi-linear) MSE: " + str(np.round(sk.metrics.mean_squared_error(full_res_img, final_conv_img), 4)) + "\n" +
          "Convolution (averaging) MSE: " + str(np.round(sk.metrics.mean_squared_error(full_res_img, final_conv_img_d), 4)))

    interp_compare = np.round(compare_with_original(full_res_img, image_interp), 4)
    conv_compare = np.round(compare_with_original(full_res_img, final_conv_img), 4)
    conv_compare_d = np.round(compare_with_original(full_res_img, final_conv_img_d), 4)

    print("Interpolation PSNR: " + str(interp_compare[0]) + "\n" +
          "Convolution (bi-linear) PSNR: " + str(conv_compare[0]) + "\n" +
          "Convolution (averaging) PSNR: " + str(conv_compare_d[0]))

    print("Interpolation SSIM: " + str(interp_compare[1]) + "\n" +
          "Convolution (bi-linear) SSIM: " + str(conv_compare[1]) + "\n" +
          "Convolution (averaging) SSIM: " + str(conv_compare_d[1]))
    
    """

    """ COMPARE INTERPOLATION AND CONVOLUTION 
    sim_interp_img = simulate_cfa_3d(image_interp)  # Simulate CFA
    sim_conv_img = simulate_cfa_3d(final_conv_img)  # Simulate CFA

    interp_mse = sk.metrics.mean_squared_error(image_conv, sim_interp_img)
    conv_mse = sk.metrics.mean_squared_error(image_conv, sim_conv_img)

    print("Interpolation MSE: " + str(interp_mse))
    print("Convolution MSE: " + str(conv_mse))

    """

    """ PLOTTING """

    # Custom color-maps for plotting the channels
    reds = np.array([(0, 0, 0), (1, 0, 0)])
    greens = np.array([(0, 0, 0), (0, 1, 0)])
    blues = np.array([(0, 0, 0), (0, 0, 1)])
    cm_r = LinearSegmentedColormap.from_list('red', reds, N=20)
    cm_g = LinearSegmentedColormap.from_list('green', greens, N=20)
    cm_b = LinearSegmentedColormap.from_list('blue', blues, N=20)


    # # Plotting the channels
    # _, ax = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'left': 0.03, 'right': 0.97, 'top': 0.96, 'bottom': 0.04, 'wspace': 0.11, 'hspace': 0.2})
    #
    # ax[0][0].imshow(image_interp)  # Demosaiced image
    # ax[0][0].set_title('Demosaiced image')
    # ax[0][1].imshow(image_interp[:, :, 0], cmap=cm_r)  # Red channel
    # ax[0][1].set_title('Red channel')
    # ax[1][0].imshow(image_interp[:, :, 1], cmap=cm_g)  # Green channel
    # ax[1][0].set_title('Green channel')
    # ax[1][1].imshow(image_interp[:, :, 2], cmap=cm_b)  # Blue channel
    # ax[1][1].set_title('Blue channel')
    """
    # Plotting comparison between original and interpolation
    _, ax2 = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'left': 0.03, 'right': 0.97, 'wspace': 0.11})

    ax2[0].imshow(image_conv)
    ax2[0].set_title('Original Bayer Image')
    ax2[1].imshow(image_interp)
    ax2[1].set_title('Demosaiced Image (Interpolation)')

    # Plotting comparison between original and convolution
    _, ax3 = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'left': 0.03, 'right': 0.97, 'wspace': 0.11})

    ax3[0].imshow(image_conv)
    ax3[0].set_title('Original Bayer Image')
    ax3[1].imshow(final_conv_img)
    ax3[1].set_title('Demosaiced Image (Convolution)')
    """

    # Plotting comparison between convolution and interpolation
    _, ax4 = plt.subplots(1, 2, figsize=(12, 6), gridspec_kw={'left': 0.03, 'right': 0.97, 'wspace': 0.11})

    # ax4[0].imshow(image_interp)
    # ax4[0].set_title('Demosaiced Image (Interpolation)')
    # ax4[1].imshow(final_conv_img)
    # ax4[1].set_title('Demosaiced Image (Convolution)')

    # ax4[0].imshow(final_conv_img)
    # ax4[0].set_title('Demosaiced Image (bi-linear)')
    # ax4[1].imshow(final_conv_img_d)
    # ax4[1].set_title('Demosaiced Image (averaging)')

    A = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    """ CONVOLUTION KERNELS """
    kernel_edge = np.array([[0, 1, 0],
                            [1, -4, 1],
                            [0, 1, 0]])
    kernel_sobel_x = np.array([[1, 0, -1],
                               [2, 0, -2],
                               [1, 0, -1]], dtype='float64')
    kernel_sobel_y = np.array([[1, 2, 1],
                               [0, 0, 0],
                               [-1, -2, -1]], dtype='float64')
    kernel_mid = (np.array([[1, 1, 1],
                           [1, 1, 1],
                           [1, 1, 1]]) / 9).astype('float64')
    kernel_gauss = np.array([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype='float64')
    kernel_gauss /= np.sum(kernel_gauss)
    kernel_gauss_5x5 = np.array([[1, 4, 6, 4, 1],
                                [4, 16, 24, 16, 4],
                                [6, 24, 36, 24, 6],
                                [4, 16, 24, 16, 4],
                                [1, 4, 6, 4, 1]], dtype='float64')
    kernel_gauss_5x5 /= np.sum(kernel_gauss_5x5)
    kernel_sharp = np.array([[0, -1, 0],
                            [-1, 5, -1],
                            [0, -1, 0]])

    nyc_test_img = sk.io.imread("img/edge_test_nyc.png")[:, :, :3]

    grayscale_conv_int = np.array(sk.color.rgb2gray(nyc_test_img) * 255, dtype='uint8')
    grayscale_conv_float = sk.color.rgb2gray(nyc_test_img)

    # Sobel edge detection (X and Y)
    sobel_x_img = np.asarray(convolve2d(np.asarray(grayscale_conv_int).astype('float64') / 255, kernel_sobel_x)).astype(np.float64)
    sobel_y_img = np.asarray(convolve2d(np.asarray(grayscale_conv_int).astype('float64') / 255, kernel_sobel_y)).astype(np.float64)

    # Sobel filter (detector (X + Y))
    fake_sobel = sobel_x_img + sobel_y_img
    fake_sobel /= np.max(fake_sobel)
    fake_sobel = np.clip(fake_sobel, 0, 1)

    # Sobel-Feldman filter
    sobel_edge_img = np.asarray(np.sqrt(sobel_x_img * sobel_x_img + sobel_y_img * sobel_y_img)).astype('float64')
    if np.max(sobel_edge_img) > 1:
        sobel_edge_img /= np.max(sobel_edge_img)


    # not_clipped = convolve2d(nyc_test_img, kernel_edge)
    # clipped = np.clip(not_clipped, 0, 255)
    # clipped_uint8 = clipped.astype(np.uint8)

    # ax4[0].imshow(sobel_edge_img, cmap='gray')
    # ax4[0].set_title('Edge detection (Sobel filter)')
    # ax4[1].imshow(np.clip(convolve2d(grayscale_conv_float, kernel_edge)*3, 0, 1), cmap='gray')
    # ax4[1].set_title('Edge detection (Laplace filter (amplified x3)')

    # DISPLAY BASIC EDGE DETECTION (LAPLACE FILTER) WITH CHANNELS
    # _, ax5 = plt.subplots(2, 2, figsize=(14, 10), gridspec_kw={'left': 0.03, 'right': 0.97, 'top': 0.96, 'bottom': 0.04, 'wspace': 0.11, 'hspace': 0.2})
    #
    # ax5[0][0].imshow(not_clipped)
    # ax5[0][0].set_title('Edge detection (bi-linear)')
    # ax5[0][1].imshow(not_clipped[:, :, 0], cmap=cm_r)
    # ax5[0][1].set_title('Red channel')
    # ax5[1][0].imshow(not_clipped[:, :, 1], cmap=cm_g)
    # ax5[1][0].set_title('Green channel')
    # ax5[1][1].imshow(not_clipped[:, :, 2], cmap=cm_b)
    # ax5[1][1].set_title('Blue channel')

    # Save images
    # sk.io.imsave("img/saved/interp_demosaic_min_defects.jpg", np.ndarray.astype(image_interp[400:630, 65:500, :], np.uint8))
    # sk.io.imsave("img/saved/conv_demosaic_bilinear_min_defects.jpg", np.ndarray.astype(final_conv_img[400:630, 65:500, :], np.uint8))
    # sk.io.imsave("img/saved/conv_demosaic_midkernel_min_defects.jpg", np.ndarray.astype(final_conv_img_d[400:630, 65:500, :], np.uint8))

    plt.show()
