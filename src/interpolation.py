import numpy as np
from src.utilities import find_closest, find_closest_indexes


# def interpolate_row(row, krn):
#     kernels = []
#     space = np.linspace(0, 1, 2 * len(row))
#
#     for x, y in zip(space.tolist(), row.tolist()):
#         kernel = krn(space, offset=2 * x, width=1 / len(row))
#         # print(kernel.shape)
#         kernels.append(y * kernel)
#
#     return space, np.sum(np.asarray(kernels), axis=0)


def interpolate_img(img, krn):
    result = []
    if len(img) < 1:
        print("Err: Given image can't be empty!")
        return 0

    for row in img:
        interpolated = np.interp(range(len(row)), range(len(row)), row)
        result.append(interpolated)

    result = np.transpose(result)

    for col in range(len(result)):
        interpolated = np.interp(range(len(result[col])), range(len(result[col])), (np.asarray(result)[col]))
        result[col] = interpolated

    result = np.transpose(result)
    return np.asarray(result)


def interpolate_row(row, data_start=0, data_step=2):
    data = row[data_start::data_step]

    result = np.interp(range(len(row)), range(len(row))[data_start::data_step], data)
    return result


def interpolate_red(red):
    for row_r in range(len(red)):
        if row_r % 2 == 0:
            red[row_r] = interpolate_row(red[row_r], data_start=1, data_step=2)

    red = np.transpose(red)

    for col_r in range(len(red)):
        red[col_r] = interpolate_row(red[col_r], data_start=0, data_step=2)

    return np.transpose(red.astype(int))


def interpolate_green(green):
    for row_g in range(len(green)):
        if row_g % 2 == 0:
            green[row_g] = interpolate_row(green[row_g], data_start=0, data_step=2)
        else:
            green[row_g] = interpolate_row(green[row_g], data_start=1, data_step=2)

    green = np.transpose(green)

    for col_g in range(len(green)):
        green[col_g] = interpolate_row(green[col_g], data_start=0, data_step=2)

    green = np.transpose(green)
    return green.astype(int)


def interpolate_blue(blue):
    for row_b in range(len(blue)):
        if row_b % 2 == 1:
            blue[row_b] = interpolate_row(blue[row_b], data_start=0, data_step=2)

    blue = np.transpose(blue)

    for col_b in range(len(blue)):
        blue[col_b] = interpolate_row(blue[col_b], data_start=1, data_step=2)

    return np.transpose(blue.astype(int))


#
# def interpolate(x_arr: np.ndarray, y_arr: np.ndarray, x_result: np.ndarray, kernel, interp_range=0):
#     """
#     Interpolate data using the specified kernel
#
#     Args:
#         :param x_arr: The x-values of the original data (function).
#         :param y_arr: The y-values of the original data (function).
#         :param x_result: The x-values for interpolation.
#         :param kernel: The interpolation kernel function
#         :param interp_range: The range of points near the interpolated point on which we perform interpolation  # UPDATE
#
#     :return: numpy.ndarray: The interpolated y-values
#     """
#     if len(x_arr) < 1:
#         print("Err: x_arr can't be empty!")
#         return 0
#
#     y_result = []
#     # range_len = np.abs(x_arr[0] - x_arr[-1])  # Length of measured range
#     # distance = range_len / (len(x_result) - 1)  # Distance between two points
#
#     for i in range(len(x_result)):
#         if interp_range > 0:
#             temp_x_arr = find_closest(x_result[i], x_arr, interp_range)
#             temp_y_arr = []
#             for index in find_closest_indexes(x_result[i], x_arr, interp_range):
#                 temp_y_arr.append(y_arr[int(index)])
#         else:
#             temp_x_arr = x_arr
#             temp_y_arr = y_arr
#
#         weights = kernel(x_result[i] - temp_x_arr)
#         weights = weights.astype(float)
#         total_weight = np.sum(weights)
#
#         if total_weight != 0:
#             weights /= total_weight
#
#             y = np.sum(weights * temp_y_arr)
#             y_result.append(y)
#         else:
#             y_result.append(0)
#
#     return y_result
