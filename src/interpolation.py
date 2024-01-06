import matplotlib.pyplot as plt
import numpy as np


def interpolate_img(img):
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

    return np.transpose(red)


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
    return green


def interpolate_blue(blue):
    for row_b in range(len(blue)):
        if row_b % 2 == 1:
            blue[row_b] = interpolate_row(blue[row_b], data_start=0, data_step=2)

    blue = np.transpose(blue)

    for col_b in range(len(blue)):
        blue[col_b] = interpolate_row(blue[col_b], data_start=1, data_step=2)

    return np.transpose(blue)
