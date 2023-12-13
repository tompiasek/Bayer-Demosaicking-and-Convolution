import numpy as np


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
