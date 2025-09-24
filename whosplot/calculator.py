
import numpy as np

def calculator(method, *args, **kwargs):
    """
    Selects and executes a math method based on the given method name.

    Parameters:
    method : str
        The name of the math method to use.
    *args : tuple
        Positional arguments passed to the math method.
    **kwargs : dict
        Keyword arguments passed to the math method.
    """
    method_dict = {
        'var': __var,
        'std': __std,
        'mean': __mean,
        'max': __max,
        'min': __min,
        'mmnorm': __mmnorm,
        'znorm': __znorm,
        'dist2d': __dist2d,
        'dist3d': __dist3d,
        'dist': __dist,
        'norm_2d': __norm_2d,
        'of_time_average': __of_time_average
    }
    return method_dict[method](*args, **kwargs)


def __var(data: np.ndarray, axis: int):
    """
    calculate the variance of the data.
    :param data:
    :param axis:
    :return:
    """
    if axis == 1:
        row_num, col_num = data.shape
        return np.array([np.var(data[:, num]) for num in range(col_num)]).reshape(1, col_num)
    elif axis == 0:
        row_num, col_num = data.shape
        return np.array([np.var(data[num, :]) for num in range(row_num)]).reshape(row_num, 1)

def __std(data: np.ndarray, axis: int):
    """
    calculate the standard deviation of the data.
    :param data:
    :param axis:
    :return:
    """
    if axis == 1:
        row_num, col_num = data.shape
        return np.array([np.std(data[:, num]) for num in range(col_num)]).reshape(1, col_num)
    elif axis == 0:
        row_num, col_num = data.shape
        return np.array([np.std(data[num, :]) for num in range(row_num)]).reshape(row_num, 1)

def __mean(data: np.ndarray, axis: int):
    """
    calculate the mean of the data.
    :param data:
    :param axis:
    :return:
    """
    if axis == 1:
        row_num, col_num = data.shape
        return np.array([np.mean(data[:, num]) for num in range(col_num)]).reshape(1, col_num)
    elif axis == 0:
        row_num, col_num = data.shape
        return np.array([np.mean(data[num, :]) for num in range(row_num)]).reshape(row_num, 1)

def __max(data: np.ndarray, axis: int):
    """
    calculate the max of the data.
    :param data:
    :param axis:
    :return:
    """
    if axis == 1:
        row_num, col_num = data.shape
        return np.array([np.max(data[:, num]) for num in range(col_num)]).reshape(1, col_num)
    elif axis == 0:
        row_num, col_num = data.shape
        return np.array([np.max(data[num, :]) for num in range(row_num)]).reshape(row_num, 1)

def __min(data: np.ndarray, axis: int):
    """
    calculate the min of the data.
    :param data:
    :param axis:
    :return:
    """
    if axis == 1:
        row_num, col_num = data.shape
        return np.array([np.min(data[:, num]) for num in range(col_num)]).reshape(1, col_num)
    elif axis == 0:
        row_num, col_num = data.shape
        return np.array([np.min(data[num, :]) for num in range(row_num)]).reshape(row_num, 1)

def __mmnorm(data: np.ndarray, axis: int):
    """
    normalize the data.
    :param data:
    :param axis:
    :return:
    """
    if axis == 1:
        row_num, col_num = data.shape
        max_ = __max(data, axis)
        min_ = __min(data, axis)
        return np.array([(data[:, num] - min_[:, num]) / (max_[:, num] - min_[:, num]) for num in range(col_num)]).T
    elif axis == 0:
        row_num, col_num = data.shape
        max_ = __max(data, axis)
        min_ = __min(data, axis)
        return np.array([(data[num, :] - min_[num, :]) / (max_[num, :] - min_[num, :]) for num in range(row_num)])

def __znorm(data: np.ndarray, axis: int):
    """
    normalize the data.
    :param data:
    :param axis:
    :return:
    """
    if axis == 1:
        row_num, col_num = data.shape
        mean_ = __mean(data, axis)
        std_ = __std(data, axis)
        return np.array([(data[:, num] - mean_[:, num]) / std_[:, num] for num in range(col_num)]).T
    elif axis == 0:
        row_num, col_num = data.shape
        mean_ = __mean(data, axis)
        std_ = __std(data, axis)
        return np.array([(data[num, :] - mean_[num, :]) / std_[num, :] for num in range(row_num)])

def __dist2d(x, y):
    """

    :param x:
    :param y:
    :return:
    """

    return np.sqrt(np.square(x) + np.square(y))

def __dist3d(x, y, z):
    """

    :param x:
    :param y:
    :param z:
    :return:
    """

    return np.sqrt(np.square(x) + np.square(y) + np.square(z))

def __norm_2d(data: np.ndarray):
    """

    :param x:
    :param y:
    :return:
    """

    return np.linalg.norm(data) / np.sqrt(data.size)

def __dist(data: np.ndarray, axis: int, power: int):
    """

    :param data:
    :param axis:
    :return:
    """
    if axis == 1:
        row_num, col_num = data.shape
        return np.array([np.power(np.sum(np.power(data[:, num], power)), 1 / power) for num in range(col_num)]).reshape(1, col_num)
    elif axis == 0:
        row_num, col_num = data.shape
        return np.array([np.power(np.sum(np.power(data[num, :], power)), 1 / power) for num in range(row_num)]).reshape(row_num, 1)
    
def __of_time_average(data: np.ndarray, axis: int):
    """
    weighted average of the data.
    :param data:
    :param axis:
    :return:
    """
    if axis == 1:
        row_num, col_num = data.shape
        diff = np.diff(data[:, 0], axis=0).reshape(row_num - 1, 1)
        pred_last = __mean(diff, axis=axis).reshape(1, 1)
        weight = np.insert(diff, row_num-1, pred_last, axis=0)
        return np.array([np.sum(data[:, num] * weight[:, 0]) / np.sum(weight[:, 0]) for num in range(col_num)]).reshape(1, col_num)

    elif axis == 0:
        row_num, col_num = data.shape
        diff = np.diff(data[0, :], axis=0).reshape(1, col_num - 1)
        pred_last = __mean(diff, axis=axis).reshape(1, 1)
        weight = np.insert(diff, col_num-1, pred_last, axis=1)
        return np.array([np.sum(data[num, :] * weight[0, :]) / np.sum(weight[0, :]) for num in range(row_num)]).reshape(row_num, 1)