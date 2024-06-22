
import os
import numpy as np
from scipy.interpolate import griddata
import platform
import cProfile
from functools import wraps

from whosplot.__init__ import \
    (__title__,
     __description__,
     __version__,
     __author__,
     __author_email__,
     __license__,
     __copyright__)


alphabet = [chr(i) for i in range(97, 123)]


def func_cprofile(f):
    """
    内建分析器
    """

    @wraps(f)
    def wrapper(*args, **kwargs):
        profile = cProfile.Profile()
        try:
            profile.enable()
            result = f(*args, **kwargs)
            profile.disable()
            return result
        finally:
            profile.print_stats(sort='time')

    return wrapper


def printwp(value):
    """

    :return:
    """
    print(value)


def printer_septal_line():
    """

    :return:
    """
    print('======================================================')


def check_recur(list_: list):
    """
    check repeated items from a list.
    :param list_:
    :return:
    """
    if len(set(list_)) < len(list_):
        return True
    else:
        return False


def calvar(data: np.ndarray, axis: int):
    """
    calculate the variance of the data.
    :param data:
    :param axis:
    :return:
    """
    if isinstance(data, np.ndarray):
        if axis == 1:
            row_num, col_num = data.shape
            return np.array([np.var(data[:, num]) for num in range(col_num)]).reshape(1, col_num)
        elif axis == 0:
            row_num, col_num = data.shape
            return np.array([np.var(data[num, :]) for num in range(row_num)]).reshape(row_num, 1)
    else:
        raise TypeError('Data type is not correct!')


def calstd(data: np.ndarray, axis: int):
    """
    calculate the standard deviation of the data.
    :param data:
    :param axis:
    :return:
    """
    if isinstance(data, np.ndarray):
        if axis == 1:
            row_num, col_num = data.shape
            return np.array([np.std(data[:, num]) for num in range(col_num)]).reshape(1, col_num)
        elif axis == 0:
            row_num, col_num = data.shape
            return np.array([np.std(data[num, :]) for num in range(row_num)]).reshape(row_num, 1)
    else:
        raise TypeError('Data type is not correct!')


def calmean(data: np.ndarray, axis: int):
    """
    calculate the mean of the data.
    :param data:
    :param axis:
    :return:
    """
    if isinstance(data, np.ndarray):
        if axis == 1:
            row_num, col_num = data.shape
            return np.array([np.mean(data[:, num]) for num in range(col_num)]).reshape(1, col_num)
        elif axis == 0:
            row_num, col_num = data.shape
            return np.array([np.mean(data[num, :]) for num in range(row_num)]).reshape(row_num, 1)
    else:
        raise TypeError('Data type is not correct!')


def calmax(data: np.ndarray, axis: int):
    """
    calculate the max of the data.
    :param data:
    :param axis:
    :return:
    """
    if isinstance(data, np.ndarray):
        if axis == 1:
            row_num, col_num = data.shape
            return np.array([np.max(data[:, num]) for num in range(col_num)]).reshape(1, col_num)
        elif axis == 0:
            row_num, col_num = data.shape
            return np.array([np.max(data[num, :]) for num in range(row_num)]).reshape(row_num, 1)
    else:
        raise TypeError('Data type is not correct!')


def calmin(data: np.ndarray, axis: int):
    """
    calculate the min of the data.
    :param data:
    :param axis:
    :return:
    """
    if isinstance(data, np.ndarray):
        if axis == 1:
            row_num, col_num = data.shape
            return np.array([np.min(data[:, num]) for num in range(col_num)]).reshape(1, col_num)
        elif axis == 0:
            row_num, col_num = data.shape
            return np.array([np.min(data[num, :]) for num in range(row_num)]).reshape(row_num, 1)
    else:
        raise TypeError('Data type is not correct!')


def mmnorm(data: np.ndarray, axis: int):
    """
    normalize the data.
    :param data:
    :param axis:
    :return:
    """
    if isinstance(data, np.ndarray):
        if axis == 1:
            row_num, col_num = data.shape
            max_ = calmax(data, axis)
            min_ = calmin(data, axis)
            return np.array([(data[:, num] - min_[:, num]) / (max_[:, num] - min_[:, num])
                             for num in range(col_num)]).T
        elif axis == 0:
            row_num, col_num = data.shape
            max_ = calmax(data, axis)
            min_ = calmin(data, axis)
            return np.array([(data[num, :] - min_[num, :]) / (max_[num, :] - min_[num, :])
                             for num in range(row_num)])

    else:
        raise TypeError('Data type is not correct!')


def znorm(data: np.ndarray, axis: int):
    """
    normalize the data.
    :param data:
    :param axis:
    :return:
    """
    if isinstance(data, np.ndarray):
        if axis == 1:
            row_num, col_num = data.shape
            mean_ = calmean(data, axis)
            std_ = calstd(data, axis)
            return np.array([(data[:, num] - mean_[:, num]) / std_[:, num]
                             for num in range(col_num)]).T
        elif axis == 0:
            row_num, col_num = data.shape
            mean_ = calmean(data, axis)
            std_ = calstd(data, axis)
            return np.array([(data[num, :] - mean_[num, :]) / std_[num, :]
                             for num in range(row_num)])

    else:
        raise TypeError('Data type is not correct!')


def cal2ddist(x, y):
    """

    :param x:
    :param y:
    :return:
    """

    return np.sqrt(np.square(x) + np.square(y))


def caldist(data: np.ndarray, axis: int):
    """

    :param data:
    :param axis:
    :return:
    """
    if isinstance(data, np.ndarray):
        if axis == 1:
            row_num, col_num = data.shape
            square = np.square(data)
            return np.sqrt(np.sum(square, axis=0)).reshape(1, col_num)
        elif axis == 0:
            row_num, col_num = data.shape
            square = np.square(data)
            return np.sqrt(np.sum(square, axis=1)).reshape(row_num, 1)

    else:
        raise TypeError('Data type is not correct!')


def caltimeave(data: np.ndarray, axis: int):
    """

    :param data:
    :param axis:
    :return:
    """
    if isinstance(data, np.ndarray):
        if axis == 1:
            row_num, col_num = data.shape
            diff = np.diff(data[:, 0], axis=0).reshape(row_num - 1, 1)
            pred_last = calmean(diff, axis=axis).reshape(1, 1)
            weight = np.insert(diff, row_num-1, pred_last, axis=0)
            return np.array([np.sum(data[:, num] * weight[:, 0]) / np.sum(weight[:, 0])
                             for num in range(col_num)]).reshape(1, col_num)

        elif axis == 0:
            row_num, col_num = data.shape
            diff = np.diff(data[0, :], axis=0).reshape(1, col_num - 1)
            pred_last = calmean(diff, axis=axis).reshape(1, 1)
            weight = np.insert(diff, col_num-1, pred_last, axis=1)
            return np.array([np.sum(data[num, :] * weight[0, :]) / np.sum(weight[0, :])
                             for num in range(row_num)]).reshape(row_num, 1)


def mag(data: np.ndarray, axis: int):
    """

    """
    if isinstance(data, np.ndarray):
        if axis == 1:
            row_num, col_num = data.shape
            return np.array([np.sqrt(np.sum(np.square(data[num, :]))) for num in range(row_num)]).reshape(row_num, 1)

        elif axis == 0:
            row_num, col_num = data.shape
            return np.array([np.sqrt(np.sum(np.square(data[:, num]))) for num in range(col_num)]).reshape(1, col_num)


def scale_up(data: np.ndarray, axis: int):
    """

    :param :
    :param :
    :param :
    :return:
    """
    if isinstance(data, np.ndarray):
        array_insert_0_ = np.insert(data, 0, 0, axis=axis)
        array_insert_0 = np.delete(array_insert_0_, -1, axis=axis)

        array_interp_ = (data + array_insert_0) / 2
        array_interp = np.delete(array_interp_, 0, axis=axis)
        index_ = np.arange(1, data.shape[axis], 1)
        return np.insert(data, index_, array_interp, axis=axis)
    else:
        raise TypeError('Data type is not correct!')


def array_sort(data, axis_, num_):
    """
    sort the array.
    :param:
    :return:
    """
    if axis_ == 0:
        axis = 1
        dic = {num: data[:, num] for num in range(data.shape[axis])}
        dic_sorted = sorted(dic.items(), key=lambda x: x[1][num_])
        return np.array([dic_sorted[num][1] for num in range(data.shape[axis])]).T
    else:
        axis = 0
        dic = {num: data[num, :] for num in range(data.shape[axis])}
        dic_sorted = sorted(dic.items(), key=lambda x: x[1][num_])
        return np.array([dic_sorted[num][1] for num in range(data.shape[axis])])


def calrecur():
    """

    :param :
    :return:
    """
    pass
    # folder_list = []
    # dic = {}
    # for folder in os.listdir(folder_):
    #     folder_list.append(folder)
    # lis_column0 = []
    # lis_column1 = []
    # dic_counter = {}
    # lis_minmax = []
    # for i in folder_list:
    #     lis_column0 += list(dict_[i][:, 0])
    #     lis_column1 += list(dict_[i][:, 1])
    #     lis = [str([x[0], x[1]]) for x in dict_[i]]
    #     dic_counter[i] = Counter(lis)


def if_none(value):
    """
    check if the value is None.
    :param value:
    :return:
    """
    if value is None:
        return None
    elif value == 'None':
        return None
    elif value == 'none':
        return None
    elif value == 'NONE':
        return None
    elif value == 'off':
        return None
    elif value == 'OFF':
        return None
    else:
        return value


def if_false(value):
    """

    :param value:
    :return:
    """
    if value is False:
        return False
    elif value == 'False':
        return False
    elif value == 'false':
        return False
    elif value == 'FALSE':
        return False
    elif value == 'off':
        return False
    elif value == 'OFF':
        return False
    else:
        return value


def if_true(value):
    """

    :param value:
    :return:
    """
    if value is True:
        return True
    elif value == 'True':
        return True
    elif value == 'true':
        return True
    elif value == 'TRUE':
        return True
    elif value == 'on':
        return True
    elif value == 'ON':
        return True
    else:
        return value


def check_file(file_path):
    """
    Check if the specified file exists.
    
    :param file_path: Path to the file to check.
    :raises FileNotFoundError: If the file does not exist.
    """
    if os.path.exists(file_path):
        pass
    else:
        raise FileNotFoundError('Data file is not found!')


def init_config():
    with open('config.ini', 'w') as cfg:
        cfg.write('; Who\'s Plot - a plot template on Python Language.\n')
        cfg.write('; It aims to provide a convenient method for those who want to plot in science style.\n')
        cfg.write(';\n')
        cfg.write('; Author: Hu, Zehua\n')
        cfg.write('; Email: merlinihu0828@gmail.com\n')
        cfg.write('; Version: 0.0.1\n')
        cfg.write('; The MIT License\n')
        cfg.write('; Copyright(c) 2023 Hu Zehua\n')
        cfg.write('\n')
        cfg.write('\n')
        cfg.write('[Default]\n')
        cfg.write('\n')
        cfg.write('file_location = \n')
        cfg.write('; Both absolute path and file name are accepted. Required!\n')
        cfg.write('\n')
        cfg.write('figure_kind = \n')
        cfg.write(r"; set as ['plot', 'scatter', 'mix', ...] "
                  r"; Attention, only string type in list is readable. Required!" + '\n')
        cfg.write('; The length of the list is the same as figure number.\n')
        cfg.write('\n')
        cfg.write('y_label_type = \n')
        cfg.write("; ['linear', 'log', 'symlog', 'logit', 'function', ...], Default is linear." + '\n')
        cfg.write("; Setting as y_label_type = None is also valid." + '\n')
        cfg.write('\n')
        cfg.write('x_label_type = \n')
        cfg.write('; Ditto. \n')
        cfg.write('\n')
        cfg.write('y_ticklabel_format = \n')
        cfg.write("; ['plain', 'scientific', 'sci', ...], Default is plain." + '\n')
        cfg.write("; Setting as y_label_type = None is also valid." + '\n')
        cfg.write('\n')
        cfg.write('x_ticklabel_format = \n')
        cfg.write('; Ditto. \n')
        cfg.write('\n')
        cfg.write('side_ylabel = \n')
        cfg.write('; On the horizon ... ...\n')
        cfg.write('\n')
        cfg.write('mixed_ylegend = \n')
        cfg.write("; mixed_ylegend = [[x1, y1, y2, y3], [], [] ]\n")
        cfg.write("; insert x to y left->  [x1, y1, x2, y2, x3, y3]\n")
        cfg.write(";    item index     ->   0    1   2   3   4   5\n")
        cfg.write("; if someone want to scatter y2 and y3 in a mix kind of figure,  \n")
        cfg.write("; then mixed_ylegend = [[3, 5], [], [] ]\n")
        cfg.write('title = \n')
        cfg.write("; ['title1', 'title2', ...], Default is None." + '\n')
        cfg.write('\n')
        cfg.write('line_with_marker = \n')
        cfg.write(';\n')
        cfg.write('\n')
        cfg.write('legend_location = \n')
        cfg.write("; ['best', 'upper right', 'upper left', 'lower left', 'lower right', ...], "
                  "Default is 'best'." + '\n')
        cfg.write('\n')
        cfg.write('limited_label = None\n')
        cfg.write("; [[xmin, xmax, ymin, ymax], [......], ...], if xmin and xmax equals zero, "
                  "it will turn off of the limitation of xaxix." + '\n')
        cfg.write('\n')
        cfg.write('color_map = viridis \n')
        cfg.write(';; Perceptually Uniform Sequential:    viridis, plasma, inferno, magma, cividis.\n')
        cfg.write('\n')
        cfg.write('figure_matrix = [1,1]\n')
        cfg.write(';\n')
        cfg.write('\n')
        cfg.write('figure_size = [8,6]\n')
        cfg.write(';\n')
        cfg.write('\n')
        cfg.write('multi_line_style = True\n')
        cfg.write(';\n')
        cfg.write('\n')
        cfg.write('line_width = 1.5\n')
        cfg.write(';\n')
        cfg.write('\n')
        cfg.write('debug_mode = False\n')
        cfg.write(';\n')
        cfg.write('\n')


def print_header():
    platform_total = str(platform.platform() + ' \n' +
                         platform.machine() + ' - Python-' + platform.python_version())
    printer_septal_line()
    printwp(__title__ + ' - ' + __version__)
    printwp(__description__)
    printwp('Author: ' + __author__)
    printwp('Email: ' + __author_email__)
    printwp(__license__)
    printwp(__copyright__)
    printer_septal_line()
    printwp(platform_total)
    printer_septal_line()
    if not os.path.exists('config.ini'):
        init_config()
        printwp('config.ini has been created. Please set parameter.')
        printer_septal_line()
        raise SystemExit
