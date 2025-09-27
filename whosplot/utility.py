
import os
import numpy as np
import platform
import time
import math
from PIL import Image

from whosplot.__init__ import \
    (__title__,
     __description__,
     __version__,
     __author__,
     __author_email__,
     __license__,
     __copyright__)

def numpy_save_data(file_path: str, data, header: list) -> None:
    """
    Save the data into a numpy array.
    
    :param file_path: Path to the file where data will be saved.
    :param data: Data array to save.
    :param header: List of header strings.
    """
    np.savetxt(fname=file_path, X=data, fmt='%.16f', delimiter=',', header=','.join(header), comments='')

def numpy_load_data(file_path: str, skiprows: int = 0) -> np.ndarray:
    """
    Load data from a CSV file into a numpy array.
    
    :param file_path: Path to the CSV file.
    :param skiprows: Number of rows to skip at the beginning of the file.
    :return: Numpy array containing the loaded data.
    """
    numpy_data = np.loadtxt(fname=file_path,
                            dtype=np.float64,
                            encoding='utf-8-sig',
                            delimiter=',',
                            skiprows=skiprows)
    return numpy_data

def array_sort_old(data: np.ndarray, axis_: int, num_: int):
    """
    sort the array.
    :param:
    :return:
    """
    raise DeprecationWarning('This function is deprecated. Please use array_sort instead.')
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

def array_sort(data: np.ndarray, axis: int, num: int, reverse: bool = False):
    """
    sort the array.
    :param:
    :return:
    """
    data = np.asarray(data)

    if axis == 0:
        sorted_indices = np.argsort(data[num, :])
        if reverse:
            sorted_indices = sorted_indices[::-1]
        sorted_data = data[:, sorted_indices]
    elif axis == 1:
        sorted_indices = np.argsort(data[:, num])
        if reverse:
            sorted_indices = sorted_indices[::-1]
        sorted_data = data[sorted_indices, :]
    return sorted_data

alphabet = [chr(i) for i in range(97, 123)]

def figop_remove_margin(file_path: str, save_path: str = None) -> None:
    img = Image.open(file_path)
    img_array = np.array(img)
    non_white = np.any(img_array < 255, axis=-1)
    rows = np.any(non_white, axis=1)
    columns = np.any(non_white, axis=0)
    cropped_img = img_array[rows][:, columns]
    cropped_img_pil = Image.fromarray(cropped_img)
    if save_path is None:
        return cropped_img_pil
    else:
        file_name = os.path.splitext(file_path)[0]
        cropped_img_pil.save(file_name + '.pdf')
        return cropped_img_pil


def create_empty_data_csv(file_path: str, num: int) -> None:

    header = ['::', 'whos_plot'] * num
    data = np.zeros((5, 2 * num))
    numpy_save_data(file_path, data, header)


def timeit(func):
    """
    Decorator to measure the execution time of a function.

    Parameters:
    func : function
        The function whose execution time is to be measured.

    Returns:
    wrapper : function
        The wrapped function that prints its execution time.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time:.6f} seconds")
        return result
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
        cfg.write('debug_mode = False\n')
        cfg.write(';\n')
        cfg.write('\n')


def print_header():
    platform_total = str(platform.platform() + ' \n' +
                         platform.machine() + ' - Python-' + platform.python_version())
    printer_septal_line()
    printwp(__title__ + ' - ' + __version__)
    printwp(__description__)
    printer_septal_line()
    printwp(platform_total)
    printer_septal_line()
    if not os.path.exists('config.ini'):
        init_config()
        printwp('config.ini has been created. Please set parameter.')
        printer_septal_line()
        raise SystemExit

def _pow_signed(x, p):
    return np.sign(x) * (np.abs(x) ** p)

def euler_to_matrix(angles, order='ZYX', degrees=True):
    """Return 3x3 rotation matrix from Euler angles.

    angles: (a,b,c) in degrees by default.
    order:  one of 'XYZ','XZY','YXZ','YZX','ZXY','ZYX'. Default ZYX (yaw,pitch,roll).
    """
    ax, ay, az = angles
    if degrees:
        ax, ay, az = np.deg2rad([ax, ay, az])
    order = order.upper()
    def Rx(t):
        c,s = np.cos(t), np.sin(t)
        return np.array([[1,0,0],[0,c,-s],[0,s,c]])
    def Ry(t):
        c,s = np.cos(t), np.sin(t)
        return np.array([[c,0,s],[0,1,0],[-s,0,c]])
    def Rz(t):
        c,s = np.cos(t), np.sin(t)
        return np.array([[c,-s,0],[s,c,0],[0,0,1]])
    mapping = {'X': Rx, 'Y': Ry, 'Z': Rz}
    R = np.eye(3)
    for ch, ang in zip(order, (ax, ay, az)):
        R = R @ mapping[ch](ang)
    return R

def superellipsoid_grid(a1,a2,a3,e1,e2, u_count=64, v_count=64):
    """Generate an axis-aligned superellipsoid surface grid at the origin.

    Returns X,Y,Z in shape (u_count, v_count).
    Parametrization uses:
        eta in [-pi/2, pi/2], omega in [-pi, pi]
    """
    eta = np.linspace(-np.pi/2, np.pi/2, u_count)
    omega = np.linspace(-np.pi, np.pi, v_count)
    Eta, Om = np.meshgrid(eta, omega, indexing='ij')
    cu = _pow_signed(np.cos(Eta), e1)
    su = _pow_signed(np.sin(Eta), e1)
    cv = _pow_signed(np.cos(Om), e2)
    sv = _pow_signed(np.sin(Om), e2)
    X = a1 * (cu * cv)
    Y = a2 * (cu * sv)
    Z = a3 * (su)
    return X, Y, Z

def apply_pose_to_grid(X, Y, Z, R, t):
    """Apply rotation R (3x3) and translation t (3,) to grid arrays X,Y,Z."""
    P = np.stack([X, Y, Z], axis=-1).reshape(-1, 3)
    P = (P @ R.T) + np.asarray(t).reshape(1,3)
    Xo, Yo, Zo = P[:,0].reshape(X.shape), P[:,1].reshape(Y.shape), P[:,2].reshape(Z.shape)
    return Xo, Yo, Zo

def superellipsoid_points(a1,a2,a3,e1,e2, u_count=64, v_count=64, R=None, t=(0,0,0)):
    X, Y, Z = superellipsoid_grid(a1,a2,a3,e1,e2, u_count=u_count, v_count=v_count)
    if R is None:
        R = np.eye(3)
    X, Y, Z = apply_pose_to_grid(X, Y, Z, R, t)
    P = np.stack([X, Y, Z], axis=-1).reshape(-1,3)
    return P

def downsample_points(P, max_points=3000, seed=0):
    if len(P) <= max_points:
        return P
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(P), size=max_points, replace=False)
    return P[idx]