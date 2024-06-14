from whosplot.parameter import Parameter
from whosplot.utility import *
from whosplot.abstract import Abstract
import csv
import copy
import os
import re
import numpy as np


class BasicCsv:
    """
    This code defines a class called BasicCsv, which has two class methods: numpy_save_data and numpy_load_data.

    The __init__ method is empty, which means it doesn't do anything when an instance of the class is created.

    The numpy_save_data class method takes in a file path, a data array, and a list of header strings. It uses the
    numpy.savetxt function to save the data to a file in CSV format, with the specified header. The function returns the
    column of the saved file.

    The numpy_load_data class method takes in a file path and an optional skiprows argument, which is the number of rows
    to skip when loading the data. It uses the numpy.loadtxt function to load the data from the specified file in CSV
    format and returns a numpy array containing the loaded data.

    Overall, this code provides a convenient way to save and load data in CSV format using numpy. However,
    the class could benefit from additional methods for manipulating and processing CSV data, such as methods for
    filtering, transforming, and joining data.
    """

    def __init__(self):
        pass

    @classmethod
    def numpy_save_data(cls, file_path: str, data, header: list):
        """
        Save the data into a numpy array.
        :return: column
        """
        np.savetxt(fname=file_path, X=data, fmt='%.12f', delimiter=',', header=','.join(header), comments='')

    @classmethod
    def numpy_load_data(cls, file_path: str, skiprows: int = 0):
        numpy_data = np.loadtxt(fname=file_path,
                                dtype=np.float64,
                                encoding='utf-8-sig',
                                delimiter=',',
                                skiprows=skiprows)
        return numpy_data


class CsvReader(Abstract):
    """
    The CsvReader class is used to read the csv file and return the data to be plotted.
    The csv file must be in the same directory as the script.
    The first row is the header.
    The only required formulation is label, that must be setted as x_label::y1_label::y2_label ...
    """

    def __init__(self):
        super(CsvReader, self).__init__()
        self.__Parameter = Parameter()
        self.__set_file_location()
        self.__set_cols_and_rows()
        self.__set_figure_kind()
        self.__set_mixed_ylegend()
        self.__set_limited_label()
        self.__load_data()

    def __set_file_location(self) -> None:
        """
        Get the file path of the csv file.
        :return: file path
        """
        self.__file_location = self.__Parameter.__getattr__('file_location')

    def __set_cols_and_rows(self) -> None:
        """
        Get the cols and rows of the csv file.
        :return: cols and rows
        """
        self.__cols = self.__Parameter.__getattr__('cols')
        self.__rows = self.__Parameter.__getattr__('rows')

    def __set_figure_kind(self) -> None:
        """
        Get the figure kind of the csv file.
        :return: figure kind
        """
        self.__figure_kind = self.__Parameter.__getattr__('figure_kind')

    def __set_mixed_ylegend(self) -> None:
        """
        Get the mixed ylegend of the csv file.
        :return: mixed ylegend
        """
        self.__mixed_ylegend = self.__Parameter.__getattr__('mixed_ylegend')

    def __set_limited_label(self) -> None:
        """
        Get the limited label of the csv file.
        :return: limited label
        """
        self.__limited_label = self.__Parameter.__getattr__('limited_label')

    @classmethod
    def __extract_label(cls, header: list) -> list:
        """
        Extract the label from the csv header.
        :param header: csv header
        :return: label
        """

        labels = [x for x in header if x.split('::').__len__() > 1]
        return labels

    @classmethod
    def __insert_whos_plot(cls, list_: list):
        """
        Insert the whos_plot into the list.
        :param list_:
        :return:
        """
        list_new = list_
        for num in range(len(list_new)):
            if 'whos_plot' in list_new[num]:
                for num_ in range(0, 100, 2):
                    if num_ < len(list_new[num]) and 'whos_plot' not in list_new[num][num_] and num_ != 0:
                        list_new[num].insert(num_, 'whos_plot')
                    elif num_ >= len(list_new[num]):
                        break
            elif 'whos_plot' not in list_new[num]:
                for num_ in range(0, 100, 2):
                    if num_ < len(list_new[num]) and num_ != 0:
                        list_new[num].insert(num_, 'whos_plot')
                    elif num_ >= len(list_new[num]):
                        break
        return list_new

    @classmethod
    def __split_header_from_data(cls, rows: list) -> tuple:
        """
        Create the empty columns for an unreadable data file.
        :return: column
        """
        return rows[0], rows[1:]

    @classmethod
    def __replace_empty_data(cls, rows: list) -> list:
        """
        Fill the empty columns.
        :return: column
        """
        for row_num in range(len(rows)):
            for column_num in range(len(rows[row_num])):
                if rows[row_num][column_num] == '':
                    rows[row_num][column_num] = rows[row_num - 1][column_num]

                else:
                    rows[row_num][column_num] = float(rows[row_num][column_num])

        return rows

    @classmethod
    def __numpy_save_data(cls, file_name: str, data, header: list) -> int:
        """
        Save the data into a numpy array.
        :return: column
        """
        np.savetxt(fname='{}_standardized.csv'.format(file_name), X=data, fmt='%.18f', delimiter=',',
                   header=','.join(header), comments='')
        return 0

    @classmethod
    def __insert_x_data(cls, data: np.ndarray, legend: list, header: list) -> np.ndarray:
        """
        Split the data into several parts.
        :return: data_list
        """
        flag = True
        start_num = 0
        columns_ = len(legend)
        if columns_ == 2:
            data_array = data
            return data_array
        else:
            data_array = np.zeros((data.shape[0], columns_))
            while flag:
                for num in range(start_num, columns_):
                    if legend[num] == header[num] and num != columns_ - 1:
                        data_array[:, num] = data[:, num]
                    elif legend[num] != header[num] and num != columns_ - 1:
                        header.insert(num, legend[num])
                        data = np.insert(data, num, data[:, num - 2], axis=1)
                        data_array[:, num] = data[:, num]
                        start_num = num + 1
                        break
                    elif legend[num] == header[num] and num == columns_ - 1:
                        data_array[:, num] = data[:, num]
                        flag = False
        return data_array

    @classmethod
    def __split_data_array(cls, data: np.ndarray or list, label_position: list) -> list:
        """
        Split the data into several parts.
        :return: data_list
        """
        data_list = []

        if label_position.__len__() == 1:
            data_list.append(data)
            return data_list

        elif label_position.__len__() > 1 and type(data) == np.ndarray:
            for num in range(len(label_position)):
                if num == 0:
                    data_list.append(data[:, :label_position[num + 1]])
                elif 0 < num < len(label_position) - 1:
                    data_list.append(data[:, label_position[num]:label_position[num + 1]])
                elif num == len(label_position) - 1:
                    data_list.append(data[:, label_position[num]:])
            return data_list
        elif label_position.__len__() > 1 and type(data) == list:
            for num in range(len(label_position)):
                if num == 0:
                    data_list.append(data[:label_position[num + 1]])
                elif 0 < num < len(label_position) - 1:
                    data_list.append(data[label_position[num]:label_position[num + 1]])
                elif num == len(label_position) - 1:
                    data_list.append(data[label_position[num]:])
            return data_list

    def __get_language(self) -> str:
        """
        Get the language from the command line arguments.
        :return: language
        """
        for word in self.header:
            for character in word:
                if '\u4e00' <= character <= '\u9fff':
                    return 'chinese'
        return 'english'

    def __set_language(self) -> int:
        """
        Set the language.
        :return:
        """
        if self.__get_language() == 'chinese':
            self.language = 'chinese'
            return 0
        elif self.__get_language() == 'english':
            self.language = 'english'
            return 0
        else:
            raise ValueError("language is not valid")

    def __get_label_position(self) -> list:
        """
        Get the label position.
        :return: label_position
        """
        return [count for count in range(len(self.header)) if self.header[count].split('::').__len__() > 1]

    def __set_label_position(self):
        """
        Set the label position.
        :return: 0
        """
        self.__label_position = self.__get_label_position()

    def __set_kind_index(self):
        """
        Set the mixed ylegend.
        :return: 0
        """
        self.kind_index = {}
        figure_number = int(self.__cols * self.__rows)
        legend_len_list = self.legend_len_list
        for num in range(figure_number):
            if self.__figure_kind[num] == 'scatter':
                self.kind_index[num] = [x for x in range(legend_len_list[num] * 2)]
            elif self.__figure_kind[num] == 'plot':
                self.kind_index[num] = []
            elif self.__figure_kind[num] == 'mix':
                scatter_index = self.__mixed_ylegend[num]
                self.kind_index[num] = scatter_index

    def __get_csv_header(self) -> list:
        """
        Get the csv header from the command line arguments.
        :return: header
        """
        path = self.__file_location
        with open(file=path, mode='r', encoding='utf-8-sig', newline='') as fp:
            csv_reader = csv.reader(fp)
            header = csv_reader.__next__()
        return header

    def __set_csv_header(self):
        """
        Set the csv header.
        :return: 0 or 1
        """
        self.header = self.__get_csv_header()

    def __get_label_list(self) -> tuple:
        """
        Get the label list.
        :return:
        """
        raw_labels_ = self.__extract_label(self.header)
        label = [label.split('::') for label in raw_labels_]
        label_len_list = [len(label) for label in label]
        return label, label_len_list

    def __set_label_list(self):
        """
        Set the label list.
        :return:
        """
        self.label, self.label_len_list = self.__get_label_list()
        for num in range(len(self.label)):
            if self.label_len_list[num] != len(self.__limited_label[num]) / 2:
                raise ValueError("The number of labels is not equal to the number of limited labels.")

    def __get_legend_list(self) -> tuple:
        """
        Get the legend list.
        :return: legend_list
        """
        header_ = copy.deepcopy(self.header)
        split_header = self.__split_data_array(header_, self.__label_position)
        legend = self.__insert_whos_plot(split_header)
        legend_len_list = [int(len(list_) / 2) for list_ in legend]
        return legend, legend_len_list

    def __set_legend_list(self):
        """
        Set the legend list.
        :return: 0
        """
        self.legend, self.legend_len_list = self.__get_legend_list()

    def __get_figure_number(self) -> int:
        """
        Get the figure number from the header.
        :return: figure_number
        """
        raw_labels_ = self.__extract_label(self.header)
        figure_number = len(raw_labels_)
        return figure_number

    def __set_figure_number(self):
        """
        Set the figure number.
        :return: 0
        """
        self.figure_number = self.__get_figure_number()

    def __numpy_get_data(self) -> np.ndarray:
        """
        Get the csv data from the command line arguments.
        :return: data_load
        """
        path = self.__file_location
        numpy_data = np.loadtxt(fname=path, dtype=np.float64, encoding='utf-8-sig', delimiter=',', skiprows=1)
        return numpy_data

    def __csv_get_data_in_row(self) -> list:
        """
        Get the csv data from the command line arguments.
        :return: data_load
        """
        np.set_printoptions(threshold=np.inf)
        path = self.__file_location
        with open(file=path, mode='r', encoding='utf-8-sig', newline='') as fp:
            reader = csv.reader(fp, delimiter=',')
            rows = [row for row in reader]
        return rows

    def __check_data_array_shape(self) -> int:
        """
        Check the data shape.
        :return: 0
        """
        path = self.__file_location
        with open(file=path, mode='r', encoding='utf-8-sig', newline='') as fp:
            reader = csv.reader(fp)
            for row in reader:
                if '' in row:
                    return 1
        return 0

    def __standardize_csv_array(self) -> np.ndarray:
        """
        Standardize the data array shape.
        :return: data_load
        """
        check_data_array_shape = self.__check_data_array_shape()

        if check_data_array_shape == 1:
            rows = self.__csv_get_data_in_row()
            header, data = self.__split_header_from_data(rows)
            new_rows = self.__replace_empty_data(data)
            stack_list = [np.array(value).flatten() for value in new_rows]
            csv_array = np.column_stack(stack_list).T
            # self.__numpy_save_data(self.file_location.replace('.csv', ''), csv_array, header)
            return csv_array

        elif check_data_array_shape == 0:
            csv_array = self.__numpy_get_data()
            return csv_array

    def __standardize_data_array(self) -> list:
        """
        Standardize the data.
        :return: data_list
        """
        csv_array = self.__standardize_csv_array()
        legend = self.legend
        header = self.__split_data_array(self.header, self.__label_position)
        data_list_ = self.__split_data_array(csv_array, self.__label_position)
        data_list = [self.__insert_x_data(data=data_list_[num],
                                          legend=legend[num],
                                          header=header[num])
                     for num in range(len(data_list_))]
        return data_list

    def __set_data(self) -> int:
        """
        Set the csv data.
        :return: 0
        """
        self.data = self.__standardize_data_array()
        return 0

    def __load_data(self):
        """
        Load the csv data.
        :return: 0
        """
        self.__set_csv_header()
        self.__set_label_list()
        self.__set_label_position()
        self.__set_legend_list() 
        self.__set_language()
        self.__set_figure_number()
        self.__set_kind_index()
        self.__set_data()


class RegularExpression:
    def __init__(self):
        pass

    @classmethod
    def Bn(cls, text, root):
        look_up = re.findall(r'nucleationSiteModel.*?Bn.*?([\d+/.*]+);', text, re.DOTALL)
        if len(look_up) == 0:
            raise ValueError(f'In {root} Bn is not found!')
        return look_up[0]

    @classmethod
    def Cn(cls, text, root):
        look_up = re.findall(r'nucleationSiteModel.*?Cn.*?([\d+/.*]+);', text, re.DOTALL)
        if len(look_up) == 0:
            raise ValueError(f'In {root} Cn is not found!')
        return look_up[0]

    @classmethod
    def T(cls, text, root):
        look_up = re.findall(r'heated_walls.*?([\d+/.*]+);', text, re.DOTALL)
        if len(look_up) == 0:
            raise ValueError(f'In {root} T is not found!')
        return look_up[0]


class OfPostProcess(BasicCsv):
    def __init__(self, folder_path: str,
                 dict_key: list = None,
                 drop_out: int = 1
                 ):
        super().__init__()
        self.folder_path = folder_path
        self.file_name = dict_key
        self.drop_out = drop_out

    # -------------------------- Private member -------------------------- #

    # -------------------------- Class method -------------------------- #
    @classmethod
    def __get_folder_tree(cls, folder_path: str):
        """
        Get the folder tree.
        :param folder_path: The folder path.
        :return: folder_tree
        """
        folder_tree = os.walk(folder_path)
        for root, dirs, files in folder_tree:
            yield root, dirs, files

    @classmethod
    def __check_str_in_list(cls, list_: list, str_: str) -> bool:
        """
        Get the list of the string in the list.
        :param list_: The list.
        :param str_: The string.
        :return: list_
        """
        for num in range(len(list_)):
            check = re.findall(list_[num], str_)
            if check:
                return True
        return False

    @classmethod
    def __get_surfaceFieldValue_path(cls, folder_path: str, file_list: list) -> list:
        """
        Get the log file path.
        :param folder_path: The folder path.
        :return: log_file_path
        """
        surfaceFieldValue_path = []
        for root, dirs, files in cls.__get_folder_tree(folder_path):
            for file in files:
                if file.endswith('.dat') and cls.__check_str_in_list(file_list, root):
                    surfaceFieldValue_path.append(os.path.join(root, file))
        return surfaceFieldValue_path

    @classmethod
    def __get_probes_path(cls, folder_path: str, file_list: list) -> list:
        """
        Get the probes file path.
        :param folder_path: The folder path.
        :return: probes_file_path
        """
        probes_path = []
        for root, dirs, files in cls.__get_folder_tree(folder_path):
            for file in files:
                if cls.__check_str_in_list(file_list, os.path.join(root, file)):
                    probes_path.append(os.path.join(root, file))
        return probes_path

    @classmethod
    def __get_zero_folder_path(cls, folder_path: str, file_list: list) -> list:
        """
        Get the zero folder path.
        :param folder_path: The folder path.
        :return: zero_folder_path
        """
        zero_folder_path = []
        for root, dirs, files in cls.__get_folder_tree(folder_path):
            for file in files:
                if cls.__check_str_in_list(file_list, os.path.join(root, file)):
                    zero_folder_path.append(os.path.join(root, file))
        return zero_folder_path

    @classmethod
    def __surfaceFieldValue_get_header(cls, file_path: str) -> list:
        """
        Get the header.
        :param file_path: The file path.
        :return: header
        """
        with open(file=file_path, mode='r', encoding='utf-8-sig', newline='') as fp:
            reader = fp.readlines()
            surfaceFieldValue = [line for line in reader if line != '\n']
            for line in surfaceFieldValue:
                if '# Time' in line:
                    header = line.strip().split('\t')
                    return [x.replace('#', '').replace(' ', '') for x in header]

    @classmethod
    def __data_between_limitation(cls, data: np.ndarray) -> np.ndarray:
        """
        Get the data limited time.
        :param data: The data.
        :return: data
        """
        start_time = data[:, 0][-1] - 0.5
        end_time = data[:, 0][-1]
        data0 = data[start_time <= data[:, 0]]
        data_ = data0[data0[:, 0] <= end_time]
        return data_

    @classmethod
    def __surfaceFieldValue_numpy_load(cls, file_name: str):
        """
        Load the surfaceFieldValue file.
        :param file_name.
        :return: data
        """
        data = np.loadtxt(file_name, delimiter='\t', dtype=np.float64, comments='#')
        return data

    @classmethod
    def __probes_vector_to_array(cls, file_path: str, drop_out):
        velocity_list = []
        time_list = []
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            reader = f.readlines()
            for line in reader[::drop_out]:
                if not line.startswith('#'):
                    re_find = re.findall(r'\((.*?)\)', line)
                    velocity_line = np.array([np.array(Uxyz.split(' ')).astype(float) for Uxyz in re_find]).flatten()
                    velocity_list.append(velocity_line)
                    time_list.append(float(line.split(' ', 1)[0]))

        data = np.insert(np.array(velocity_list), 0, np.array(time_list).T, axis=1)
        return data

    @classmethod
    def __residuals_to_array(cls, file_path: str, skiprows=0):
        try:
            data = np.loadtxt(file_path, delimiter='\t', dtype=np.float64, comments='#', skiprows=skiprows)
            if skiprows != 0:
                print('skiprows = ', skiprows)
            return data
        except ValueError:
            return cls.__residuals_to_array(file_path, skiprows=skiprows + 1)

    @classmethod
    def __dict_in_list_to_array(cls, dict_in_list: list, row: int, col: int, lambda_func: callable):
        """
        Save the dict to npy file.
        :param dict_in_list
        :return: None
        """

        dict_in_list.sort(key=lambda_func)
        value_array = np.array([list(x.values())[0] for x in dict_in_list]).reshape((row, col))
        key_array = np.array([list(x.keys())[0] for x in dict_in_list]).reshape((row, col))
        return key_array, value_array

    def surfaceFieldValue_to_csv(self):
        log_file_path = self.__get_surfaceFieldValue_path(self.folder_path, self.file_name)
        for log_file in log_file_path:
            data_load = self.__surfaceFieldValue_numpy_load(log_file)
            header = self.__surfaceFieldValue_get_header(log_file)
            case_num = re.findall(r'.*?\\case(\d+)', log_file)[0]
            for file in self.file_name:
                flag = re.findall(r'\\({})\\'.format(file), log_file)
                if flag:
                    self.numpy_save_data(f'./{case_num}-{flag[0]}.csv', data_load, header=header)
                    break

    def residuals_to_csv(self, auto_save=True):
        dic = {}
        residuals_file_path = self.__get_surfaceFieldValue_path(self.folder_path, file_list=['residuals'])
        for residuals_file in residuals_file_path:
            data_load = self.__residuals_to_array(residuals_file)
            header = self.__surfaceFieldValue_get_header(residuals_file)
            case_num = re.findall(r'.*?\\case(\d+)', residuals_file)[0]

            if auto_save:
                self.numpy_save_data(f'./case{case_num}-residuals.csv', data_load, header=header)
            dic[f'case{case_num}'] = data_load

        return dic

    def probes_vector_to_csv(self, auto_save=True):
        dic = {}
        probesU_file_path = self.__get_probes_path(self.folder_path, self.file_name)
        for probesU_file in probesU_file_path:
            data_load = self.__probes_vector_to_array(probesU_file, drop_out=self.drop_out)
            last_num = int((len(data_load[0]) - 1) / 3)
            header = ['Time'] + [f'Ux{i}, Uy{i}, Uz{i}' for i in range(0, last_num)]
            case_num = re.findall(r'.*?\\case(\d+)', probesU_file)[0]
            if auto_save:
                self.numpy_save_data(f'./case{case_num}-probesU.csv', data_load, header=header)
            dic[f'case{case_num}'] = data_load
        return dic

    def probes_scalar_to_csv(self, auto_save=True):
        pass

    def probes_scalar_magU_to_csv(self, auto_save=True):
        pass

    def surfaceFieldValue_time_average_scalar(self, auto_save=True):
        dic = {}
        ave_scalar_list = []
        log_file_path = self.__get_surfaceFieldValue_path(self.folder_path, self.file_name)
        for log_file in log_file_path:
            data_load = self.__surfaceFieldValue_numpy_load(log_file)
            data_limited = self.__data_between_limitation(data_load)
            ave_scalar = caltimeave(data_limited, 1)[0][1]
            case_num = re.findall(r'.*?\\case(\d+)', log_file)[0]
            for file in self.file_name:
                flag = re.findall(r'\\({})\\'.format(file), log_file)
                if flag:
                    ave_scalar_list.append({f'{case_num}-{flag[0]}': ave_scalar})
                    break
        row = len(log_file_path) // len(self.file_name)
        col = len(self.file_name)

        ave_scalar_list.sort(key=lambda x: int(list(x.keys())[0].split('-')[0]))
        save_value = np.array([list(x.values())[0] for x in ave_scalar_list]).reshape((row, col))
        save_key = np.array([list(x.keys())[0] for x in ave_scalar_list]).reshape((row, col))

        if auto_save:
            self.numpy_save_data('./ave_scalar.csv', save_value, header=self.file_name)
            np.savetxt('./ave_scalar_key.csv', save_key, delimiter=',', fmt='%s', header=','.join(self.file_name))

        dic['save_key'] = save_key
        dic['save_value'] = save_value
        return dic

    def probes_vector_U_time_average(self, auto_save=True):
        dic = {}
        probesU_file_path = self.__get_probes_path(self.folder_path, self.file_name)
        for probesU_file in probesU_file_path:
            data_load = self.__probes_vector_to_array(probesU_file, drop_out=self.drop_out)
            ave_velocity = caltimeave(data_load, 1)
            last_num = int((len(data_load[0]) - 1) / 3)
            case_num = re.findall(r'.*?\\case(\d+)', probesU_file)[0]

            if auto_save:
                header = ['Time_mean'] + [f'UxMean{i}, UyMean{i}, UzMean{i}' for i in range(0, last_num)]
                self.numpy_save_data(f'./case{case_num}-ave_velocity.csv', ave_velocity, header=header)

            dic[f'case{case_num}'] = ave_velocity
        return dic

    @classmethod
    def __probes_vector_magU(cls, u_array: np.ndarray, axis: int):
        rows = u_array.shape[0]
        cols = u_array.shape[1]
        array_ = np.zeros((rows, cols // 3))
        for num in range(0, cols, 3):
            array_[:, num // 3:num // 3 + 1] = mag(u_array[:, num:num + 3], axis=axis)
        return array_

    @classmethod
    def __get_case_num(cls, file_path: str):
        try:
            return re.findall(r'.*?\\case(\d+)', file_path)[0]
        except IndexError:
            raise IndexError('Case identity must be case$num!')

    def probes_scalar_magU_maldistribution(self, auto_save=True):
        dic = {}
        scalar_list = []
        probesU_file_path = self.__get_probes_path(self.folder_path, self.file_name)
        for probesU_file in probesU_file_path:
            data_load = self.__probes_vector_to_array(probesU_file, drop_out=self.drop_out)
            magU = self.__probes_vector_magU(data_load[:, 1::], axis=1)
            field_ave_magU = calmean(magU, axis=0)
            field_std_magU = calstd(magU, axis=0)
            maldistribution = field_std_magU / field_ave_magU
            array_ = np.column_stack((data_load[:, 0], maldistribution))
            case_num = self.__get_case_num(probesU_file)

            dic[f'case{case_num}'] = array_
            scalar_list.append({f'case{case_num}': array_})

        row = len(probesU_file_path) // len(self.file_name)
        col = len(self.file_name)
        scalar_list.sort(key=lambda x: int(list(x.keys())[0].replace('case', '')[0]))

        value_array = np.array([])

        for num in range(0, len(scalar_list)):
            if num == 0:
                for key, value in scalar_list[num].items():
                    value_array = value
            else:
                for key, value in scalar_list[num].items():
                    if value_array.shape[0] == value.shape[0]:
                        value_array = np.append(value_array, value, axis=1)
                    else:
                        add_rows = abs(value_array.shape[0] - value.shape[0])
                        if value_array.shape[0] > value.shape[0]:
                            add_array = np.array(list(value[-1]) * add_rows).reshape((add_rows, value.shape[1]))
                            value = np.append(value, add_array, axis=0)
                            value_array = np.append(value_array, value, axis=1)
                        else:
                            add_array = np.array(
                                list(value_array[-1]) * add_rows).reshape((add_rows, value_array.shape[1])
                                                                          )
                            value_array = np.append(value_array, add_array, axis=0)
                            value_array = np.append(value_array, value, axis=1)

        key_array = np.array([list(x.keys())[0] for x in scalar_list]).reshape((row, col))
        if auto_save:
            self.numpy_save_data('./magU_maldistribution.csv', value_array, header=[','])
            np.savetxt('./magU_maldistribution_key.csv',
                       key_array,
                       delimiter=',',
                       fmt='%s',
                       header=',' * len(probesU_file_path))

        return value_array, key_array

    def probes_magU_field_average(self, auto_save=True):
        dic = {}
        probesU_file_path = self.__get_probes_path(self.folder_path, self.file_name)
        for probesU_file in probesU_file_path:
            data_load = self.__probes_vector_to_array(probesU_file, drop_out=self.drop_out)
            rows = data_load.shape[0]
            cols = data_load.shape[1] - 1
            u_array = data_load[:, 1::]
            zero_array = np.zeros((rows, cols // 3))
            save_array = np.zeros((rows, 2))
            save_array[:, 0] = data_load[:, 0]
            for num in range(0, cols, 3):
                zero_array[:, num // 3:num // 3 + 1] = mag(u_array[:, num:num + 3], axis=1)
            save_array[:, 1:2] = calmean(zero_array, axis=0)

            case_num = re.findall(r'.*?\\case(\d+)', probesU_file)[0]

            if auto_save is True:
                path = f'./case{case_num}-probesU_field_average.csv'
                header = ['Time', 'magUMean']
                self.numpy_save_data(path, save_array, header=header)

            dic[f'case{case_num}'] = save_array
        return dic

    @classmethod
    def __of_cpp_dict_read(cls, file_path: str):
        with open(file_path, 'r') as f:
            reader = f.read()
            return reader

    def of_cases_sort(self, file_name_dict: {str: {str: callable}}):
        dic = {}
        file_path = self.__get_zero_folder_path(self.folder_path, self.file_name)
        for cpp_file in file_path:

            reader = self.__of_cpp_dict_read(cpp_file)
            try:
                case_num = re.findall(r'.*?\\(case\d+)', cpp_file)[0]
                for file_name, parameters in file_name_dict.items():
                    if file_name in cpp_file:
                        parameter_key = list(parameters.keys())

                        for key in parameter_key:
                            dic[f'{case_num}-{key}'] = float(parameters[key](reader, cpp_file))

            except IndexError:
                print(f'case num not found in {cpp_file} \nand this folder will be ignored')
        dic_ = {}

        for key, value in dic.items():
            if key.split('-')[0] not in dic_.keys():
                dic_[key.split('-')[0]] = []
            dic_[key.split('-')[0]].append(value)

        return dic_