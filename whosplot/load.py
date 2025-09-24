from whosplot.parameter import Parameter
from whosplot.utility import if_none, numpy_load_data
from whosplot.calculator import calculator
from whosplot.abstract import Abstract
import csv
import copy
import os
import re
import numpy as np


class CsvReader(Abstract):
    """
    The CsvReader class is used to read the csv file and return the data to be plotted.
    The csv file must be in the same directory as the script.
    The first row is the header.
    The only required formulation is label, that must be setted as x_label::y1_label::y2_label ...
    """

    def __init__(self, key=None):
        super(CsvReader, self).__init__()
        self.__Parameter = Parameter()
        self.__set_attributes(key=key)
        self.__load_data()

    def __set_attributes(self, key):
        """
        Set various attributes for the execution and plotting of CSV data.
        """
        if if_none(key):
            file_location = self.__Parameter.__getattr__('file_location')
            file_name = self.__Parameter.__getattr__('file_name')
            self.__Parameter.__setattr__('file_location', file_location.replace(file_name, key))

        param_attrs = ['file_location', 'file_name','cols', 'rows', 'color_map', 'figure_kind', 'mixed_ylegend', 'limited_label']
        for attr in param_attrs:
            setattr(self, f"_CsvReader__{attr}", getattr(self.__Parameter, attr))

    @classmethod
    def __extract_label(cls, header: list) -> list:
        """
        Extract labels from the CSV header.
        
        :param header: CSV header.
        :return: List of extracted labels.
        """
        return [x for x in header if len(x.split('::')) > 1]
    
    @classmethod
    def __insert_whos_plot(cls, list_: list) -> list:
        """
        Insert 'whos_plot' into the list at specified intervals.
        
        :param list_: Input list.
        :return: Modified list with 'whos_plot' inserted.
        """
        list_new = list_
        for num in range(len(list_new)):
            if 'whos_plot' in list_new[num]:
                cls.__insert_whos_plot_helper(list_new, num)
            elif 'whos_plot' not in list_new[num]:
                cls.__insert_whos_plot_helper(list_new, num)
        return list_new

    @staticmethod
    def __insert_whos_plot_helper(list_, num):
        for num_ in range(0, 100, 2):
            if num_ < len(list_[num]) and 'whos_plot' not in list_[num][num_] and num_ != 0:
                list_[num].insert(num_, 'whos_plot')
            elif num_ >= len(list_[num]):
                break

    @classmethod
    def __split_header_from_data(cls, rows: list) -> tuple:
        """
        Split the header from the data rows.
        
        :param rows: List of rows.
        :return: Tuple containing the header and data rows.
        """
        return rows[0], rows[1:]

    @classmethod
    def __replace_empty_data(cls, rows: list) -> list:
        """
        Replace empty cells in the data rows.
        
        :param rows: List of rows.
        :return: List of rows with empty cells replaced.
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
        Insert X data into the array.

        :param data: Data array.
        :param legend: List of legend strings.
        :param header: List of header strings.
        :return: Modified data array with inserted X data.
        """
        columns = len(legend)
        if columns == 2:
            return data

        data_array = np.zeros((data.shape[0], columns))
        start_num = 0
        flag = True

        while flag:
            for num in range(start_num, columns):
                if legend[num] == header[num] and num != columns - 1:
                    data_array[:, num] = data[:, num]
                elif legend[num] != header[num] and num != columns - 1:
                    header.insert(num, legend[num])
                    data = np.insert(data, num, data[:, num - 2], axis=1)
                    data_array[:, num] = data[:, num]
                    start_num = num + 1
                    break
                elif legend[num] == header[num] and num == columns - 1:
                    data_array[:, num] = data[:, num]
                    flag = False

        return data_array

    @classmethod
    def __split_data_array(cls, data: np.ndarray, label_position: list) -> list:
        """
        Split the data array into several parts based on label positions.

        :param data: Input data array or list.
        :param label_position: List of label positions.
        :return: List of split data arrays.
        """
        data_list = []

        if len(label_position) == 1:
            data_list.append(data)
            return data_list

        if isinstance(data, np.ndarray):
            for num in range(len(label_position)):
                if num == 0:
                    data_list.append(data[:, :label_position[num + 1]])
                elif 0 < num < len(label_position) - 1:
                    data_list.append(data[:, label_position[num]:label_position[num + 1]])
                elif num == len(label_position) - 1:
                    data_list.append(data[:, label_position[num]:])
        elif isinstance(data, list):
            for num in range(len(label_position)):
                if num == 0:
                    data_list.append(data[:label_position[num + 1]])
                elif 0 < num < len(label_position) - 1:
                    data_list.append(data[label_position[num]:label_position[num + 1]])
                elif num == len(label_position) - 1:
                    data_list.append(data[label_position[num]:])

        return data_list

    def __set_language(self) -> None:
        """Set the language based on the content of the header."""
        language = self.__get_language()
        if language in ['chinese', 'english']:
            self.language = language
        else:
            raise ValueError("Language is not valid")

    def __get_language(self) -> str:
        """
        Detect the language from the header.

        :return: Detected language ('chinese' or 'english').
        """
        for word in self.header:
            for character in word:
                if '\u4e00' <= character <= '\u9fff':
                    return 'chinese'
        return 'english'

    def __get_label_position(self) -> list:
        """
        Get the positions of labels in the header.

        :return: List of label positions.
        """
        return [count for count in range(len(self.header)) if len(self.header[count].split('::')) > 1]

    def __set_label_position(self) -> None:
        """Set the label positions in the class attribute."""
        self.__label_position = self.__get_label_position()

    def __set_kind_index(self) -> None:
        """Set the index for different figure kinds."""
        self.kind_index = {}
        figure_number = self.__cols * self.__rows
        legend_len_list = self.legend_len_list

        for num in range(figure_number):
            if self.__figure_kind[num] == 'scatter':
                self.kind_index[num] = [x for x in range(legend_len_list[num] * 2)]
            elif self.__figure_kind[num] == 'plot':
                self.kind_index[num] = []
            elif self.__figure_kind[num] == 'mix':
                self.kind_index[num] = self.__mixed_ylegend[num]

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
        Standardize the CSV array by filling empty cells.

        :return: Standardized numpy array.
        """
        if self.__check_data_array_shape() == 1:
            rows = self.__csv_get_data_in_row()
            header, data = self.__split_header_from_data(rows)
            new_rows = self.__replace_empty_data(data)
            stack_list = [np.array(value).flatten() for value in new_rows]
            csv_array = np.column_stack(stack_list).T
            return csv_array

        return self.__numpy_get_data()

    def __standardize_data_array(self) -> list:
        """
        Standardize the data array.

        :return: List of standardized data arrays.
        """
        csv_array = self.__standardize_csv_array()
        legend = self.legend
        header = self.__split_data_array(self.header, self.__label_position)
        data_list_ = self.__split_data_array(csv_array, self.__label_position)
        data_list = [self.__insert_x_data(data_list_[num], legend[num], header[num])
                     for num in range(len(data_list_))]
        return data_list

    def __set_data(self) -> None:
        """Set the CSV data."""
        self.data = self.__standardize_data_array()

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


class OfPostProcess:
    def __init__(self, folder_path: str,
                 dict_key: list = None,
                 drop_out: int = 1
                 ):
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
            ave_scalar = calculator(mathod='of_time_average', data=data_limited, axis=1)[0][1]
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
            ave_velocity = calculator(mathod='of_time_average', data=data_load, axis=1)
            last_num = int((len(data_load[0]) - 1) / 3)
            case_num = re.findall(r'.*?\\case(\d+)', probesU_file)[0]

            if auto_save:
                header = ['Time_mean'] + [f'UxMean{i}, UyMean{i}, UzMean{i}' for i in range(0, last_num)]
                self.numpy_save_data(f'./case{case_num}-ave_velocity.csv', ave_velocity, header=header)

            dic[f'case{case_num}'] = ave_velocity
        return dic

    @classmethod
    def __of_cpp_dict_read(cls, file_path: str):
        with open(file_path, 'r') as f:
            reader = f.read()
            return reader

    def of_cases_sort(self, file_name_dict):
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