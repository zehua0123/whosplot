
from whosplot.utility import *
from whosplot.abstract import Abstract
import os
from configparser import ConfigParser


class Parameter(Abstract):
    """
    Parameter class for handling configuration and settings.
    """

    def __init__(self):
        """
        Initialize the class.
        :return:
        """
        super(Parameter, self).__init__()
        self.__set_parameters()

    def __set_parameters(self) -> None:
        """
        Read and set parameters from config.ini file.
        """
        self.__load_config()
        parameter_methods = [
            self.__set_debug_mode,
            self.__set_file_location,
            self.__set_figure_kind,
            self.__set_y_label_type,
            self.__set_x_label_type,
            self.__set_y_ticklabel_format,
            self.__set_x_ticklabel_format,
            self.__set_title,
            self.__set_line_with_marker,
            self.__set_legend_location,
            self.__set_scale_factor,
            self.__set_skip_rows,
            self.__set_limited_label,
            self.__set_color_map,
            self.__set_figure_matrix,
            self.__set_figure_size,
            self.__set_mixed_ylegend,
            self.__set_line_style
        ]

        for method in parameter_methods:
            method()

    @classmethod
    def __get_config_path(cls) -> str:
        """
        Get the path of config.ini file.
        :return:
        """
        return os.path.abspath('config.ini')

    def __load_config(self) -> None:
        """
        Load the config.ini file.
        :return:
        """
        config_path = self.__get_config_path()
        self.configini = ConfigParser()
        self.configini.read(config_path)

    @classmethod
    def __convert_basename_to_abs_path(cls, value: str) -> str:
        """
        Convert the relative path to absolute path.
        :param value:file_location
        :return:
        """
        if '.csv' in value:
            return os.path.abspath(value)
        else:
            return os.path.abspath(value + '.csv')

    @classmethod
    def __convert_abs_path_to_basename(cls, value: str) -> str:
        """
        Convert the relative path to absolute path.
        :param value:file_location
        :return:
        """
        return os.path.basename(value)

    def __get_file_location(self) -> tuple:
        """
        Get the file location from the command line arguments.
        :return: file_location
        """

        if if_none(self.configini.get('Default', 'file_location')) is not None:
            file_location_ = self.configini.get('Default', 'file_location')
            file_location = self.__convert_basename_to_abs_path(file_location_)
            file_name = self.__convert_abs_path_to_basename(file_location.replace('.csv', ''))
            check_file(file_location)

            return file_location, file_name

        else:
            raise ValueError('The file location must be set in config.ini file!')

    def __set_file_location(self) -> None:
        """
        Set the file location.
        :return: 0
        """
        self.file_location, self.file_name = self.__get_file_location()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_file_location() ......')
            printwp('self.file_location: ' + self.file_location)
            printwp('self.file_name: ' + self.file_name)

    def __get_figure_kind(self):
        """
        Get the figure kind from the command line arguments.
        :return: figure_kind
        """
        if if_none(self.configini.get('Default', 'figure_kind')) is not None:
            figure_kind_str = self.configini.get('Default', 'figure_kind')
            figure_kind = eval(figure_kind_str)

            return figure_kind

        else:
            raise ValueError('The figure kind must be set in config.ini file!')

    def __set_figure_kind(self):
        """
        Set the figure kind.
        :param:
        :return:
        """
        self.figure_kind = self.__get_figure_kind()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_figure_kind() ......')
            printwp('self.figure_kind: ' + str(self.figure_kind))

    def __get_y_label_type(self):
        """
        Get the y label type from the command line arguments.
        :return: y_label_type
        """

        if if_none(self.configini.get('Default', 'y_label_type')) is not None:
            y_label_type_str = self.configini.get('Default', 'y_label_type')
            y_label_type = eval(y_label_type_str)

            return y_label_type

        else:
            return None

    def __set_y_label_type(self):
        """
        Set the y label type.
        :param :
        :return:
        """
        self.y_label_type = self.__get_y_label_type()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_y_label_type() ......')
            printwp('self.y_label_type: ' + str(self.y_label_type))

    def __get_x_label_type(self):
        """
        Get the x label type from the command line arguments.
        :return: x_label_type
        """
        if if_none(self.configini.get('Default', 'x_label_type')) is not None:
            x_label_type_str = self.configini.get('Default', 'x_label_type')
            x_label_type = eval(x_label_type_str)

            return x_label_type

        else:
            return None

    def __set_x_label_type(self):
        """
        Set the x label type.
        :param :
        :return:
        """
        self.x_label_type = self.__get_x_label_type()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_x_label_type() ......')
            printwp('self.x_label_type: ' + str(self.x_label_type))

    def __get_y_ticklabel_format(self):
        """
        Get the y label type from the command line arguments.
        :return: y_label_type
        """

        if if_none(self.configini.get('Default', 'y_ticklabel_format')) is not None:
            y_ticklabel_format_str = self.configini.get('Default', 'y_ticklabel_format')
            y_ticklabel_format = eval(y_ticklabel_format_str)

            return y_ticklabel_format

        else:
            return None

    def __set_y_ticklabel_format(self):
        """
        Set the y tick label format.
        :return:
        """
        self.y_ticklabel_format = self.__get_y_ticklabel_format()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_y_ticklabel_format() ......')
            printwp('self.y_ticklabel_format: ' + str(self.y_ticklabel_format))

    def __get_x_ticklabel_format(self):
        """
        Get the x label type from the command line arguments.
        :return: x_label_type
        """
        if if_none(self.configini.get('Default', 'x_ticklabel_format')) is not None:
            x_ticklabel_format_str = self.configini.get('Default', 'x_ticklabel_format')
            x_ticklabel_format = eval(x_ticklabel_format_str)

            return x_ticklabel_format

        else:
            return None

    def __set_x_ticklabel_format(self):
        """
        Set the x tick label format.
        :return:
        """
        self.x_ticklabel_format = self.__get_x_ticklabel_format()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_x_ticklabel_format() ......')
            printwp('self.x_ticklabel_format: ' + str(self.x_ticklabel_format))

    # def __get_side_ylabel(self):
    #     """
    #     Get the side y label from the command line arguments.
    #     :return: side_ylabel
    #     """
    #     if self.configini is not None:
    #         if if_none(self.configini.get('Default', 'side_ylabel')) is not None:
    #             side_ylabel_str = self.configini.get('Default', 'side_ylabel')
    #             side_ylabel = eval(side_ylabel_str)
    #             return side_ylabel
    #
    #         else:
    #             return None
    #
    #     elif self.parser is not None:
    #         if if_none(self.parser.parse_args().side_ylabel) is not None:
    #             side_ylabel_str = self.parser.parse_args().side_ylabel
    #             side_ylabel = eval(side_ylabel_str)
    #             return side_ylabel
    #
    #         else:
    #             return None

    # def __set_side_ylabel(self):
    #     side_ylabel = self.__get_side_ylabel()
    #     self.legend_side_index = {}
    #     for num in range(self.figure_number):
    #         if self.label_len_list[num] == 3:
    #             legend_side_index = [self.legend[num].index(side) for side in self.side_ylabel[num]]
    #             self.legend_side_index[num] = legend_side_index
    #         else:
    #             self.legend_side_index[num] = []

    def __get_mixed_ylegend(self):
        """
        Get the mixed y legend from the command line arguments.
        :return: mixed_ylegend
        """
        if if_none(self.configini.get('Default', 'mixed_ylegend')) is not None:
            mixed_ylegend_str = self.configini.get('Default', 'mixed_ylegend')
            mixed_ylegend = eval(mixed_ylegend_str)

            return mixed_ylegend
        else:
            return None

    def __set_mixed_ylegend(self):
        """
        Set the mixed y legend.
        :param :
        :return:
        """
        self.mixed_ylegend = self.__get_mixed_ylegend()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_mixed_ylegend() ......')
            printwp('self.mixed_ylegend: ' + str(self.mixed_ylegend))

    def __get_title(self):
        """
        Get the title from the command line arguments.
        :return: title
        """
        if if_none(self.configini.get('Default', 'title')) is not None:
            title = self.configini.get('Default', 'title')

            return title

        else:
            return None

    def __set_title(self):
        """
        Set the title.
        :param:
        :return:
        """
        self.title = self.__get_title()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_title() ......')
            printwp('self.title: ' + str(self.title))

    def __get_line_with_marker(self):
        """
        Get the line with marker from the command line arguments.
        :return: line_with_marker
        """
        if if_false(self.configini.get('Default', 'line_with_marker')) is not False:
            line_with_marker_str = self.configini.get('Default', 'line_with_marker')
            line_with_marker = eval(line_with_marker_str)

            return line_with_marker

        else:
            return False

    def __set_line_with_marker(self):
        """
        Set the line with marker.
        :param:
        :return:
        """
        self.line_with_marker = self.__get_line_with_marker()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_line_with_marker() ......')
            printwp('self.line_with_marker: ' + str(self.line_with_marker))

    def __get_legend_location(self):
        """
        Get the legend from the command line arguments.
        :return: legend
        """
        if if_none(self.configini.get('Default', 'legend_location')) is not None:
            legend_location_str = self.configini.get('Default', 'legend_location')
            legend_location = eval(legend_location_str)

            return legend_location

        else:
            return None

    def __set_legend_location(self):
        """
        Set the legend.
        :return:
        """
        self.legend_location = self.__get_legend_location()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_legend_location() ......')
            printwp('self.legend_location: ' + str(self.legend_location))

    def __set_scale_factor(self):
        """
        Set the scale factor.
        :return:
        """
        self.scale_factor = 1.0

    def __set_skip_rows(self):
        """
        Set the skip rows.
        :return:
        """
        self.skip_rows = 0

    def __get_limited_label(self):
        """
        Get the limited label from the command line arguments.
        :return:
        """
        if if_none(self.configini.get('Default', 'limited_label')) is not None:
            limited_label_str = self.configini.get('Default', 'limited_label')
            limited_label = eval(limited_label_str)

            return limited_label
        else:
            return None

    def __set_limited_label(self):
        """
        Set the limited label.
        :return:
        """
        self.limited_label = self.__get_limited_label()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_limited_label() ......')
            printwp('self.limited_label: ' + str(self.limited_label))

    def __get_color_map(self):
        """
        Get the color map from the command line arguments.
        :return:
        """
        if if_none(self.configini.get('Default', 'color_map')) is not None:
            color_map_str = self.configini.get('Default', 'color_map')
            color_map = eval(color_map_str)

            return color_map

        else:
            return ['viridis'] * len(self.limited_label)

    def __set_color_map(self):
        """
        Set the color map.
        :return:
        """
        self.color_map = self.__get_color_map()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_color_map() ......')
            printwp('self.color_map: ' + str(self.color_map))

    def __get_figure_matrix(self):
        """
        Get the figure matrix from the command line arguments.
        :return:
        """
        if if_none(self.configini.get('Default', 'figure_matrix')) is not None:
            figure_matrix_str = self.configini.get('Default', 'figure_matrix')
            figure_matrix = eval(figure_matrix_str)

            return figure_matrix

        else:
            return [1, 1]

    def __set_figure_matrix(self):
        """
        Set the figure matrix.
        :return:
        """

        self.rows = self.__get_figure_matrix()[0]
        self.cols = self.__get_figure_matrix()[1]

        if self.debug_mode is True:
            printwp('executing Parameter.__set_figure_matrix() ......')
            printwp('self.rows: ' + str(self.rows))
            printwp('self.cols: ' + str(self.cols))

    def __get_figure_size(self):
        """
        Get the figure size from the command line arguments.
        :return:
        """
        if if_none(self.configini.get('Default', 'figure_size')) is not None:
            figure_size_str = self.configini.get('Default', 'figure_size')
            figure_size = eval(figure_size_str)

            return figure_size
        else:
            return [8, 6]

    def __set_figure_size(self):
        """
        Set the figure size.
        :return:
        """
        self.width = self.__get_figure_size()[0]
        self.height = self.__get_figure_size()[1]

        if self.debug_mode is True:
            printwp('executing Parameter.__set_figure_size() ......')
            printwp('self.width: ' + str(self.width))
            printwp('self.height: ' + str(self.height))

    def __get_multi_line_style(self):
        """
        Get the line style from the command line arguments.
        :return:
        """
        if if_false(self.configini.get('Default', 'multi_line_style')) is not False:
            multi_line_style_str = self.configini.get('Default', 'multi_line_style')
            multi_line_style = eval(multi_line_style_str)
            return multi_line_style
        else:
            return False

    def __set_line_style(self):
        """
        Set the line style.
        :return:
        """
        self.multi_line_style = self.__get_multi_line_style()

        if self.debug_mode is True:
            printwp('executing Parameter.__set_line_style() ......')
            printwp('self.multi_line_style: ' + str(self.multi_line_style))

    def __get_debug(self):
        """
        Get the debug from the command line arguments.
        :return:
        """
        if if_none(self.configini.get('Default', 'debug_mode')) is not None:
            debug_mode = self.configini.getboolean('Default', 'debug_mode')
            return debug_mode

    def __set_debug_mode(self):
        """
        Set the debug mode.
        :return:
        """
        self.debug_mode = self.__get_debug()
        if self.debug_mode:
            printwp('Debug mode is on.')
