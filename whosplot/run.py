
from whosplot.calculator import calculator
from whosplot.utility import *
from whosplot.style import Style
from whosplot.parameter import Parameter
from whosplot.load import CsvReader
from whosplot.abstract import Abstract
from matplotlib.colors import Normalize, LogNorm, SymLogNorm

import numpy as np
import sys, subprocess, os
import os


class Run(Abstract):
    """
    A class to handle the execution and plotting of CSV data.

    Attributes:
    ----------
    axs : matplotlib.axes._axes.Axes
        Array of subplot axes.
    fig : matplotlib.figure.Figure
        Figure object for plotting.
    plt : module
        Matplotlib.pyplot module.

    Methods:
    -------
    save_fig(fig_format='jpg')
        Save the generated figure in the specified format.
    show(fig_format='jpg')
        Display the saved figure file.
    text(text, xypos, horizontalalignment='center', verticalalignment='center', fontsize=18)
        Annotate text on the plot.
    set_axis_off()
        Turn off the axis of the plot.
    two_d_subplots()
        Create 2D subplots for the data.
    color_gradient_two_d_subplots()
        Create 2D subplots with color gradients.
    """

    def __init__(self):
        super(Run, self).__init__()
        self.__Parameter = Parameter()
        self.__CsvReader = CsvReader()
        self.__Style = Style()
        self.axs = self.__Style.axs
        self.fig = self.__Style.fig
        self.plt = self.__Style.plt
        self.__set_attributes()
        print_header()

    def __set_attributes(self):
        """
        Set various attributes for the execution and plotting of CSV data.
        """
        param_attrs = ['file_location', 'cols', 'rows', 'color_map', 'figure_kind', 'width', 'height']
        csv_attrs = ['data', 'figure_number', 'legend_len_list', 'legend', 'kind_index']
        style_attrs = [
            'marker', 'color_map_array', 'line_style', 'line_width',
            'scatter_size', 'markersize', 'markeredgewidth', 'frame_linewidth', 'scale_per_fig'
        ]
        for attr in param_attrs:
            setattr(self, f"_Run__{attr}", getattr(self.__Parameter, attr))
        for attr in csv_attrs:
            setattr(self, f"_Run__{attr}", getattr(self.__CsvReader, attr))
        for attr in style_attrs:
            setattr(self, f"_Run__{attr}", getattr(self.__Style, attr))

    def __set_axis_style(self, **kwargs):
        """
        Set the axis style of the plot.
        """
        self.__Style.set_axis_style(**kwargs)

    def save_fig(self, fig_name=None, fig_format='jpg'):
        if fig_name is None:
            fig_name = self.__file_location.replace('.csv', '')
        self.plt.savefig('{}.{}'.format(fig_name, fig_format))
        printwp('Figure saved successfully.')
        printer_septal_line()

    def show(self, fig_format='jpg'):
        
        p = self.__file_location.replace('.csv', f'.{fig_format}')
        if sys.platform.startswith('win'): os.startfile(p)
        elif sys.platform == 'darwin': subprocess.run(['open', p], check=False)
        else: subprocess.run(['xdg-open', p], check=False)


    def __scatter(self, fig_num, line_num, color_array):
        ax = self.axs[int(fig_num // self.__cols), int(fig_num % self.__cols)]
        ax.scatter(
            self.__data[fig_num][:, line_num * 2],
            self.__data[fig_num][:, int(line_num * 2 + 1)],
            color=color_array[fig_num, line_num],
            marker=self.__marker[fig_num][line_num],
            s=self.__scatter_size[fig_num],
            linewidths=self.__markeredgewidth[fig_num],
            clip_on=False,
            zorder=100
        )

    def __scatter_cmap(self, fig_num, line_num, color_array):
        ax = self.axs[int(fig_num // self.__cols), int(fig_num % self.__cols)]
        ax.scatter(
            self.__data[fig_num][:, line_num * 2],
            self.__data[fig_num][:, int(line_num * 2 + 1)],
            c=color_array[fig_num, line_num],
            cmap=self.__color_map[fig_num],
            marker=self.__marker[fig_num][line_num],
            s=self.__scatter_size[fig_num],
            linewidths=self.__markeredgewidth[fig_num],
            clip_on=False,
            zorder=100
        )

    def __plot(self, fig_num, line_num, color_array):
        ax = self.axs[int(fig_num // self.__cols), int(fig_num % self.__cols)]
        ax.plot(
            self.__data[fig_num][:, line_num * 2],
            self.__data[fig_num][:, int(line_num * 2 + 1)],
            color=color_array[fig_num, line_num],
            linestyle=self.__line_style[fig_num][line_num],
            linewidth=self.__line_width[fig_num][line_num],
            marker=self.__marker[fig_num][line_num],
            markersize=self.__markersize[fig_num],
            markerfacecolor=None,
            markeredgecolor=None,
            markeredgewidth=self.__markeredgewidth[fig_num],
            alpha=1,
            clip_on=False,
            zorder=100
        )

    def __colorbar(self, fig_num, data, label, orientation='vertical'):
        ax = self.axs[int(fig_num // self.__cols), int(fig_num % self.__cols)]
        if data is None:
            max_ = 0
            min_ = 0
        else:
            max_ = np.max(data)
            min_ = np.min(data)
        ticks = np.linspace(min_, max_, 5)
        cbar = self.fig.colorbar(
            self.plt.cm.ScalarMappable(
                norm=self.plt.Normalize(vmin=min_, vmax=max_, clip=True),
                cmap=self.__color_map[fig_num]),
            ax=ax,
            label=label,
            fraction=0.04,
            pad=0.01,
            aspect=20,
            format='%.2f',
            ticks=ticks,
            orientation=orientation,
            drawedges=False)
        cbar.outline.set_linewidth(0)
        cbar.ax.tick_params(
            direction='out', 
            length=2 * self.__scale_per_fig[fig_num], 
            width=0.5 * self.__scale_per_fig[fig_num], 
            labelsize=12 * self.__scale_per_fig[fig_num], 
            pad=1 * self.__scale_per_fig[fig_num]
            )

    def text(self, text, xypos, horizontalalignment='center', verticalalignment='center', fontsize=18):
        """
        Annotate text on the plot.

        :param text: List of text annotations.
        :param xypos: Tuple of x and y positions.
        :param horizontalalignment: Horizontal alignment of text (default is 'center').
        :param verticalalignment: Vertical alignment of text (default is 'center').
        :param fontsize: Font size of text (default is 18).
        """
        for fig_num in range(self.__figure_number):
            self.axs[
                int(fig_num // self.__cols), int(fig_num % self.__cols)].annotate(
                text[fig_num],
                xy=xypos,
                xycoords='axes fraction',
                xytext=(0, 0),
                textcoords='offset pixels',
                horizontalalignment=horizontalalignment,
                verticalalignment=verticalalignment,
                size=fontsize,
                zorder=100)

    def set_axis_off(self):
        """Turn off the axis of the plot."""
        for fig_num in range(self.__figure_number):
            self.axs[int(fig_num // self.__cols), int(fig_num % self.__cols)].set_axis_off()

    def __figure_serial(self, fig_num, use_tex):
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        if use_tex:
            serial = r'\textbf{(' + alphabet[fig_num] + ')}'
        else:
            serial = '(' + alphabet[fig_num] + ')'
        ax = self.axs[int(fig_num // self.__cols), int(fig_num % self.__cols)]
        ax.annotate(
            serial,
            xy=(0.08, 0.93),
            xycoords='axes fraction',
            xytext=(0, 0),
            textcoords='offset pixels',
            horizontalalignment='right',
            verticalalignment='top',
            size=self.__scale_per_fig[fig_num] * 18
        )

    def __cal_color_map_array(self, color_gradient: list, func='Euclidean_distance', sort_value=None):
        color_array = {}

        for fig_num in range(self.__figure_number):
            legend_len = self.__legend_len_list[fig_num]
            color_map_array_len = len(self.__color_map_array[fig_num]) - 1

            if self.__figure_kind[fig_num] == 'mix':
                scatter_index_ = self.__kind_index[fig_num]
                line_index_list = [int(i) for i in range(legend_len) if int((i * 2) + 1) not in scatter_index_] 
                scatter_index_list = [int((i - 1) / 2) for i in scatter_index_]
                for scatter_num in range(len(scatter_index_list)):
                    index_ = int(scatter_num * color_map_array_len / max((len(scatter_index_list) - 1), 1))
                    color_array[fig_num, scatter_index_list[scatter_num]] = self.__color_map_array[fig_num][index_]
                for line_num in range(len(line_index_list)):
                    index_ = int(line_num * color_map_array_len / max((len(line_index_list) - 1), 1))
                    color_array[fig_num, line_index_list[line_num]] = self.__color_map_array[fig_num][index_]

            else:
                for line_num in range(legend_len):
                    if color_gradient[fig_num][line_num] and func == 'Euclidean_distance':
                        data_mmnorm = calculator(method='mmnorm', data=self.__data[fig_num], axis=1)
                        x = data_mmnorm[:, 0]
                        y = data_mmnorm[:, int(line_num * 2 + 1)]
                        dist_ = calculator(method='dist', data=np.array([x, y]).T, axis=0, power=2)
                        color_array[fig_num, line_num] = dist_

                    elif color_gradient[fig_num][line_num] and func != 'Euclidean_distance':
                        color_array[fig_num, line_num] = sort_value[fig_num, line_num]

                    else:
                        index_ = int(line_num * color_map_array_len / max((legend_len - 1), 1))
                        color_array[fig_num, line_num] = self.__color_map_array[fig_num][index_]

        return color_array

    def __cal_color_gradient(self, switch=False):
        color_gradient = []
        for fignum in range(self.__figure_number):
            color_gradient.append([])
            for line_num in range(self.__legend_len_list[fignum]):
                if switch:
                    color_gradient[fignum].append(True)
                else:
                    color_gradient[fignum].append(False)
        return color_gradient

    def two_d_subplots(self, filling=False, **kwargs):
        """Create 2D subplots for the data."""
        color_gradient = self.__cal_color_gradient()
        color_array = self.__cal_color_map_array(color_gradient)
        for fig_num in range(self.__figure_number):
            legend_len = self.__legend_len_list[fig_num]
            scatter_index = self.__kind_index[fig_num]
            for line_num in range(legend_len):
                if int((line_num * 2) + 1) not in scatter_index:
                    self.__plot(fig_num, line_num, color_array)

                else:
                    self.__scatter(fig_num, line_num, color_array)

                if self.__figure_number > 1:
                    self.__figure_serial(fig_num, use_tex=self.plt.rcParams['text.usetex'])

            if filling:
                try:
                    filling_data = kwargs['filling_data']
                    extra_legend = kwargs['extra_legend']
                except KeyError:
                    raise ValueError('The filling data and label list are required.')
                self.__filling(filling_data, extra_legend, fig_num, color_array)

        self.__set_axis_style(**kwargs)

    def color_gradient_two_d_subplots(self, func='Euclidean_distance', sort_value=None, data_range=None):
        """Create 2D subplots with color gradients."""
        color_gradient = self.__cal_color_gradient(switch=True)
        color_array = self.__cal_color_map_array(color_gradient, func, sort_value)

        for fig_num in range(self.__figure_number):
            legend_len = self.__legend_len_list[fig_num]
            scatter_index = self.__kind_index[fig_num]
            for line_num in range(legend_len):
                if int((line_num * 2) + 1) not in scatter_index:
                    self.__plot(fig_num, line_num, color_array)

                elif color_gradient[fig_num][line_num]:
                    self.__scatter_cmap(fig_num, line_num, color_array)

                else:
                    self.__scatter(fig_num, line_num, color_array)

                if self.__figure_number > 1:
                    self.__figure_serial(fig_num, use_tex=self.plt.rcParams['text.usetex'])
            if color_gradient[fig_num][0] and data_range is not None:
                self.__colorbar(fig_num, data_range, func, orientation='horizontal')

        self.__set_axis_style()

    def __filling(self, filling_data, label_list, fig_num, color_array):
        number = filling_data[fig_num].shape[1] / 4

        for i in range(int(number)):
            self.axs[
                int(fig_num // self.__cols), int(fig_num % self.__cols)
            ].fill_between(
                x = filling_data[fig_num][:, i * 4],
                y1 = filling_data[fig_num][:, int(i * 4 + 1)],
                y2 = filling_data[fig_num][:, int(i * 4 + 3)],
                color=color_array[fig_num, i],
                alpha=0.5,
                zorder=0,
                linewidth=0,
                label=label_list[fig_num][i]
            )

    def pv_screenshot_arrange(self, img_path, label_type, data_range, label_name, tick_type=None, fmt=None, tick_num=None, colorbar_off=False):
        """
        Arrange screenshots with optional colorbars based on label types:
        - label_type 1: Individual colorbar for each subplot.
        - label_type 2: A single colorbar for the entire figure.
        - label_type 3: Custom colorbar arrangement with equal image dimensions.

        Args:
        - img_path (list): List of image file paths.
        - label_type (int): Type of colorbar arrangement (1, 2, or 3).
        - data_range: For label_type 1, list of [min, max] pairs for each subplot;
                    For label_type 2, a [min, max] pair for the entire figure;
                    For label_type 3, list of [min, max] pairs for each colorbar.
        - label_name: For label_type 1, list of labels for each subplot;
                    For label_type 2, label for the entire figure;
                    For label_type 3, list of labels for each colorbar.
        - tick_type: For label_type 1 and 2, 'linear' or 'log';
                    For label_type 3, list of 'linear' or 'log' for each colorbar.
        - fmt (list): Format string for colorbar tick labels.
        - tick_num (list): Number of ticks on the colorbar.
        - colorbar_off (bool): If True, colorbars are not displayed.

        Raises:
        - ValueError: If the number of images does not match the number of subplots.
        """
        if tick_num is None:
            tick_num = [6 for _ in range(100)]
        if fmt is None:
            fmt = [r'%.2f' for _ in range(100)]
        if tick_type is None:
            tick_type = ['linear' for _ in range(100)]

        if len(img_path) != self.__figure_number:
            raise ValueError('The number of images does not match the number of subplots.')
        
        if label_type == 1:
            # Handle tick_type for individual subplots
            for fig_num in range(self.__figure_number):
                min_, max_ = data_range[fig_num]
                if tick_type[fig_num] == 'log':
                    norm = SymLogNorm(vmin=min_, vmax=max_, linthresh=1e-8)
                    ticks = np.logspace(np.log10(min_), np.log10(max_), tick_num[fig_num])
                else:
                    norm = Normalize(vmin=min_, vmax=max_)
                    ticks = np.linspace(min_, max_, tick_num[fig_num])
                img = figop_remove_margin(img_path[fig_num])

                self.axs[
                            int(fig_num // self.__cols), int(fig_num % self.__cols)
                        ].imshow(np.array(img))
                self.axs[
                            int(fig_num // self.__cols), int(fig_num % self.__cols)
                        ].axis('off')
                
                if colorbar_off:
                    continue

                if self.__figure_number > 1:
                    serial = r'\textbf{(' + alphabet[fig_num] + ')}'
                    self.axs[
                        int(fig_num // self.__cols), int(fig_num % self.__cols)
                    ].annotate(
                        serial,
                        xy=(0.08, 1.05),
                        xycoords='axes fraction',
                        xytext=(0, 0),
                        textcoords='offset pixels',
                        horizontalalignment='right',
                        verticalalignment='top',
                        size=18)

                self.__colorbar(fig_num, data_range[fig_num], label_name[fig_num], 'horizontal')
            
        elif label_type == 2:

            min_, max_ = data_range[0]
            if tick_type[0] == 'log':
                norm = SymLogNorm(vmin=min_, vmax=max_, linthresh=1e-8)
                ticks = np.logspace(np.log10(min_), np.log10(max_), tick_num[0])
            else:
                norm = Normalize(vmin=min_, vmax=max_)
                ticks = np.linspace(min_, max_, tick_num[0])

            fig_ratio = [
                figop_remove_margin(img_path[fig_num]).size[1] / figop_remove_margin(img_path[fig_num]).size[0]
                for fig_num in range(self.__figure_number)
            ]
            mean_ratio = np.mean(fig_ratio)

            for fig_num in range(self.__figure_number):

                img = figop_remove_margin(img_path[fig_num])
                width = self.__width
                extent = [0, 0.95 * width, 0, 0.95 * width * mean_ratio]
                self.axs[
                            int(fig_num // self.__cols), int(fig_num % self.__cols)
                        ].imshow(np.array(img), extent=extent)
                self.axs[
                            int(fig_num // self.__cols), int(fig_num % self.__cols)
                        ].axis('off')

                if self.__figure_number > 1:
                    serial = r'\textbf{(' + alphabet[fig_num] + ')}'

                    self.axs[
                        int(fig_num // self.__cols), int(fig_num % self.__cols)
                    ].annotate(
                        serial,
                        xy=(0.08, 1.05),
                        xycoords='axes fraction',
                        xytext=(0, 0),
                        textcoords='offset pixels',
                        horizontalalignment='right',
                        verticalalignment='top',
                        size=18)
                
            if colorbar_off:
                return 0

            fig_width, fig_height = self.fig.get_size_inches()
            cbar_width = fig_width * 0.5

            cbar_left = (fig_width - cbar_width) / 2
            cbar_ax = self.fig.add_axes([cbar_left / fig_width, 0.05, cbar_width / fig_width, 0.02])  # [left, bottom, width, height] 
            sm = self.plt.cm.ScalarMappable(norm=norm, cmap=self.__color_map[fig_num])
            cbar = self.fig.colorbar(
                sm, cax=cbar_ax, orientation='horizontal', drawedges=False,
                fraction=0.04, pad=0.01, aspect=20, format=fmt[0], ticks=ticks
            )
            cbar.set_label(label_name[0])
            cbar.outline.set_linewidth(0)
            cbar.ax.tick_params(
                direction='out', length=2, width=0.5, labelsize=12, pad=0
            )
            cbar.ax.xaxis.set_label_position('bottom')

        elif label_type == 3:

            if len(tick_type) != len(data_range) and len(tick_type) != 100:
                raise ValueError('The number of tick types does not match the number of colorbars.')
            
            # Handle tick_type for custom colorbar arrangement
            num_colorbars = len(data_range)
            if isinstance(tick_type, str):
                tick_type_list = [tick_type] * num_colorbars
            else:
                tick_type_list = tick_type

            fig_ratio = [
                figop_remove_margin(img_path[fig_num]).size[1] / figop_remove_margin(img_path[fig_num]).size[0]
                for fig_num in range(self.__figure_number)
            ]
            mean_ratio = np.mean(fig_ratio)
            for fig_num in range(self.__figure_number):
                img = figop_remove_margin(img_path[fig_num])
                width = self.__width
                extent = [0, 0.95 * width, 0, 0.95 * width * mean_ratio]
                self.axs[
                    int(fig_num // self.__cols), int(fig_num % self.__cols)
                ].imshow(np.array(img), extent=extent)
                self.axs[
                    int(fig_num // self.__cols), int(fig_num % self.__cols)
                ].axis('off')

                if self.__figure_number > 1:
                    serial = r'\textbf{(' + alphabet[fig_num] + ')}'
                    self.axs[
                        int(fig_num // self.__cols), int(fig_num % self.__cols)
                    ].annotate(
                        serial,
                        xy=(0.08, 1.05),
                        xycoords='axes fraction',
                        xytext=(0, 0),
                        textcoords='offset pixels',
                        horizontalalignment='right',
                        verticalalignment='top',
                        size=18)
            
            padding_left = 0.05  # Padding on the left side of the figure
            padding_between = 0.05  # Space between adjacent colorbars
            total_cbar_area = 0.9 - (num_colorbars - 1) * padding_between
            cbar_width = total_cbar_area / num_colorbars
            cbar_height = 0.02
            cbar_bottom = 0.05

            for label_num in range(num_colorbars):
                min_, max_ = data_range[label_num]
                if tick_type_list[label_num] == 'log':
                    norm = SymLogNorm(vmin=min_, vmax=max_, linthresh=1e-14)
                    ticks = np.logspace(np.log10(min_), np.log10(max_), tick_num[label_num])
                else:
                    norm = Normalize(vmin=min_, vmax=max_)
                    ticks = np.linspace(min_, max_, tick_num[label_num])

                fig_width, fig_height = self.fig.get_size_inches()
                cbar_left = padding_left + label_num * (cbar_width + padding_between)
                cbar_ax = self.fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
                sm = self.plt.cm.ScalarMappable(norm=norm, cmap=self.__color_map[label_num])
                cbar = self.fig.colorbar(
                    sm, cax=cbar_ax, orientation='horizontal', drawedges=False,
                    fraction=0.04, pad=0.01, aspect=20, format=fmt[label_num], ticks=ticks
                )
                cbar.set_label(label_name[label_num])
                cbar.outline.set_linewidth(0)
                cbar.ax.tick_params(
                    direction='out', length=2, width=0.5, labelsize=12, pad=0
                )
                cbar.ax.xaxis.set_label_position('bottom')

    def pie_chart(self):
        pass

    def histogram(self):
        pass

    def box_plot(self):
        pass

    def violin_plot(self):
        pass


    
    