
from whosplot.utility import *
from whosplot.style import Style
from whosplot.parameter import Parameter
from whosplot.load import CsvReader
from whosplot.abstract import Abstract
import numpy as np
import time
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
        param_attrs = ['file_location', 'cols', 'rows', 'color_map']
        csv_attrs = ['data', 'figure_number', 'legend_len_list', 'legend', 'kind_index']
        style_attrs = ['marker', 'color_map_array', 'line_style', 'line_width']

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

    def save_fig(self, fig_format='jpg'):
        self.plt.savefig('{}.{}'.format(self.__file_location.replace('.csv', ''), fig_format))
        # self.plt.savefig('{}-eps-converted-to.pdf'.format(self.__file_location.replace('.csv', '')))
        # self.plt.savefig('{}.eps'.format(self.__file_location.replace('.csv', '')))
        time.sleep(1)
        printwp('Figure saved successfully.')
        printer_septal_line()

    def show(self, fig_format='jpg'):
        os.startfile(self.__file_location.replace('.csv', f'.{fig_format}'))

    def __scatter(self, fig_num, line_num, color_array):
        self.axs[int(fig_num // self.__cols), int(fig_num % self.__cols)].scatter(
            self.__data[fig_num][:, line_num * 2],
            self.__data[fig_num][:, int(line_num * 2 + 1)],
            color=color_array[fig_num, line_num],
            marker=self.__marker[fig_num][line_num],
            clip_on=False,
            zorder=100)

    def __scatter_cmap(self, fig_num, line_num, color_array):
        self.axs[int(fig_num // self.__cols), int(fig_num % self.__cols)].scatter(
            self.__data[fig_num][:, 0],
            self.__data[fig_num][:, int(line_num * 2 + 1)],
            c=color_array[fig_num, line_num],
            cmap=self.__color_map[fig_num],
            marker=self.__marker[fig_num][line_num],
            clip_on=False,
            zorder=100)

    def __plot(self, fig_num, line_num, color_array):
        self.axs[
            int(fig_num // self.__cols), int(fig_num % self.__cols)
        ].plot(
            self.__data[fig_num][:, line_num * 2],
            self.__data[fig_num][:, int(line_num * 2 + 1)],
            color=color_array[fig_num, line_num],
            linestyle=self.__line_style[fig_num][line_num],
            linewidth=self.__line_width[fig_num][line_num],
            marker=self.__marker[fig_num][line_num],
            markersize=8,
            markerfacecolor=None,
            markeredgecolor=None,
            markeredgewidth=1,
            alpha=1,
            clip_on=False,
            zorder=100
        )

    def __colorbar(self, fig_num, data, label):
        ax = self.axs[int(fig_num // self.__cols), int(fig_num % self.__cols)]
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
            format='%.1f',
            ticks=ticks,
            orientation='vertical',
            drawedges=False)
        cbar.outline.set_linewidth(0)
        cbar.ax.tick_params(direction='out', length=2, width=0.5, labelsize=12, pad=1)

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
                size=fontsize)

    def set_axis_off(self):
        """Turn off the axis of the plot."""
        for fig_num in range(self.__figure_number):
            self.axs[int(fig_num // self.__cols), int(fig_num % self.__cols)].set_axis_off()

    def __figure_serial(self, fig_num, use_tex):
        if use_tex:
            serial = r'\textbf{(' + alphabet[fig_num] + ')}'
        else:
            serial = r'(' + alphabet[fig_num] + ')',
        self.axs[
            int(fig_num // self.__cols), int(fig_num % self.__cols)
        ].annotate(
            serial,
            xy=(0.08, 0.93),
            xycoords='axes fraction',
            xytext=(0, 0),
            textcoords='offset pixels',
            horizontalalignment='right',
            verticalalignment='top',
            size=18)

    def __cal_color_map_array(self, color_gradient: list, func='Euclidean_distance'):
        color_array = {}

        for fig_num in range(self.__figure_number):
            legend_len = self.__legend_len_list[fig_num]
            color_map_array_len = len(self.__color_map_array[fig_num]) - 1

            for line_num in range(legend_len):
                if color_gradient[fig_num][line_num] and func == 'Euclidean_distance':
                    data_mmnorm = mmnorm(self.__data[fig_num], axis=1)
                    x = data_mmnorm[:, 0]
                    y = data_mmnorm[:, int(line_num * 2 + 1)]
                    dist_ = cal2ddist(x, y)
                    color_array[fig_num, line_num] = dist_

                elif color_gradient[fig_num][line_num] and func != 'Euclidean_distance':
                    raise ValueError('The function is not supported yet.')

                else:
                    index_ = int(line_num * color_map_array_len / max((legend_len - 1), 1))
                    color_array[fig_num, line_num] = self.__color_map_array[fig_num][index_]
        return color_array

    def __cal_color_gradient(self, switch=False):
        color_gradient = []
        for fignum in range(self.__figure_number):
            color_gradient.append([])
            for line_num in range(self.__legend_len_list[fignum]):
                if 'color_gradient' in self.__legend[fignum][int((line_num * 2) + 1)] and switch:
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

    def color_gradient_two_d_subplots(self):
        """Create 2D subplots with color gradients."""
        color_gradient = self.__cal_color_gradient(switch=True)
        color_array = self.__cal_color_map_array(color_gradient)

        for fig_num in range(self.__figure_number):
            legend_len = self.__legend_len_list[fig_num]
            scatter_index = self.__kind_index[fig_num]
            for line_num in range(legend_len):
                if int((line_num * 2) + 1) not in scatter_index:
                    self.__plot(fig_num, line_num, color_array)

                elif color_gradient[fig_num][line_num]:
                    self.__scatter_cmap(fig_num, line_num, color_array)
                    self.__colorbar(fig_num, color_array[fig_num, line_num], 'Euclidean distance')

                else:
                    self.__scatter(fig_num, line_num, color_array)

                if self.__figure_number > 1:
                    self.__figure_serial(fig_num, use_tex=self.plt.rcParams['text.usetex'])

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

    def pie_chart(self):
        pass

    def histogram(self):
        pass

    def box_plot(self):
        pass

    def violin_plot(self):
        pass


    
    