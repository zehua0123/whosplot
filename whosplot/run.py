
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
        self.__attribute()
        print_header()

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

    def __set_data(self) -> None:
        """
        Get the data of the csv file.
        :return: data
        """
        self.__data = self.__CsvReader.__getattr__('data')

    def __set_color_map(self) -> None:
        """
        Get the color map of the csv file.
        :return: color map
        """
        self.__color_map = self.__Parameter.__getattr__('color_map')

    def __set_figure_number(self) -> None:
        """
        Get the figure number of the csv file.
        :return: figure number
        """
        self.__figure_number = self.__CsvReader.__getattr__('figure_number')

    def __set_legend_len_list(self) -> None:
        """
        Get the legend len list of the csv file.
        :return: legend len list
        """
        self.__legend_len_list = self.__CsvReader.__getattr__('legend_len_list')

    def __set_legend(self) -> None:
        """
        Get the legend of the csv file.
        :return: legend
        """
        self.__legend_ = self.__CsvReader.__getattr__('legend')

    def __set_kind_index(self):
        """
        Get the kind index of the csv file.
        :return: kind index
        """
        self.__kind_index = self.__CsvReader.__getattr__('kind_index')

    def __set_marker(self):
        """
        Get the marker of the csv file.
        :return: marker
        """
        self.__marker = self.__Style.__getattr__('marker')

    def __set_color_map_array(self):
        """
        Get the color map array of the csv file.
        :return: color map array
        """
        self.__color_map_array_ = self.__Style.__getattr__('color_map_array')

    def __set_axis_style(self):
        """
        Set the axis style of the csv file.
        :return: None
        """
        self.__Style.set_axis_style()

    def __set_line_style(self):
        """
        Set the line style of the csv file.
        :return: None
        """
        self.__line_style = self.__Style.__getattr__('line_style')

    def __set_line_width(self):
        """
        Set the line width of the csv file.
        :return: None
        """
        self.__line_width = self.__Style.__getattr__('line_width')

    def __attribute(self):
        self.__set_file_location()
        self.__set_cols_and_rows()
        self.__set_data()
        self.__set_color_map()
        self.__set_figure_number()
        self.__set_legend_len_list()
        self.__set_legend()
        self.__set_kind_index()
        self.__set_marker()
        self.__set_color_map_array()
        self.__set_line_style()
        self.__set_line_width()

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

    def __color_map_array(self, color_gradient: list, func='Euclidean_distance'):
        color_array = {}

        for fig_num in range(self.__figure_number):
            legend_len = self.__legend_len_list[fig_num]
            color_map_array_len = len(self.__color_map_array_[fig_num]) - 1

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
                    color_array[fig_num, line_num] = self.__color_map_array_[fig_num][index_]
        return color_array

    def __color_gradient(self, switch=False):
        color_gradient = []
        for fignum in range(self.__figure_number):
            color_gradient.append([])
            for line_num in range(self.__legend_len_list[fignum]):
                if 'color_gradient' in self.__legend_[fignum][int((line_num * 2) + 1)] and switch:
                    color_gradient[fignum].append(True)
                else:
                    color_gradient[fignum].append(False)
        return color_gradient

    def two_d_subplots(self):
        """Create 2D subplots for the data."""
        color_gradient = self.__color_gradient()
        color_array = self.__color_map_array(color_gradient)
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

        self.__set_axis_style()

    def color_gradient_two_d_subplots(self):
        """Create 2D subplots with color gradients."""
        color_gradient = self.__color_gradient(switch=True)
        color_array = self.__color_map_array(color_gradient)

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

    # def filling_two_d_subplots(self):
    #     """Create 2D subplots with filling."""
    #     color_gradient = self.__color_gradient()
    #     color_array = self.__color_map_array(color_gradient)
    #     for fig_num in range(self.__figure_number):
    #         legend_len = self.__legend_len_list[fig_num]
    #         scatter_index = self.__kind_index[fig_num]
    #         for line_num in range(legend_len):
    #             if int((line_num * 2) + 1) not in scatter_index:
    #                 self.__plot(fig_num, line_num, color_array)

    #             else:
    #                 self.__scatter(fig_num, line_num, color_array)

    #             if self.__figure_number > 1:
    #                 self.__figure_serial(fig_num, use_tex=self.plt.rcParams['text.usetex'])

    # def draw_igd(self):

    #     x_data = np.loadtxt(r'.\igd-ave.csv', delimiter=',', dtype=np.float64, encoding='utf-8', usecols=0)
    #     ave_data = np.loadtxt(r'.\igd-ave.csv', delimiter=',', dtype=np.float64, encoding='utf-8',usecols=(1, 2, 3, 4, 5, 6))
    #     sigma_data = np.loadtxt(r'.\igd-sigma.csv', delimiter=',', dtype=np.float64, encoding='utf-8',usecols=(1, 2, 3, 4, 5, 6))
    #     ave_all = [np.average(num) for num in ave_data]
    #     lis = ['MOEAD',  'MOEAD-DE',  'NSGA-II','NSGA-II-DE', 'NSGA-III', 'NSGA-III-DE']
    #     lis_abc = ['a','b','c','d','e','f']
    #     fig, axs = plt.subplots(3, 2, figsize=(18, 18))
    #     for num in range(6):
    #         upper_bound = ave_data[:, num] + sigma_data[:, num]
    #         lower_bound = ave_data[:, num] - sigma_data[:, num]
    #         axs[int(num//2), int(num % 2)].plot(x_data, ave_data[:, num], lw=2, label=lis[num])
    #         axs[int(num//2), int(num % 2)].plot(x_data, ave_all, lw=1.5, label='Mean', color='C0', ls='--',dashes=(6,4))
    #         axs[int(num//2), int(num % 2)].fill_between(x_data, lower_bound, upper_bound, facecolor='C0', alpha=0.4, label=r'$\rm\mp\sigma$ range')
    #         axs[int(num//2), int(num % 2)].legend(loc='upper right', frameon=False)
    #         # axs[int(num//2), int(num % 2)].fill_between(x_data, upper_bound, ave_all, where=ave_all > upper_bound, fc='red', alpha=0.4)
    #         # axs[int(num//2), int(num % 2)].fill_between(x_data, lower_bound, ave_all, where=ave_all < lower_bound, fc='red', alpha=0.4)
    #         axs[int(num//2), int(num % 2)].set_xlabel('Generation')

    #         axs[int(num//2), int(num % 2)].set_ylabel('Indicator generational distance')
    #         axs[int(num//2), int(num % 2)].annotate('({})'.format(lis_abc[num]),xy=(0.08, 0.93), xycoords='axes fraction',xytext=(0,0), textcoords='offset pixels',
    #         horizontalalignment='right',verticalalignment='top',size=18)
    #         axs[int(num//2), int(num % 2)].set_xlim(0,100)



    def pie_chart(self):
        pass

    def histogram(self):
        pass

    def box_plot(self):
        pass

    def violin_plot(self):
        pass


    
    