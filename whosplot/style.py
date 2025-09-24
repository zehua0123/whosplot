
import numpy as np
from whosplot.abstract import Abstract
from whosplot.load import CsvReader
from whosplot.parameter import Parameter
import matplotlib.pyplot as plt


class Style(Abstract):
    def __init__(self):
        """
        Initialize the Style class.
        Sets up the parameters and styles for plotting.
        """
        super().__init__()
        self.plt = plt
        self.__Parameter = Parameter()
        self.__CsvReader = CsvReader()
        self.__set_attributes()
        self.__set_content_style()

    def __set_attributes(self):
        """
        Set various attributes for the figure.
        Calls a series of methods to set up the rows, columns, figure number, label lengths, 
        x and y label types, tick label formats, title, limited labels, legend properties, 
        language, and other styles.
        """
        parameter_attributes = [
            'rows', 'cols', 'width', 'height', 'x_label_type', 'y_label_type', 
            'x_ticklabel_format', 'y_ticklabel_format', 'title', 'limited_label', 
            'legend_location', 'line_with_marker', 'multi_line_style', 
            'figure_kind', 'color_map'
        ]
        
        csvreader_attributes = [
            'figure_number', 'label_len_list', 'legend_len_list', 'legend', 
            'language', 'label', 'kind_index'
        ]

        for attr in parameter_attributes:
            setattr(self, f"_{self.__class__.__name__}__{attr}", getattr(self.__Parameter, attr))

        for attr in csvreader_attributes:
            setattr(self, f"_{self.__class__.__name__}__{attr}", getattr(self.__CsvReader, attr))

    def __create_figure(self):
        """
        Create the figure and subplots.
        Sets up the figure with multiple subplots based on the specified rows and columns.
        If any subplot requires a secondary y-axis, it sets that up as well.
        """
        self.__set_rcParams()
        self.axs_dict = {}
        self.fig, self.axs = plt.subplots(self.__rows, self.__cols,
                                          figsize=(self.__width * self.__cols, self.__height * self.__rows),
                                          sharex='none',
                                          sharey='none',
                                          squeeze=False)
        for num in range(self.__figure_number):
            if self.__label_len_list[num] == 3:
                self.ax0 = self.axs[int(num // self.__cols), int(num % self.__cols)].twinx()

        self.__REF_WIDTH_IN = 8.0  
        self.__REF_AXES_LW = 2.0   
        self.__REF_LABELSIZE = 16    
        self.__REF_LABELPAD = 8    
        self.__REF_TICKLABEL = 14   
        self.__REF_TICK_MAJ_LEN = 5   
        self.__REF_TICK_MIN_LEN = 3.5  
        self.__REF_TICK_MAJ_W = 2    
        self.__REF_TICK_MIN_W = 1.5  
        self.__REF_TICK_PAD = 6     
        self.__REF_LINE_MS = 8.0   
        self.__REF_MARKEREDGEW = 1.0  
        self.__REF_SCATTER_S = 64.0 

        self.scale_per_fig = [max(self.__width / self.__REF_WIDTH_IN, 0.1)] * self.__figure_number

        self.frame_linewidth = [self.__REF_AXES_LW * s for s in self.scale_per_fig]
        self.line_width = [[lw for _ in range(100)] for lw in self.frame_linewidth] 

        self.markersize = [self.__REF_LINE_MS * s for s in self.scale_per_fig]
        self.markeredgewidth = [self.__REF_MARKEREDGEW * s for s in self.scale_per_fig]

        self.scatter_size = [self.__REF_SCATTER_S * (s ** 2) for s in self.scale_per_fig]

        self.__apply_axis_scaling()

    def __apply_axis_scaling(self):
        for num in range(self.__figure_number):
            ax = self.axs[int(num // self.__cols), int(num % self.__cols)]
            s = self.scale_per_fig[num]

            for spine in ax.spines.values():
                spine.set_linewidth(self.__REF_AXES_LW * s)

            ax.xaxis.label.set_size(self.__REF_LABELSIZE * s)
            ax.yaxis.label.set_size(self.__REF_LABELSIZE * s)
            ax.xaxis.labelpad = self.__REF_LABELPAD * s
            ax.yaxis.labelpad = self.__REF_LABELPAD * s

            ax.tick_params(axis='both', which='major',
                        labelsize=self.__REF_TICKLABEL * s,
                        length=self.__REF_TICK_MAJ_LEN * s,
                        width=self.__REF_TICK_MAJ_W * s,
                        pad=self.__REF_TICK_PAD * s)
            ax.tick_params(axis='both', which='minor',
                        labelsize=self.__REF_TICKLABEL * s,
                        length=self.__REF_TICK_MIN_LEN * s,
                        width=self.__REF_TICK_MIN_W * s,
                        pad=self.__REF_TICK_PAD * s)
            
    '''
    ============================== global configuration inherited from matplotlibrc =================================
    '''

    def __set_axis(self):
        """
        Set the axis properties using matplotlib rcParams.
        Configures the appearance and behavior of the axes.
        """
        self.plt.rcParams['axes.facecolor'] = 'white'  # axes background color
        self.plt.rcParams['axes.edgecolor'] = 'black'  # axes edge color
        self.plt.rcParams['axes.linewidth'] = 2  # edge line width
        self.plt.rcParams['axes.grid'] = False  # display grid or not
        self.plt.rcParams['axes.grid.which'] = 'major'  # grid lines at {major, minor, both} ticks
        self.plt.rcParams['axes.titlelocation'] = 'center'  # alignment of the title: {left, right, center}
        self.plt.rcParams['axes.titlesize'] = 'large'  # font size of the axes title
        self.plt.rcParams['axes.titleweight'] = 'normal'  # font weight of title
        self.plt.rcParams['axes.titlecolor'] = 'auto'
        # color of the axes title, auto falls back to text.color as default value
        self.plt.rcParams['axes.titley'] = None  # position title (axes relative units).  None implies auto
        self.plt.rcParams['axes.titlepad'] = 6.0  # pad between axes and title in points
        self.plt.rcParams['axes.labelsize'] = 16  # font size of the x and y labels
        self.plt.rcParams['axes.labelpad'] = 8  # space between label and axis
        self.plt.rcParams['axes.labelweight'] = 'normal'  # weight of the x and y labels
        self.plt.rcParams['axes.labelcolor'] = 'black'
        self.plt.rcParams['axes.axisbelow'] = 'line'
        # draw axis gridlines and ticks: below patches (True); above patches but below lines ('line'); above all (False)
        self.plt.rcParams['axes.formatter.limits'] = [-10, 10]
        # use scientific notation if log10 of the axis range is smaller than the first or larger than the second
        self.plt.rcParams['axes.formatter.use_locale'] = False
        # When True, format tick labels according to the user's locale.
        # For example, use ',' as a decimal separator in the fr_FR locale.
        self.plt.rcParams['axes.formatter.use_mathtext'] = False  # When True, use mathtext for scientific notation.
        self.plt.rcParams['axes.formatter.min_exponent'] = 0  # minimum exponent to format in scientific notation
        self.plt.rcParams['axes.formatter.useoffset'] = True
        # If True, the tick label formatter will default to labeling ticks relative to an offset
        # when the data range is small compared to the minimum absolute value of the data.
        self.plt.rcParams['axes.formatter.offset_threshold'] = 4
        # When useoffset is True, the offset will be used
        # when it can remove at least this number of significant digits from tick labels.
        self.plt.rcParams['axes.spines.left'] = True  # display axis spines
        self.plt.rcParams['axes.spines.bottom'] = True
        self.plt.rcParams['axes.spines.top'] = True
        self.plt.rcParams['axes.spines.right'] = True
        self.plt.rcParams['axes.unicode_minus'] = False
        # use Unicode for the minus symbol rather than hyphen.
        # See https://en.wikipedia.org/wiki/Plus_and_minus_signs#Character_codes
        # self.plt.rcParams['axes.prop_cycle'] = cycler (
        # 'color', ['1f77b4', 'ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf'])
        # color cycle for plot lines as list of string color specs: single letter,
        # long name, or web-style hex As opposed to all other parameters in this file,
        # the color values must be enclosed in quotes for this parameter, e.g. '1f77b4', instead of 1f77b4.
        # See also https://matplotlib.org/tutorials/intermediate/color_cycle.html for more details on prop_cycle usage.
        self.plt.rcParams['axes.xmargin'] = .0  # x margin.  See `axes.Axes.margins`
        self.plt.rcParams['axes.ymargin'] = .05  # y margin.  See `axes.Axes.margins`
        self.plt.rcParams['axes.zmargin'] = .05  # z margin.  See `axes.Axes.margins`
        # self.plt.rcParams['axes.autolimit_mode'] = data
        # If "data", use axes.xmargin and axes.ymargin as is. If "round_numbers",
        # after application of margins, axis limits are further expanded to the nearest "round" number.
        self.plt.rcParams['polaraxes.grid'] = True  # display grid on polar axes
        self.plt.rcParams['axes3d.grid'] = True  # display grid on 3D axes
        self.plt.rcParams['xaxis.labellocation'] = 'center'  # alignment of the xaxis label: {left, right, center}
        self.plt.rcParams['yaxis.labellocation'] = 'center'  # alignment of the yaxis label: {bottom, top, center}

    def __set_tick(self):
        """
        Set the properties for the ticks using matplotlib rcParams.
        Configures the appearance and behavior of the x and y ticks.
        """
        self.plt.rcParams['xtick.top'] = False  # draw ticks on the top side
        self.plt.rcParams['xtick.bottom'] = True  # draw ticks on the bottom side
        self.plt.rcParams['xtick.labeltop'] = False  # draw label on the top
        self.plt.rcParams['xtick.labelbottom'] = True  # draw label on the bottom
        self.plt.rcParams['xtick.major.size'] = 5  # major tick size in points
        self.plt.rcParams['xtick.minor.size'] = 3.5  # minor tick size in points
        self.plt.rcParams['xtick.major.width'] = 2  # minor tick size in points
        self.plt.rcParams['xtick.minor.width'] = 1.5  # minor tick width in points
        self.plt.rcParams['xtick.major.pad'] = 6  # distance to major tick label in points
        self.plt.rcParams['xtick.minor.pad'] = 6.1  # distance to the minor tick label in points
        self.plt.rcParams['xtick.color'] = 'black'  # color of the ticks
        self.plt.rcParams['xtick.labelcolor'] = 'inherit'  # color of the tick labels or inherit from xtick.color
        self.plt.rcParams['xtick.labelsize'] = 14  # font size of the tick labels
        self.plt.rcParams['xtick.direction'] = 'in'  # direction: {in, out, inout}
        self.plt.rcParams['xtick.minor.visible'] = False  # visibility of minor ticks on x-axis
        self.plt.rcParams['xtick.major.top'] = True  # draw x axis top major ticks
        self.plt.rcParams['xtick.major.bottom'] = True  # draw x axis bottom major ticks
        self.plt.rcParams['xtick.minor.top'] = True  # draw x axis top minor ticks
        self.plt.rcParams['xtick.minor.bottom'] = True  # draw x axis bottom minor ticks
        self.plt.rcParams['xtick.alignment'] = 'center'  # alignment of xticks

        self.plt.rcParams['ytick.left'] = True  # draw ticks on the left side
        self.plt.rcParams['ytick.right'] = False  # draw ticks on the right side
        self.plt.rcParams['ytick.labelleft'] = True  # draw label on the left side
        self.plt.rcParams['ytick.labelright'] = False  # draw label on the right side
        self.plt.rcParams['ytick.major.size'] = 5  # major tick size in points
        self.plt.rcParams['ytick.minor.size'] = 3.5  # minor tick size in points
        self.plt.rcParams['ytick.major.width'] = 2  # minor tick size in points
        self.plt.rcParams['ytick.minor.width'] = 1.5  # minor tick width in points
        self.plt.rcParams['ytick.major.pad'] = 6  # distance to major tick label in points
        self.plt.rcParams['ytick.minor.pad'] = 6.1  # distance to the minor tick label in points
        self.plt.rcParams['ytick.color'] = 'black'  # color of the ticks
        self.plt.rcParams['ytick.labelcolor'] = 'inherit'  # color of the tick labels or inherit from ytick.color
        self.plt.rcParams['ytick.labelsize'] = 14  # font size of the tick labels
        self.plt.rcParams['ytick.direction'] = 'in'  # direction: {in, out, inout}
        self.plt.rcParams['ytick.minor.visible'] = False  # visibility of minor ticks on y-axis
        self.plt.rcParams['ytick.major.left'] = True  # draw x axis left major ticks
        self.plt.rcParams['ytick.major.right'] = True  # draw x axis right major ticks
        self.plt.rcParams['ytick.minor.left'] = True  # draw x axis left minor ticks
        self.plt.rcParams['ytick.minor.right'] = True  # draw x axis right minor ticks
        self.plt.rcParams['ytick.alignment'] = 'center'  # alignment of yticks

    def __set_figure(self):
        """
        Set the properties for the figure using matplotlib rcParams.
        Configures the overall appearance and layout of the figure.
        """
        self.plt.rcParams['figure.titlesize'] = 'large'  # size of the figure title (``Figure.suptitle()``)
        self.plt.rcParams['figure.titleweight'] = 'normal'  # weight of the figure title
        self.plt.rcParams['figure.figsize'] = [8., 6.]  # figure size in inches
        self.plt.rcParams['figure.dpi'] = 600  # figure dots per inch
        self.plt.rcParams['figure.facecolor'] = 'white'  # figure face color
        self.plt.rcParams['figure.edgecolor'] = 'white'  # figure edge color
        self.plt.rcParams['figure.frameon'] = True  # enable figure frame
        self.plt.rcParams['figure.max_open_warning'] = 20
        # The maximum number of figures to open through the pyplot interface before emitting a warning.
        # If less than one this feature is disabled.
        self.plt.rcParams['figure.raise_window'] = False  # Raise the GUI window to front when show() is called.

        '''The figure subplot parameters.  All dimensions are a fraction of the figure width and height.'''
        self.plt.rcParams['figure.subplot.left'] = 0.125  # the left side of the subplots of the figure
        self.plt.rcParams['figure.subplot.right'] = 0.9  # the right side of the subplots of the figure
        self.plt.rcParams['figure.subplot.bottom'] = 0.11  # the bottom of the subplots of the figure
        self.plt.rcParams['figure.subplot.top'] = 0.88  # the top of the subplots of the figure
        self.plt.rcParams['figure.subplot.wspace'] = 0.2
        # the amount of width reserved for space between subplots, expressed as a fraction of the average axis width
        self.plt.rcParams['figure.subplot.hspace'] = 0.2
        # the amount of height reserved for space between subplots, expressed as a fraction of the average axis height

        '''Figure layout'''
        self.plt.rcParams['figure.autolayout'] = False
        # When True, automatically adjust subplot parameters to make the plot fit the figure using `tight_layout`
        self.plt.rcParams['figure.constrained_layout.use'] = True
        # When True, automatically make plot elements fit on the figure. (Not compatible with `autolayout`, above).
        self.plt.rcParams[
            'figure.constrained_layout.h_pad'] = 0.04167  # Padding around axes objects. Float representing
        self.plt.rcParams['figure.constrained_layout.w_pad'] = 0.04167  # inches. Default is 3/72 inches (3 points)
        # self.plt.rcParams['figure.constrained_layout.w_pad'] = 0.6  # inches. Default is 3/72 inches (3 points)
        self.plt.rcParams['figure.constrained_layout.hspace'] = 0.02  # Space between subplot groups. Float representing
        self.plt.rcParams[
            'figure.constrained_layout.wspace'] = 0.02  # a fraction of the subplot widths being separated.

    def __set_font(self):
        """
        Set the font properties using matplotlib rcParams.
        Configures the appearance of text elements in the figure.
        """
        self.plt.rcParams['font.family'] = ['serif', 'monospace', 'cursive', 'sans-serif', 'SimSun']
        self.plt.rcParams['font.style'] = 'normal'
        self.plt.rcParams['font.variant'] = 'normal'
        self.plt.rcParams['font.weight'] = 'normal'
        self.plt.rcParams['font.stretch'] = 'normal'
        self.plt.rcParams['font.size'] = 14
        self.plt.rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Computer Modern Roman',
                                           'Bitstream Vera Serif', 'New Century Schoolbook', 'Century Schoolbook L',
                                           'Utopia', 'ITC Bookman', 'Bookman', 'Nimbus Roman No9 L', 'Times',
                                           'Palatino',
                                           'Charter', 'serif']
        self.plt.rcParams['font.sans-serif'] = ['Helvetica', 'DejaVu Sans', 'Bitstream Vera Sans',
                                                'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva',
                                                'Lucid',
                                                'Arial', 'Avant Garde', 'sans-serif']
        self.plt.rcParams['font.cursive'] = ['Apple Chancery', 'Textile', 'Zapf Chancery', 'Sand', 'Script MT',
                                             'Felipa', 'Comic Neue', 'Comic Sans MS', 'cursive']
        self.plt.rcParams['font.fantasy'] = ['Chicago', 'Charcoal', 'Impact', 'Western', 'Humor Sans', 'xkcd',
                                             'fantasy']
        self.plt.rcParams['font.monospace'] = ['DejaVu Sans Mono', 'Bitstream Vera Sans Mono',
                                               'Computer Modern Typewriter', 'Andale Mono', 'Nimbus Mono L',
                                               'Courier New', 'Courier', 'Fixed', 'Terminal', 'monospace']

    def __set_latex(self):
        """
        Set the properties for LaTeX rendering using matplotlib rcParams.
        Configures the use of LaTeX for text handling and math text.
        """
        self.plt.rcParams['text.usetex'] = True
        # use latex for all text handling. The following fonts are supported through the usual rc parameter settings:
        # new century schoolbook, bookman, times, palatino, zapf chancery, charter, serif, sans-serif, helvetica,
        # avant garde, courier, monospace, computer modern roman, computer modern sans serif,
        # computer modern typewriter.
        
        # self.plt.rcParams['text.latex.preamble'] = r'\usepackage{newtxtext, newtxmath}'
        self.plt.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

        # Preamble to pass to latex when rendering text with latex. The string should contain valid LaTeX code.
        # text.latex.preamble is a single line of LaTeX code that will be passed on to the LaTeX system. It may contain
        # any code that is valid for the LaTeX "preamble", i.e. between the "\documentclass" and "\begin{document}"
        # statements.
        # Note that it has to be put on a single line, which may become quite long.
        # The following packages are always loaded with usetex, so beware of package collisions:
        # geometry, inputenc, type1cm.
        # PostScript (PSNFSS) font packages may also be loaded, depending on your font settings.

        """The following settings allow you to select the fonts in math mode."""
        self.plt.rcParams['mathtext.fontset'] = 'cm'  # The font set to use for math text.
        # Should be 'dejavusans' (default), 'dejavuserif', 'cm' (Computer Modern), 'stix', 'stixsans'
        # or 'custom' (unsupported, may go away in the future)
        self.plt.rcParams['mathtext.bf'] = 'sans:bold'
        # The bold version of the font. Should be a valid key in the font.family dictionary.
        self.plt.rcParams['mathtext.it'] = 'sans:italic'
        # The italic version of the font. Should be a valid key in the font.family dictionary.
        self.plt.rcParams['mathtext.rm'] = 'sans'
        # The roman (regular) version of the font. Should be a valid key in the font.family dictionary.
        self.plt.rcParams['mathtext.sf'] = 'sans'
        # The sans-serif font used for symbols. Should be a valid key in the font.family dictionary.
        self.plt.rcParams['mathtext.tt'] = 'monospace'
        # The monospace font used for text. Should be a valid key in the font.family dictionary.
        self.plt.rcParams['mathtext.cal'] = 'cursive'
        # The calligraphic font used for text. Should be a valid key in the font.family dictionary.

    def __set_label_type(self):
        """
        Set the label type.
        :return: None
        """
        def format_func(val, pos, format_str):
            return f"{val:{format_str}}" if val != 0 else "0"
        
        if self.__x_label_type is None or self.__x_label_type == 'linear':
            x_label_type = [None] * self.__figure_number
        else:
            x_label_type = self.__x_label_type

        if self.__y_label_type is None or self.__y_label_type == 'linear':
            y_label_type = [None] * self.__figure_number
        else:
            y_label_type = self.__y_label_type

        for num in range(self.__figure_number):
            row = int(num // self.__cols)
            col = int(num % self.__cols)
            ax = self.axs[row, col]

            if x_label_type[num] is not None and x_label_type[num] != 'linear':
                # Set x-axis to logarithmic scale with base 10
                ax.set_xscale(x_label_type[num], base=10)
                ax.xaxis.set_major_locator(plt.LogLocator(base=10))
                # Format x-axis tick labels
                if self.__x_ticklabel_format[num] != 'plain':
                    ax.xaxis.set_major_formatter(
                        plt.FuncFormatter(lambda val, pos: format_func(val, pos, self.__x_ticklabel_format[num]))
                    )
            if y_label_type[num] is not None and y_label_type[num] != 'linear':
                # Set y-axis to logarithmic scale with base 10
                ax.set_yscale(y_label_type[num], base=10)
                ax.yaxis.set_major_locator(plt.LogLocator(base=10))
                # Format y-axis tick labels
                if self.__y_ticklabel_format[num] != 'plain':
                    ax.yaxis.set_major_formatter(
                        plt.FuncFormatter(lambda val, pos: format_func(val, pos, self.__y_ticklabel_format[num]))
                    )

    def __set_ticklabel_format(self):
        """
        Set the scale type for x and y labels for each subplot.
        """
        for num in range(self.__figure_number):
            if self.__x_ticklabel_format is None or self.__x_ticklabel_format[num] == 'plain':
                x_ticklabel_format = None
            else:
                x_ticklabel_format = self.__x_ticklabel_format[num]

            if self.__y_ticklabel_format is None or self.__y_ticklabel_format[num] == 'plain':
                y_ticklabel_format = None
            else:
                y_ticklabel_format = self.__y_ticklabel_format[num]

            if x_ticklabel_format is not None:
                self.axs[int(num // self.__cols), int(num % self.__cols)].ticklabel_format(
                    axis='x',
                    style=x_ticklabel_format,
                    scilimits=(0, 0),
                    useMathText=True)
            if y_ticklabel_format is not None:
                self.axs[int(num // self.__cols), int(num % self.__cols)].ticklabel_format(
                    axis='y',
                    style=y_ticklabel_format,
                    scilimits=(0, 0),
                    useMathText=True)

    def __set_title(self):
        """
        Set the title for each subplot.
        """
        if self.__title:
            for num in range(self.__figure_number):
                self.axs[
                    int(num // self.__cols),
                    int(num % self.__cols)].set_title(self.__title, fontdict={'size': 26, 'weight': 'bold'})

    def __set_label_limitation(self):
        """
        Set the limits for the x and y labels for each subplot.
        """
        if self.__limited_label is None:
            return 0

        for fignum in range(self.__figure_number):
            if len(self.__limited_label[fignum]) == 4 or len(self.__limited_label[fignum]) == 6:
                if self.__limited_label[fignum][0] != 0 or self.__limited_label[fignum][1] != 0:
                    self.axs[int(fignum // self.__cols), int(fignum % self.__cols)].set_xlim(
                        self.__limited_label[fignum][0],
                        self.__limited_label[fignum][1])

                if self.__limited_label[fignum][2] != 0 or self.__limited_label[fignum][3] != 0:
                    self.axs[int(fignum // self.__cols), int(fignum % self.__cols)].set_ylim(
                        self.__limited_label[fignum][2],
                        self.__limited_label[fignum][3])

            if len(self.__limited_label[fignum]) == 6:
                if self.__limited_label[fignum][4] != 0 and self.__limited_label[fignum][5] != 0:
                    self.axs_dict[fignum].set_ylim(
                        self.__limited_label[fignum][4],
                        self.__limited_label[fignum][5])

                    # self.axs2.set_ylim(
                    #     self.limited_label[fignum][4],
                    #     self.limited_label[fignum][5])

            if len(self.__limited_label[fignum]) != 4 and len(self.__limited_label[fignum]) != 6:
                raise ValueError('The limited label size is wrong!')

    def __set_legend(self, **kwargs):
        """
        Set the legend for each subplot.
        Handles additional legend items if provided via kwargs.
        """
        for num in range(self.__figure_number):
            # Handle custom legends if provided
            if kwargs.get('extra_legend') is not None:
                arg_list = kwargs['extra_legend'][num]
                legend_new = self.__legend[num] + \
                    [item for sublist in [arg_list[i:i+2] for i in range(0, len(arg_list), 4)] for item in sublist]
                legend_items = legend_new[1::2]
            else:
                # Default handling when no extra_legend is provided
                if self.__legend_len_list[num] == 1:
                    return 0  # Assuming the function should terminate early in this case
                legend_items = self.__legend[num][1::2]

            # Configure the legend based on the label length
            if self.__label_len_list[num] != 3:
                labels = self.axs[int(num // self.__cols), int(num % self.__cols)].legend(
                    legend_items,
                    loc=self.__legend_location[num],
                    frameon=False,
                    fontsize=14 * self.scale_per_fig[num],
                    handlelength=2.5).get_texts()
                if self.__language == 'chinese':
                    for label in labels:
                        label.set_fontname('SimSun')
            else:
                raise ValueError('Side ylabel is not supported now!')

    def __set_label(self):
        """
        Set the labels for the x and y axes for each subplot.
        """
        for num in range(self.__figure_number):
            if self.__label_len_list[num] == 2 or self.__label_len_list[num] == 3:
                self.axs[int(num // self.__cols), int(num % self.__cols)].set_xlabel(
                    self.__label[num][0],
                    fontdict={'weight': 'bold'})
                self.axs[int(num // self.__cols), int(num % self.__cols)].set_ylabel(
                    self.__label[num][1],
                    fontdict={'weight': 'bold'})

            elif self.__label_len_list[num] == 3:
                self.ax0.set_ylabel(
                    self.__label[num][2],
                    fontdict={'weight': 'bold'},
                    rotation=270,
                    labelpad=24)

                # self.axs2.set_ylabel(
                #     self.label[num][2],
                #     fontdict={'weight': 'bold'},
                #     rotation=270,
                #     labelpad=24)
            else:
                raise ValueError('The label size is wrong!')

    def __set_marker(self):
        """
        Set the markers for each line in the subplots.
        Configures the marker style based on the type of figure.
        """
        self.marker = []
        line_with_marker = self.__line_with_marker
        multi_line_style = self.__multi_line_style
        filled_markers = ['+', 'x', '*', 'h', 'H', 'o', 'v', '.', ',', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p',
                          'D', 'd', '|', '_', 'P', 'X', 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
        for num in range(self.__figure_number):
            if self.__figure_kind[num] == 'scatter' and line_with_marker[num] is True:
                self.marker.append(filled_markers)
            elif self.__figure_kind[num] == 'scatter' and type(line_with_marker[num]) == list:
                self.marker.append(line_with_marker[num])
            elif self.__figure_kind[num] == 'plot' and type(line_with_marker[num]) == list:
                self.marker.append(line_with_marker[num])
            elif self.__figure_kind[num] == 'plot' and line_with_marker[num] is True and multi_line_style[num] is False:
                self.marker.append(filled_markers)
            elif self.__figure_kind[num] == 'mix':
                marker_index = self.__kind_index[num]
                marker = [None] * (len(filled_markers) - len(marker_index))
                for i in range(len(marker_index)):
                    marker.insert(marker_index[i] // 2, filled_markers[i])
                self.marker.append(marker)
            else:
                self.marker.append([None] * len(filled_markers))

    def __set_line_style(self):
        """
        Set the line styles for each line in the subplots.
        Configures the line style based on the type of figure.
        linestyle_str = [
            ('solid', 'solid'),  # Same as (0, ()) or '-'
            ('dotted', 'dotted'),  # Same as (0, (1, 1)) or ':'
            ('dashed', 'dashed'),  # Same as '--'
            ('dashdot', 'dashdot')]  # Same as '-.'

        linestyle_tuple = [
            ('loosely dotted', (0, (1, 10))),
            ('dotted', (0, (1, 1))),
            ('densely dotted', (0, (1, 1))),
            ('long dash with offset', (5, (10, 3))),
            ('loosely dashed', (0, (5, 10))),
            ('dashed', (0, (5, 5))),
            ('densely dashed', (0, (5, 1))),

            ('loosely dashdotted', (0, (3, 10, 1, 10))),
            ('dashdotted', (0, (3, 5, 1, 5))),
            ('densely dashdotted', (0, (3, 1, 1, 1))),

            ('dashdotdotted', (0, (3, 5, 1, 5, 1, 5))),
            ('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
            ('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))]
        """
        self.line_style = []
        multi_line_style = self.__multi_line_style
        linestyle_dict = {
            'loosely dotted': (0, (1, 10)),
            'dotted': (0, (1, 1)),
            'densely dotted': (0, (1, 1)),
            'long dash with offset': (5, (10, 3)),
            'loosely dashed': (0, (5, 10)),
            'dashed': (0, (5, 5)),
            'densely dashed': (0, (5, 1)),

            'loosely dashdotted': (0, (3, 10, 1, 10)),
            'dashdotted': (0, (3, 5, 1, 5)),
            'densely dashdotted': (0, (3, 1, 1, 1)),

            'dashdotdotted': (0, (3, 5, 1, 5, 1, 5)),
            'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
            'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))
        }   

        line_style = ['dotted', 'dashed', 'dashdot', 'solid']
        for num in range(self.__figure_number):
            if multi_line_style[num] is True:
                self.line_style.append(line_style * 5)
            elif type(multi_line_style[num]) == list:
                multi_line_style_ = []
                for i in range(len(multi_line_style[num])):
                    if multi_line_style[num][i] not in linestyle_dict:
                        multi_line_style_.append(multi_line_style[num][i])
                    else:
                        multi_line_style_.append(linestyle_dict[multi_line_style[num][i]])
                self.line_style.append(multi_line_style_)
            else:
                self.line_style.append(['-'] * 100)

    def __set_color_map(self):
        """
            Perceptually Uniform Sequential:    viridis,plasma,inferno,magma,cividis \n\r

            Sequential:         Greys,Purples,Blues,Greens,Oranges,Reds,YlOrBr,\r
                                YlOrRd,OrRd,PuRd,RdPu,BuPu,GnBu,PuBu,YlGnBu,PuBuGn,BuGn,YlGn \n\r

            Sequential (2):     binary,gist_yarg,gist_gray,gray,bone,pink,\r
                                spring,summer,autumn,winter,cool,Wistia,hot,afmhot,gist_heat,copper \n\r

            Diverging:          PiYG,PRGn,BrBG,PuOr,RdGy,RdBu,RdYlBu,RdYlGn,Spectral,coolwarm,bwr,seismic \n\r

            Cyclic:             twilight,twilight_shifted,hsv \n\r

            Qualitative:        Pastel1,Pastel2,Paired,Accent,Dark2,Set1,Set2,Set3,tab10,tab20,tab20b,tab20c \n\r

            Miscellaneous:      flag,prism,ocean,gist_earth,terrain,gist_stern,gnuplot,gnuplot2,CMRmap, \r
                                cubehelix,brg,gist_rainbow,rainbow,jet,turbo,nipy_spectral,gist_ncar \n\r

            Matlab:             hot,cool,spring,summer,autumn,winter \n\r
        """
        self.color_map_array = []
        for num in range(self.__figure_number):
            color_map_array_ = plt.get_cmap(self.__color_map[num]).colors
            self.color_map_array.append(np.array(color_map_array_))

    def set_axis_style(self, **kwargs):
        """
        Set the style for the axes.
        Configures the label type, tick label format, title, label limitations, legend, and labels.
        """
        self.__set_label_type()
        self.__set_ticklabel_format()
        self.__set_title()
        self.__set_label_limitation()
        self.__set_legend(**kwargs)
        self.__set_label()

    def __set_content_style(self):
        """
        Set the content style for the figure.
        Configures the figure creation, markers, line styles, line widths, and color map.
        """
        self.__create_figure()
        self.__set_marker()
        self.__set_line_style()
        self.__set_color_map()

    def __set_rcParams(self):
        """
        Set the rcParams for matplotlib.
        Configures the general appearance and behavior of the figure.
        """
        self.__set_axis()
        self.__set_tick()
        self.__set_figure()
        self.__set_font()
        self.__set_latex()
