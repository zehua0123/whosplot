
from whosplot.run import Run


class MyStyle(Run):
    def __init__(self):
        super(MyStyle, self).__init__()
        # self.plt.rcParams['text.usetex'] = False

    def my_figure(self):
        self.two_d_subplots()
        self.save_fig(fig_format='svg')
        self.show(fig_format='pdf')


MyStyle = MyStyle()
MyStyle.my_figure()
