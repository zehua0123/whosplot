
from whosplot.run import Run


class MyStyle(Run):
    def __init__(self):
        super(MyStyle, self).__init__()
        # self.plt.rcParams['text.usetex'] = False

    def my_figure(self):
        self.two_d_subplots()
        self.set_axis_off()
        self.save_fig(fig_format="pdf")
        self.show("pdf")


MyStyle = MyStyle()
MyStyle.my_figure()
