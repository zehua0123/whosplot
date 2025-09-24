
from whosplot.run import Run


class MyStyle(Run):
    def __init__(self):
        super(MyStyle, self).__init__()

    def my_figure(self):
        text = []
        for i in range(4):
            n = 0.9 + 0.1 * i
            text.append(r'Reaction order = {:.1f}'.format(n))
        self.two_d_subplots()
        xypos = (0.05, 0.8)
        self.text(text, xypos, horizontalalignment='left', fontsize=16)
        self.save_fig(fig_format='pdf')
        self.show(fig_format='pdf')


MyStyle = MyStyle()
MyStyle.my_figure()