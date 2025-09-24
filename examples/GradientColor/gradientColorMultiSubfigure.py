
from whosplot.run import Run

class MyStyle(Run):
    def __init__(self):
        super(MyStyle, self).__init__()

    def my_figure(self):
        self.color_gradient_two_d_subplots()
        text = [
            r'\TeX ing with \textbf{Matplotlib}\;\textbf{::\;viridis}',
            r'\TeX ing with \textbf{Matplotlib}\;\textbf{::\;plasma}',
            r'\TeX ing with \textbf{Matplotlib}\;\textbf{::\;inferno}',
            r'\TeX ing with \textbf{Matplotlib}\;\textbf{::\;magma}',
            r'\TeX ing with \textbf{Matplotlib}\;\textbf{::\;cividis}',
            r'\TeX ing with \textbf{Matplotlib}\;\textbf{::\;twilight}',
        ]
        xypos = (0.5, 0.9)
        self.text(text, xypos)
        # self.set_axis_off()
        self.save_fig(fig_format='svg')
        self.show(fig_format='pdf')


MyStyle = MyStyle()
MyStyle.my_figure()

