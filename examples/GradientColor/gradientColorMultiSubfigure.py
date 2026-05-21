
from __future__ import annotations

import os
import pathlib
import sys

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from whosplot.run import Run

EXAMPLE_DIR = pathlib.Path(__file__).resolve().parent
os.chdir(EXAMPLE_DIR)

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

