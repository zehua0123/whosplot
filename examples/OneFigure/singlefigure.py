
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
        # self.plt.rcParams['text.usetex'] = False

    def my_figure(self):
        self.two_d_subplots()
        self.set_axis_off()
        self.save_fig(fig_format="pdf")
        self.show("pdf")


MyStyle = MyStyle()
MyStyle.my_figure()
