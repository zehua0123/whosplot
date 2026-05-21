
from __future__ import annotations

import os
import pathlib
import sys

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from whosplot.run import Run
from whosplot.utility import create_empty_data_csv

EXAMPLE_DIR = pathlib.Path(__file__).resolve().parent
os.chdir(EXAMPLE_DIR)


class MyStyle(Run):
    def __init__(self):
        super(MyStyle, self).__init__()


    def my_figure(self):
        img_list = [f"{i}.bmp" for i in range(4)]
        label_name = [r'Particle volume fraction $ \alpha_p$ (-)', r'Particle velocity $\mathbf{U}_p$ (m/s)']
        label_type = 3
        data_range = [[0, 0.4],[0, 0.3]]
        tick_num = [5, 7]
        self.pv_screenshot_arrange(img_path=img_list,
                                   label_type=label_type,
                                   data_range=data_range,
                                   label_name=label_name,
                                   tick_num=tick_num)

        self.save_fig(fig_format='pdf')
        self.show(fig_format='pdf')


create_empty_data_csv('./ContourCombinationMultipleColorbars.csv', 4)
MyStyle = MyStyle()
MyStyle.my_figure()
