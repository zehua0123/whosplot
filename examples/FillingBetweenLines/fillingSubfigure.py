
from __future__ import annotations

import os
import pathlib
import sys

if __package__ in {None, ""}:
    sys.path.append(str(pathlib.Path(__file__).resolve().parents[2]))

from whosplot.run import Run
from whosplot.load import CsvReader

EXAMPLE_DIR = pathlib.Path(__file__).resolve().parent
os.chdir(EXAMPLE_DIR)


class FillingDataLoading:
    def __init__(self):
        self.__CsvReader = CsvReader(key='fillingdata')

    def get_data(self) -> None:
        """
        Print the data from the csv file.
        :return: None
        """
        data = self.__CsvReader.__getattr__('data')
        return data

    def get_extra_legend(self) -> tuple:
        return self.__CsvReader.legend

class MyStyle(Run):
    def __init__(self):
        super(MyStyle, self).__init__()

    def my_figure(self, filling_data, extra_legend):
        self.two_d_subplots(filling=True, filling_data=filling_data, extra_legend=extra_legend)
        self.save_fig(fig_format="pdf")
        self.show("pdf")


FillingDataLoading = FillingDataLoading()
filling_data = FillingDataLoading.get_data()
extra_legend = FillingDataLoading.get_extra_legend()
MyStyle = MyStyle()
MyStyle.my_figure(filling_data, extra_legend)
