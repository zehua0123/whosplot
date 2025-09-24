
from whosplot.run import Run
from whosplot.utility import create_empty_data_csv


class MyStyle(Run):
    def __init__(self):
        super(MyStyle, self).__init__()


    def my_figure(self):
        img_list = [f"time{i}.bmp" for i in range(5)]
        label_name = [r'Particle volume fraction $\alpha_p$ (-)']
        label_type = 2
        data_range = [[0, 0.5]]
        self.pv_screenshot_arrange(img_path=img_list,
                                   label_type=label_type,
                                   data_range=data_range,
                                   label_name=label_name)
        
        text1 = []
        for i in range(1, 6):
            text1.append('Time = ' + str(i * 5) + ' s')
        xypos1 = (0.5, 1.05)
        self.text(text1, 
                  xypos1,
                  horizontalalignment='center',
                  verticalalignment='top',
                  fontsize=20)

        self.save_fig(fig_format='pdf')
        self.show(fig_format='pdf')


create_empty_data_csv('./ContourCombinationOneColorbar.csv', 5)
MyStyle = MyStyle()
MyStyle.my_figure()