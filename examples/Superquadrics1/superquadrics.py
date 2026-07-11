from whosplot.run import Run
from whosplot.utility import create_empty_data_csv


class MyStyle(Run):
    def __init__(self):
        super(MyStyle, self).__init__()

    def my_figure(self):
        items = [
            {
                "shape": "superellipsoid",
                "a1": 1.0,
                "a2": 1.0,
                "a3": 1.0,
                "e1": 1.0,
                "e2": 1.0,
                "nu": 48,
                "nv": 48,
                "view": (25, 40),
            },
            {
                "shape": "superellipsoid",
                "a1": 1.0,
                "a2": 1.0,
                "a3": 1.0,
                "e1": 0.001,
                "e2": 0.001,
                "nu": 100,
                "nv": 100,
                "view": (25, 40),
            },
            {
                "shape": "superellipsoid",
                "a1": 0.5,
                "a2": 0.5,
                "a3": 1.0,
                "e1": 0.001,
                "e2": 1.0,
                "nu": 100,
                "nv": 100,
                "view": (25, 40),
            },
            {
                "shape": "superellipsoid",
                "a1": 1.0,
                "a2": 1.0,
                "a3": 1.0,
                "e1": 3.0,
                "e2": 1.0,
                "nu": 48,
                "nv": 48,
                "view": (25, 40),
            },
            {
                "shape": "supertoroid",
                "a_major": 1.3,
                "a_minor": 0.25,
                "a3": 0.4,
                "e1": 1.0,
                "e2": 0.9,
                "nu": 48,
                "nv": 48,
                "view": (25, 40),
            },
            {
                "shape": "supertoroid",
                "a_major": 1.0,
                "a_minor": 0.6,
                "a3": 0.5,
                "e1": 0.4,
                "e2": 0.6,
                "nu": 48,
                "nv": 48,
                "view": (25, 40),
            },
            {
                "shape": "hyperboloid_one_sheet",
                "a1": 0.65,
                "a2": 0.65,
                "a3": 1.0,
                "e1": 1.0,
                "e2": 1.0,
                "u_extent": 2,
                "nu": 48,
                "nv": 48,
                "view": (25, 40),
            },
            {
                "shape": "hyperboloid_two_sheets",
                "a1": 0.7,
                "a2": 0.7,
                "a3": 0.9,
                "e1": 0.9,
                "e2": 1.2,
                "u_min": 0.12,
                "u_max": 1.2,
                "nu": 48,
                "nv": 48,
                "view": (25, 40),
            },
            {
                "shape": "superparaboloid",
                "a1": 1.0,
                "a2": 1.0,
                "a3": 1.0,
                "e1": 1.0,
                "e2": 1.2,
                "nu": 48,
                "nv": 48,
                "view": (25, 40),
            },
        ]

        text = [
            "Superellipsoid\n$\\varepsilon_1=\\varepsilon_2=1$\n$a_1=a_2=a_3=1$",
            "Superellipsoid\n$\\varepsilon_1=\\varepsilon_2=0.001$\n$a_1=a_2=a_3=1$",
            "Superellipsoid\n$\\varepsilon_1=0.001,\\ \\varepsilon_2=1$\n$a_1=a_2=0.5,\\ a_3=1$",
            "Superellipsoid\n$\\varepsilon_1=3,\\ \\varepsilon_2=1$\n$a_1=a_2=a_3=1$",
            "Supertoroid\n$\\varepsilon_1=1.0,\\ \\varepsilon_2=0.9$\n$a_{major}=1.3,\\ a_{minor}=0.25,\\ a_3=0.4$",
            "Supertoroid\n$\\varepsilon_1=0.4,\\ \\varepsilon_2=0.6$\n$a_{major}=1.0,\\ a_{minor}=0.6,\\ a_3=0.5$",
            "Hyperboloid of one sheet\n$\\varepsilon_1=1.0,\\ \\varepsilon_2=1.0$\n$a_1=a_2=0.65,\\ a_3=1$",
            "Hyperboloid of two sheets\n$\\varepsilon_1=0.9,\\ \\varepsilon_2=1.2$\n$a_1=a_2=0.7,\\ a_3=0.9$",
            "Superparaboloid\n$\\varepsilon_1=1.0,\\ \\varepsilon_2=1.2$\n$a_1=a_2=a_3=1.0$",
        ]
        xypos = (0.5, 0.05)
        self.draw_superquadrics(items)
        self.text(
            text,
            xypos,
            horizontalalignment="center",
            verticalalignment="top",
            fontsize=11,
        )

        self.save_fig(fig_format="pdf")
        self.show(fig_format="pdf")


create_empty_data_csv("./superquadrics.csv", 9)
MyStyle = MyStyle()
MyStyle.my_figure()
