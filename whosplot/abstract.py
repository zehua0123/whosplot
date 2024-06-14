
from whosplot.utility import *
from whosplot.__init__ import (
    __version__,
    __author__,
    __author_email__,
    __license__,
    __copyright__,
    __cake__
)

"""
This code defines a class called Abstract. Here are some comments on the different parts of the code:

The import statements at the beginning of the code are used to import modules from other Python packages.

Abstract is a class that inherits from the object class, which is the base class for all Python objects.

The __init__ method initializes a config_path attribute that stores the absolute path to a configuration file named 
config.ini.

The __getattr__ method is a special method that gets called when an attribute of the class is accessed. It returns 
the value of the attribute.

The __setattr__ method is a special method that gets called when an attribute of the class is set. It sets the value 
of the attribute.

The @classmethod decorator is used to define class methods that can be called on the class itself, rather than on an 
instance of the class.

The __version__, __author__, __author_email__, __license__, __copyright__, and __cake__ methods are class methods 
that print the corresponding information to the console when called.

Overall, this code defines a basic class with some helpful class methods for printing information about the program. 
However, it is difficult to assess the usefulness of this code without knowing more about the context in which it is 
used.

"""
class Abstract(object):

    def __init__(self):
        self.config_path = os.path.abspath('config.ini')

    def __getattr__(self, item):
        """
        Get the attribute of the class.
        :param item:
        :return:
        """
        return self.__dict__[item]

    def __setattr__(self, key, value):
        """
        Set the attribute of the class.
        :param key:
        :param value:
        :return:
        """
        self.__dict__[key] = value

    @classmethod
    def __version__(cls):
        """
        Print the version of the program.
        :return:
        """
        print(__version__)

    @classmethod
    def __author__(cls):
        """
        Print the author of the program.
        :return:
        """
        print(__author__)

    @classmethod
    def __author_email__(cls):
        """
        Print the author email of the program.
        :return:
        """
        print(__author_email__)

    @classmethod
    def __license__(cls):
        """
        Print the license of the program.
        :return:
        """
        print(__license__)

    @classmethod
    def __copyright__(cls):
        """
        Print the copyright of the program.
        :return:
        """
        print(__copyright__)

    @classmethod
    def __cake__(cls):
        """
        Print the cake of the program.
        :return:
        """
        print(__cake__)
