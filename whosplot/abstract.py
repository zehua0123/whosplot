
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
    """
    A class to represent abstract information about the program.
    
    Attributes:
    ----------
    config_path : str
        The absolute path to the configuration file.

    Methods:
    -------
    __getattr__(self, item):
        Gets the attribute of the class.
    __setattr__(self, key, value):
        Sets the attribute of the class.
    version(cls):
        Prints the version of the program.
    author(cls):
        Prints the author of the program.
    author_email(cls):
        Prints the author's email.
    license(cls):
        Prints the license of the program.
    copyright(cls):
        Prints the copyright information.
    cake(cls):
        Prints the cake information.
    """
    
    def __init__(self):
        self.config_path = os.path.abspath('config.ini')

    def __getattr__(self, item):
        """
        Get the attribute of the class.
        :param item: Attribute name
        :return: Attribute value
        """
        try:
            return self.__dict__[item]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{item}'")

    def __setattr__(self, key, value):
        """
        Set the attribute of the class.
        :param key: Attribute name
        :param value: Attribute value
        """
        super().__setattr__(key, value)

    @classmethod
    def version(cls):
        """Print the version of the program."""
        print(__version__)

    @classmethod
    def author(cls):
        """Print the author of the program."""
        print(__author__)

    @classmethod
    def author_email(cls):
        """Print the author's email."""
        print(__author_email__)

    @classmethod
    def license(cls):
        """Print the license of the program."""
        print(__license__)

    @classmethod
    def copyright(cls):
        """Print the copyright information."""
        print(__copyright__)

    @classmethod
    def cake(cls):
        """Print the cake information."""
        print(__cake__)
