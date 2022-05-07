from setuptools import setup, find_packages
import re
from os import path

if __name__ == '__main__':
    setup(
        name="699", 
        packages=find_packages(exclude=['docs', 'examples']),
    )
