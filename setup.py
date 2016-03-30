from setuptools import setup
from setuptools import find_packages

setup(
    name='wiggly_bar',
    version='0.0.1',
    packages=find_packages(),
    url='',
    license='',
    author='Scott Paine',
    author_email='scott.paine1@gmail.com',
    description='Using VisPy, a GUI to examine the 2016 RIT Mechanics Midterm',
    entry_points={
        'console_scripts':
            [
                'wiggly_bar = wiggly_bar:main'
            ]
    }
)