from setuptools import setup, find_packages
import codecs
import os

here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

VERSION = '1.0.0'
DESCRIPTION = 'Python Reactor Modeling Tools (PyREMOT)'
LONG_DESCRIPTION = 'PyREMOT consists of some numerical models of packed-bed reactors which can be used for parameters estimation/simulation/optimization cases.'

# Setting up
setup(
    name="PyREMOT",
    version=VERSION,
    author="Sina Gilassi",
    author_email="<sina.gilassi@gmail.com>",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    packages=find_packages(),
    license='MIT',
    url='https://pyremot.herokuapp.com/',
    install_requires=['opencv-python', 'numpy',
                      'scipy', 'matplotlib'],
    keywords=['python', 'chemical engineering', 'packed-bed reactor',
              'homogenous reactor', 'reaction engineering'],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
