from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.

setup(
    name='model_builder',
    version='0.1',
    description='Build pytorch model from json',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/unixnme/model_builder.git',
    author='Unix&Me',
    author_email='unixnme@gmail.com',
    keywords='pytorch model builder json',  # Optional

    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    packages=['model_builder'],  # Required
    python_requires='>=3.5',
    install_requires=['torch'],
)