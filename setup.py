import io
import os
import sys

from setuptools import setup,find_packages

# Package meta-data.
NAME = "ioplin"
VERSION="0.0.1"
DESCRIPTION = "IOPLIN model implementation. Keras and TensorFlow Keras."
URL = "https://github.com/HuangSheng-CQU"
EMAIL = "20171607@cqu.edu.cn"
AUTHOR = "wenhao tang"
REQUIRES_PYTHON = "<=3.7.9"


# The rest you shouldn't have to touch too much :)
# ------------------------------------------------
# Except, perhaps the License and Trove Classifiers!
# If you do change the License, remember to change the Trove Classifier for that!

here = os.path.abspath(os.path.dirname(__file__))

# What packages are required for this module to be executed?
try:
    with open(os.path.join(here, "requirements.txt"), encoding="utf-8") as f:
        REQUIRED = f.read().split("\n")
except:
    REQUIRED = []

# Where the magic happens:
setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=DESCRIPTION,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=find_packages(exclude=("miniset")),
    # If your package is a single module, use this instead of 'packages':
    #py_modules=['mypackage'],
    # entry_points={
    #     'console_scripts': ['mycli=mymodule:cli'],
    # },
    install_requires=REQUIRED,
    #extras_require=EXTRAS,
    include_package_data=True,
    license="Apache License 2.0",
 )