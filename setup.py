from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='nfl',
    version = "0.0.4",
    description = "NFL model to predict injuries in american football",
    license = "MIT",
    authors = "Axel Gabay, Lama Itani, Patricia Stojanovic",
    author_email = "lama.itani@hotmail.com",
    #url="https://github.com/lewagon/taxi-fare",
    install_requires = requirements,
    packages = find_packages(),
    test_suite = "tests",
    # include_package_data: to install data from MANIFEST.in
    include_package_data = True,
    zip_safe=False)
