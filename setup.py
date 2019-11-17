from setuptools import setup
import setuptools

setup(
    name='MakiZoo',
    packages=setuptools.find_packages(),
    version='0.0.1',
    description='A zoo of models written using MakiFlow framework.',
    long_description='...',
    author='Kilbas Igor, Gribanov Danil',
    author_email='igor.kilbas@mail.ru',
    url='https://github.com/oKatanaaa/MakiFlow',
    include_package_data=True,  # This will include all files in MANIFEST.in in the package when installing.
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ], install_requires=[]
)