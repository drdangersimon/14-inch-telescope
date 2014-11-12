from setuptools import setup, find_packages
setup(
    name = "auto_reduce",
    version = "0.0",
    packages = find_packages(),
    install_requires = ['numpy>=1.6', 'scipy>=0.12', 'pyfits>3.0','matplotlib>=1.3','asciidata>=1','astropy>=0.3'],
    # metadata for upload to PyPI
    author = "Thuso Simon",
    author_email = "dr.danger.simon@gmail.com",
    description = "Automatic reduction, source extraction and database storage using python",
    license = "GPLv2",
    keywords = "CCD, astropy",
    url = "https://github.com/drdangersimon/14-inch-telescope.git",
    classifiers=[
        'License :: OSI Approved :: GPLv2 License',
        'Programming Language :: Python :: 2 :: Only',
        'Topic :: Scientific/Engineering :: Astronomy spectral fitting']
)
