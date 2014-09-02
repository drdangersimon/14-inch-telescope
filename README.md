14-inch-telescope
=================

Data reduction for Tony Ferrel Teach Telescope. The plan is to make an automatic pipeline that takes raw data and does photometry or spectrocipy and 
puts the results into a database for later analiysis. It will run as a daemon and as soon as new data is inputted it will start processing.

Steps in processing:
1. Organize new files into calibration and science images
2. Calibrate images and give a flag if calibration isn't to specifications or failed
3. Get astrometry for science images
4. Find sources and extract flux and put a postage stamp into a database
5. If spectra, finish spectral calibration (wavelentgh, flux, remove skylines, etc...)

So far the python depenceies are:
Astrometry.net
SeXtractor
Scipy, numpy, astropy, pyfits 


