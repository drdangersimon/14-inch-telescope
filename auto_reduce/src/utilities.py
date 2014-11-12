import pyfits as fits
import numpy as np
import os 

def fitype(fit, ftype=None):
	'''Returns True if the fit is of type ftype (Dark or Bias)
	param fit: is the fit file of intrest
	param ftye: is the type of fit(bais,flat). If ftype is None then
	 the method just checks if fit is a fits file, returns True is its a fits'''
 
	if not (fit.endswith('.fit') or fit.endswith('.fits')):
		return False
	if ftype != None:
		if fits.getval(fit,'IMAGETYP') == ftype+' Frame':
			return False

	return True


def shape(filepath, hdu = 0, verbose=True):
	"""
	Returns the 2D shape (width, height) of a FITS image.
	
	:param hdu: The hdu of the fits file that you want me to use. 0 is primary. If multihdu, 1 is usually science.

	
	"""
	hdr = fits.getheader(filepath, hdu)
	if hdr["NAXIS"] != 2:
		raise RuntimeError("Hmm, this hdu is not a 2D image !")
	if verbose:
		print "Image shape of %s : (%i, %i)" % (os.path.basename(filepath), int(hdr["NAXIS1"]), int(hdr["NAXIS2"]))
	return (int(hdr["NAXIS1"]), int(hdr["NAXIS2"]))
	

def fromfits(infilename, hdu = 0, verbose = True):
	"""
	Reads a FITS file and returns a 2D numpy array of the data.
	Use hdu to specify which HDU you want (default = primary = 0)
	"""
	
	if verbose:
		print "Reading %s ..." % (os.path.basename(infilename))
	
	pixelarray, hdr = fits.getdata(infilename, hdu, header=True)
	pixelarray = np.asarray(pixelarray).transpose()
	
	pixelarrayshape = pixelarray.shape
	if verbose :
		print "FITS import (%i, %i) BITPIX %s / %s" % (pixelarrayshape[0], pixelarrayshape[1], hdr["BITPIX"], str(pixelarray.dtype.name))
		
	return pixelarray, hdr

def tofits(outfilename, pixelarray, hdr = None, verbose = True):
	"""
	Takes a 2D numpy array and write it into a FITS file.
	If you specify a header (pyfits format, as returned by fromfits()) it will be used for the image.
	You can give me boolean numpy arrays, I will convert them into 8 bit integers.
	"""
	pixelarrayshape = pixelarray.shape
	if verbose :
		print "FITS export (%i, %i) %s ..." % (pixelarrayshape[0], pixelarrayshape[1], str(pixelarray.dtype.name))

	if pixelarray.dtype.name == "bool":
		pixelarray = np.cast["uint8"](pixelarray)

	if os.path.isfile(outfilename):
		os.remove(outfilename)
	
	if hdr == None: # then a minimal header will be created 
		hdu = fits.PrimaryHDU(pixelarray.transpose())
	else: # this if else is probably not needed but anyway ...
		hdu = fits.PrimaryHDU(pixelarray.transpose(), hdr)

	hdu.writeto(outfilename)
	
	if verbose :
		print "Wrote %s" % outfilename

def get_fits_type(indir, keyword):
    '''(list or str, str) -> list of str

    Removes files without the *.fits or *.fit extentions.
    '''
    #take out non fits files
    i = 0
    while i < len(indir):
        if not (indir[i].endswith('.fit') or 
                indir[i].endswith('.fits')):
            indir.pop(i)
        else:
            #remove non light images
            if fits.getval(indir[i],
                           'IMAGETYP').lower().startswith(keyword):
                i += 1
            else:
                indir.pop(i)
    return indir
