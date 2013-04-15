#reduices images from 14 inch for use with science
#Thuso Simon 8 Mar 2013

import tkFileDialog as tk
import numpy as np
import pylab as lab
import pyfits 
#import set_cord 
from glob import glob
import os,csv,math,shutil,sys
import asciidata,operator,copy,itertools
import scipy.spatial as spatial
import scipy.ndimage as ndimage #can use for filters (high/low pass)

fits=pyfits
nu=np
sexcmd = 'sex'

###main program
def get_flats(path=None,combine_type='mean',outdir=None,
	 Filter='FILTER'):
    '''(str,str,str,str) -> dict(ndarry), dict(asciidata)

    Opens flat field directory, combines all *.fits or *.fit files in the 
    directory, sorts by Filter option and out puts fits file to outdir and 
    also out ndarray of combined fits files and fits header with modified 
    history.  Combine types can include: "mean","median","sum" and "sigmaclip."
 
    Flats must be calibrate before combining. Counts should also be
    simmilar or combining will give wrong.

    Known Issues:
    Median and sigmaclip combine give artifacts when use.'''

    comm = comb_Type(combine_type)
    #gui options
    if path is None:
        path = gui_getdir(title='Please Select Flats dir')
        if not path:
            raise ValueError('Must specify directory where files are.')
    if not path.endswith('/'):
        path += '/'
    if outdir is None:
        outdir = gui_getdir(title='Please Select Save Directory')
    #load path to flats
    fits_path = sorted(glob(path+'*'))
    fits_path = get_fits_type(fits_path,'flat')
    filters = {}
    #sort flats by filter
    for i in fits_path:
        filt = str(fits.getval(i,Filter))
        if not filt in filters.keys():
            filters[filt] = [i]
        else:
            filters[filt].append(i)

    out,hdr = {},{}
    for i in filters.keys():
        out[i],hdr[i] = comm(filters[i])
    #save as fits?
    if outdir is None:
        outdir = path
    else:
        if not outdir.endswith('/'):
            outdir += '/'
    for i in out.keys():
        basename = os.path.split(filters[i][0])[-1]
        basename = os.path.splitext(basename)[0]
        tofits(outdir+basename+'_%s.fits'%combine_type.lower(), out[i],
               hdr[i],verbose=False)
    return out,hdr

def get_darks(path=None, combine_type='mean', outdir=None,
              Filter=('SET-TEMP','EXPTIME')):
    '''(str,str,str,tuple(str) or str) -> dict(ndarry), dict(asciidata)

    Opens dark directory, combines all *.fits or *.fit files in the 
    directory, sorts by Filter options (can have multiple) and out 
    puts fits file to outdir and also out ndarray of combined fits 
    files and fits header with modified history.  Combine types can 
    include: "mean","median","sum" and "sigmaclip."
    Known Issues:
    Median and sigmaclip combine give artifacts when use.'''

    comm = comb_Type(combine_type)
    #gui select directory
    if path is None:
        path = gui_getdir(title='Please Select Dark Directory')
        if not path:
            raise ValueError('Must specify directory where files are.')
    if not path.endswith('/'):
        path += '/'
    if outdir is None:
        outdir = gui_getdir(title='Please Select Save Directory')
    #load paths to fits
    fits_path = sorted(glob(path+'*'))
    fits_path = get_fits_type(fits_path,'dark')
    #sort by time and temp
    filters = {}
    for i in fits_path:
	if type(Filter) is tuple:
            filt = ''
            for j in Filter:
                filt += str(fits.getval(i,j)) + '_'
        else:
            filt = fits.getval(i,Filter)
        if not filt in filters.keys():
            filters[filt] = [i]
        else:
            filters[filt].append(i)

    out,hdr = {},{}
    for i in filters.keys():
        out[i],hdr[i] = comm(filters[i])
    #save as fits?
    if outdir is None:
        outdir = path
    else:
        if not outdir.endswith('/'):
            outdir += '/'
    for i in out.keys():
        basename = os.path.split(filters[i][0])[-1]
        basename = os.path.splitext(basename)[0]
        tofits(outdir+basename+'_%s.fits'%combine_type.lower(), out[i],
               hdr[i],verbose=False)
    return out,hdr



def get_bias(path=None, combine_type='mean', outdir=None, Filter='SET-TEMP'):
    '''(str, str, str, str) -> dict(ndarray), dict(asciidata)

    Opens bias directory, combines all *.fits or *.fit files in
    the directory, sorts by Filter option and out puts fits file
    to outdir and also out ndarray of combined fits files and fits
    header with modified history.  Combine types can include: 
    "mean","median","sum" and "sigmaclip."
    Known Issues:
    Median and sigmaclip combine give artifacts when use.'''

    comm = comb_Type(combine_type)
    #gui load dir
    if path is None:
        path = gui_getdir(title='Please Select Bias Directory')
        if not path:
            raise ValueError('Must specify directory where files are.')
    if not path.endswith('/'):
        path += '/'
    if outdir is None:
        outdir = gui_getdir(title='Please Select Save Directory')
    #find path to files
    fits_path = sorted(glob(path+'*'))
    fits_path = get_fits_type(fits_path,'bias')
    #sort by time
    filters = {}
    for i in fits_path:
        filt = str(fits.getval(i,Filter))
        if not filt in filters.keys():
            filters[filt] = [i]
        else:
            filters[filt].append(i)

    out,hdr = {},{}
    for i in filters.keys():
        out[i],hdr[i] = comm(filters[i])
    #save as fits?
    if outdir is None:
        outdir = path
    else:
        if not outdir.endswith('/'):
            outdir += '/'
    for i in out.keys():
        basename = os.path.split(filters[i][0])[-1]
        basename = os.path.splitext(basename)[0]
        tofits(outdir+basename+'_%s.fits'%combine_type.lower(), out[i],
               hdr[i],verbose=False)
    return out,hdr

  
def stack_images(path=None, combine_type='mean', outdir=None):
    '''(str, str, str) -> dict(ndarray), dict(asciidata)

    Opens image directory, combines all *.fits or *.fit files 
    in the directory, sorts by filter in fits header and outputs
    fits file to outdir and also out ndarray of combined fits files
    and fits header with modified history.  Combine types can
    include: "mean","median","sum" and "sigmaclip." 

    Images must be calibrated first before this program is used.

    Known Issues:
    Median and sigmaclip combine give artifacts when use.'''

    comm = comb_Type(combine_type)
    #gui load dir
    if path is None:
        path = gui_getdir(title='Please Select Fits dir')
        if not path:
            raise ValueError('Must specify directory where files are.')
    if not path.endswith('/'):
        path += '/'
    if outdir is None:
        outdir = gui_getdir(title='Please Select Save Directory')
    #load path to files
    light_path = sorted(glob(path+'*'))
    light_path = get_fits_type(light_path,'light')
    #sort fits by filter
    filters = {}
    for i in light_path:
        filt = fits.getval(i,'FILTER')
        if not filt in filters.keys():
            filters[filt] = [i]
        else:
            filters[filt].append(i)

    out,hdr = {},{}
    for i in filters.keys():
        out[i],hdr[i] = comm(filters[i])
    #save as fits?
    if outdir is None:
        outdir = indir
    else:
        if not outdir.endswith('/'):
            outdir += '/'
    for i in out.keys():
        basename = os.path.split(filters[i][0])[-1]
        basename = os.path.splitext(basename)[0]
        tofits(outdir+basename+'_%s.fits'%combine_type.lower(), out[i],
               hdr[i],verbose=False)
       
    return out,hdr     

def align_fits(path=None, outdir=None):
    '''(str, str) -> NoneType

    Aligns all light images in a directory by matching patterns
    in the stars. If no input then uses GUI to select in and out directory.

    Must have Source Extractor installed and called from bash terminal as
    "sex" or "sextractor."

    Images must be calibrated first before this program is used.
    '''

    if path is None:
        path = gui_getdir(title='Please Select Fits dir')
        if not path:
            raise ValueError('Must specify directory where files are.')
    if not path.endswith('/'):
        path += '/'
    if outdir is None:
        outdir = gui_getdir(title='Please Select Save Directory')
    #load path to fits
    light_path = sorted(glob(path+'*'))
    light_path = get_fits_type(light_path,'light')
    #start allignment
    ident = ident_run(light_path[0],light_path)
    #see if any images succesfully got aligned
    Shape = shape(light_path[0],verbose=False)
    #save algned fits to outdir
    for id in ident:
        if id.ok:
            affineremap(id.ukn.filepath, id.trans, 
                        Shape,outdir=outdir)

####supporting programs
def Normalize(a):
    '''(ndarry) -> ndarry

    Normalizes an array so that all values are in [0,1]. 
    >>> a = numpy.array([-1.,0.,1.],dtype=float)
    >>> Normalize(a)
    array([ 0. ,  0.5,  1. ])
    '''
    a = nu.asarray(a)
    return (a - a.min())/(a.ptp())


def comb_Type(combine_type):
    '''(str) -> function

    Sets correct function for input str.
    '''
    comb = ['mean', 'sum', 'median','sigmaclip']
    assert combine_type.lower() in comb
    if combine_type.lower() == 'mean':
        comm = combine_mean
    elif combine_type.lower() == 'sum':
        comm = combine_sum
    elif combine_type.lower() == 'median':
        comm = combine_medium
    elif combine_type.lower()== 'sigmaclip':
        comm = combine_sigmaclip
    return comm

def gui_getdir(initdir=os.curdir,title='Please Select Dir'):
    '''(str, str) -> unicode

    Uses tk to make a GUI to get path to a directory.'''

    return tk.askdirectory(initialdir=initdir, title=title)

def get_fits_type(indir,keyword):
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
  

def combine_sigmaclip(file_list, num=70):
    '''(str, int) -> ndarray, asciidata

    Puts list of fits files into correct order for use with
    sigma clipping function. Adds to fits header histrory.
    '''
    #fast but uses lots of ram
    if len(file_list) < num:
        shape = (fits.getval(file_list[0],'NAXIS1'),
                 fits.getval(file_list[0],'NAXIS2'),
                 len(file_list))
        temp = nu.zeros(shape)
        for i,j in enumerate(file_list):
            temp[:,:,i],hdr = fromfits(j,verbose=False)
        out = Sigmaclip(temp,axis=2)
    else:
        #slow ram but takes longer
        shape = (fits.getval(file_list[0],'NAXIS1'),
                 fits.getval(file_list[0],'NAXIS2'))
        out = nu.zeros(shape)
        temp = nu.zeros((shape[0],len(file_list)))
        for i in xrange(shape[1]):
            for j,k in enumerate(file_list):
                temp[:,j] = fromfits(k,verbose=False)[0][:,i]
            out[:,i] = Sigmaclip(temp,axis=1)
        temp,hdr = fromfits(k,verbose=False)
    hdr.add_history('Sigmaclip combine')
    return out,hdr

def combine_mean(file_list):
    '''(str) -> ndarray, asciidata

    Puts list of fits files into correct order for use with
    mean function. Adds to fits header histrory.

    '''
    out,hdr = combine_sum(file_list)
    out = out / float(len(file_list))
    hdr.add_history('Then took mean')
    return out,hdr
                    

def combine_medium(file_list , num=70):
    '''(str, int) -> ndarray, asciidata

    Puts list of fits files into correct order for use with
    median function. Adds to fits header histrory.
    '''
    #does medium assumes all have same fits header
    #high ram but quick
    if len(file_list) < num:
        shape = (fits.getval(file_list[0],'NAXIS1'),
                 fits.getval(file_list[0],'NAXIS2'),
                 len(file_list))
        temp = nu.zeros(shape)
        for i,j in enumerate(file_list):
            temp[:,:,i],hdr = fromfits(j,verbose=False)
        out = nu.median(temp,2)
    else:
        #low ram but takes longer
        shape = (fits.getval(file_list[0],'NAXIS1'),
                 fits.getval(file_list[0],'NAXIS2'))
        out = nu.zeros(shape)
        temp = nu.zeros((shape[0],len(file_list)))
        for i in xrange(shape[1]):
            for j,k in enumerate(file_list):
                temp[:,j] = fromfits(k,verbose=False)[0][:,i]
            out[:,i] = nu.median(temp,1)
        temp,hdr = fromfits(k,verbose=False)
    hdr.add_history('Medium combine')
    return out,hdr

def combine_sum(file_list):
    '''(str) -> ndarray, acsciidata

    Puts list of fits files into correct order for use with
    sum function. Adds to fits header histrory.

    '''
    #sums images together
    out = False
    for i in file_list:
        if nu.any(out):
            try:
                out += fromfits(i,verbose=False)[0]
            except ValueError:
                print i
                raise
        else:
            out,hdr = fromfits(i,verbose=False)
    hdr.add_history('Summed from %i images'%len(file_list))
    return out,hdr

def combine_SDmask(file_list, num=70):
    '''(str, int) -> ndarray, acsciidata

    Puts list of fits files into correct order for use with
    SD masking function. Adds to fits header histrory.

    '''
    pass

def Sigmaclip(array,low=4.,high=4,axis=None):
    '''(ndarray, int, int, int) -> ndarray

    Iterative sigma-clipping of along the given axis.
	   
    The output array contains only those elements of the input array `c`
    that satisfy the conditions ::
	   
    mean(c) - std(c)*low < c < mean(c) + std(c)*high
	
    Parameters
    ----------
    a : array_like
    data array
    low : float
    lower bound factor of sigma clipping
    high : float
    upper bound factor of sigma clipping
    
    Returns
    -------
    c : array
    sigma clipped mean along axis
    '''
    c = np.asarray(array)
    if axis is None or c.ndim == 1:
        from scipy.stats import sigmaclip
        return nu.mean(sigmaclip(c)[0])
    #create masked array
    c_mask = nu.ma.masked_array(c,nu.isnan(c))
    delta = 1
    while delta:
           c_std = c_mask.std(axis=axis)
           c_mean = c_mask.mean(axis=axis)
           size = c_mask.mask.sum()
           critlower = c_mean - c_std*low
           critupper = c_mean + c_std*high
           indexer = [slice(None)] * c.ndim
           for i in range(c.shape[axis]):
               indexer[axis] = slice(i,i+1)
               c_mask[indexer].mask = nu.logical_and(
                   c_mask[indexer].squeeze() > critlower, 
                   c_mask[indexer].squeeze()<critupper) == False
           delta = size - c_mask.mask.sum()
    return c_mask.mean(axis).data


####not my code
###ident
class Identification:
	"""
	Represents the identification of a transform between two ImgCat objects.
	Regroups all the star catalogs, the transform, the quads, the candidate, etc.
	
	All instance attributes are listed below.
		
	:ivar ref: ImgCat object of the reference image
	:ivar ukn: ImgCat object of the unknown image
	:ivar ok: boolean, True if the idendification was successful.
	:ivar trans: The SimpleTransform object that represents the geometrical transform from ukn to ref.
	:ivar uknmatchstars: A list of Star objects of the catalog of the unknown image...
	:ivar refmatchstars: ... that correspond to these Star objects of the reference image.
	:ivar medfluxratio: Median flux ratio between the images: "ukn * medfluxratio = ref"
		A high value corresponds to a shallow unknown image.
		It gets computed by the method calcfluxratio, using the matched stars.
	:ivar stdfluxratio: Standard error on the flux ratios of the matched stars.

		
	
	"""

	def __init__(self, ref, ukn):
		"""
		
		:param ref: The reference image
		:type ref: ImgCat object
		:param ukn: The unknown image, whose transform will be adjusted to match the ref
		:type ukn: ImgCat object
		
		"""
		self.ref = ref
		self.ukn = ukn
		
		self.ok = False
		
		self.trans = None
		self.uknmatchstars = []
		self.refmatchstars = []
		self.cand = None
		
		self.medfluxratio = None # ukn * medfluxratio --> ref (high value means shallow image)
		self.stdfluxratio = None

		
	def findtrans(self, r = 5.0, verbose=True):
		"""
		Find the best trans given the quads, and tests if the match is sufficient	
		"""
		
		# Some robustness checks
		if len(self.ref.starlist) < 4:
			if verbose:
				print "Not enough stars in the reference catalog."
			return
		if len(self.ukn.starlist) < 4:
			if verbose:
				print "Not enough stars in the unknown catalog."
			return
 
 		
		# First question : how many stars should match ?
		if len(self.ukn.starlist) < 5: # Then we should simply try to get the smallest distance...
			minnident = 4
		else:
			minnident = max(4, min(8, len(self.ukn.starlist)/5.0)) # Perfectly arbitrary, let's see how it works
		
		# Hmm, arbitrary for now :
		minquaddist = 0.005
		
		# Let's start :
		if self.ref.quadlevel == 0:
			self.ref.makemorequads(verbose=verbose)
		if self.ukn.quadlevel == 0:
			self.ukn.makemorequads(verbose=verbose)
			
		while self.ok == False:
			
			# Find the best candidates
			cands = proposecands(self.ukn.quadlist, self.ref.quadlist, n=4, verbose=verbose)
			
			if len(cands) != 0 and cands[0]["dist"] < minquaddist:
				# If no quads are available, we directly try to make more ones.
				for cand in cands:
					# Check how many stars are identified...					
					nident = identify(self.ukn.starlist, self.ref.starlist, trans=cand["trans"], r=r, verbose=verbose, getstars=False)
					if nident >= minnident:
						self.trans = cand["trans"]
						self.cand = cand
						self.ok = True
						break # get out of the for
					
			if self.ok == False:
				# We add more quads...
				addedmorerefquads = self.ref.makemorequads(verbose=verbose)
				addedmoreuknquads = self.ukn.makemorequads(verbose=verbose)
				
				if addedmorerefquads == False and addedmoreuknquads == False:
					break # get out of the while, we failed.
	

		if self.ok: # we refine the transform
			# get matching stars :
			(self.uknmatchstars, self.refmatchstars) = identify(self.ukn.starlist, self.ref.starlist, trans=self.trans, r=r, verbose=False, getstars=True)
			# refit the transform on them :
			if verbose:
				print "Refitting transform (before/after) :"
				print self.trans
			newtrans = fitstars(self.uknmatchstars, self.refmatchstars)
			if newtrans != None:
				self.trans = newtrans
				if verbose:
					print self.trans
			# Generating final matched star lists :
			(self.uknmatchstars, self.refmatchstars) = identify(self.ukn.starlist, self.ref.starlist, trans=self.trans, r=r, verbose=verbose, getstars=True)

			if verbose:
				print "I'm done !"
	
		else:
			if verbose:
				print "Failed to find transform !"
			
	def calcfluxratio(self, verbose=True):
		"""
		Computes a very simple median flux ratio between the images.
		The purpose is simply to get a crude guess, for images with e.g. different exposure times.
		Given that we have these corresponding star lists in hand, this is trivial to do once findtrans was run.
		"""
		assert len(self.uknmatchstars) == len(self.refmatchstars)
		if len(self.refmatchstars) == 0:
			if verbose:
				print "No matching stars to compute flux ratio !"
			return
		
		reffluxes = listtoarray(self.refmatchstars, full=True)[:,2]
		uknfluxes = listtoarray(self.uknmatchstars, full=True)[:,2]
		fluxratios = reffluxes / uknfluxes
		
		self.medfluxratio = float(np.median(fluxratios))
		self.stdfluxratio = float(np.std(fluxratios))
		
		if verbose:
			print "Computed flux ratio from %i matches : median %.2f, std %.2f" % (len(reffluxes), self.medfluxratio, self.stdfluxratio)
		
	
	
	def showmatch(self, show=False, verbose=True):
		"""
		A plot of the transformed stars and the candidate quad
		"""
		if self.ok == False:
			return
		if verbose:
			print "Plotting match ..."
		import matplotlib.pyplot as plt
		#import matplotlib.patches
		#import matplotlib.collections
		
		plt.figure(figsize=(10, 10))
		
		# The ref in black
		a = listtoarray(self.ref.starlist, full=True)
		plt.scatter(a[:,0], a[:,1], s=2.0, color="black")
		a = listtoarray(self.refmatchstars, full=True)
		plt.scatter(a[:,0], a[:,1], s=10.0, color="black")
		
		# The ukn in red
		a = listtoarray(self.trans.applystarlist(self.ukn.starlist), full=True)
		plt.scatter(a[:,0], a[:,1], s=2.0, color="red")
		a =  listtoarray(self.trans.applystarlist(self.uknmatchstars), full=True)
		plt.scatter(a[:,0], a[:,1], s=6.0, color="red")
		
		# The quad
		
		polycorners =  listtoarray(self.cand["refquad"].stars)
		polycorners =  ccworder(polycorners)
		plt.fill(polycorners[:,0], polycorners[:,1], alpha=0.1, ec="none", color="red")

		plt.xlim(self.ref.xlim)
		plt.ylim(self.ref.ylim)
		plt.title("Match of %s" % (str(self.ukn.name)))
		plt.xlabel("ref x")
		plt.ylabel("ref y")
		ax = plt.gca()
		ax.set_aspect('equal', 'datalim')
	
		if show:
			plt.show()
		else:
			if not os.path.isdir("alipy_visu"):
				os.makedirs("alipy_visu")
			plt.savefig(os.path.join("alipy_visu", self.ukn.name + "_match.png"))




def ident_run(ref, ukns, hdu=0, visu=False, skipsaturated=False, r = 5.0, n=500, sexkeepcat=False, sexrerun=True, verbose=True):
	"""
	Top-level function to identify transorms between images.
	Returns a list of Identification objects that contain all the info to go further.
	
	:param ref: path to a FITS image file that will act as the "reference".
	:type ref: string
	
	:param ukns: list of paths to FITS files to be "aligned" on the reference. **ukn** stands for unknown.
	:type ref: list of strings
	
	:param hdu: The hdu of the fits files (same for all) that you want me to use. 0 is primary. If multihdu, 1 is usually science.
	
	:param visu: If yes, I'll draw some visualizations of the process (good to understand problems, if the identification fails).
	:type visu: boolean
	
	:param skipsaturated: Should I skip saturated stars ?
	:type skipsaturated: boolean
	
	:param r: Identification radius in pixels of the reference image (default 5.0 should be fine).
	:type r: float
	:param n: Number of brightest stars of each image to consider (default 500 should be fine).
	:type n: int
	
	:param sexkeepcat: Put this to True if you want me to keep the SExtractor catalogs (in a dir "alipy_cats").
	:type sexkeepcat: boolean
	:param sexrerun: Put this to False if you want me to check if I have some previously kept catalogs (with sexkeepcat=True),
		instead of running SExtractor again on the images.
	:type sexrerun: boolean
	
	.. todo:: Make this guy accept existing asciidata catalogs, instead of only FITS images.

	
	"""
	
	if verbose:
		print 10*"#", " Preparing reference ..."
	ref = ImgCat(ref, hdu=hdu)
	ref.makecat(rerun=sexrerun, keepcat=sexkeepcat, verbose=verbose)
	ref.makestarlist(skipsaturated=skipsaturated, n=n, verbose=verbose)
	if visu:
		ref.showstars(verbose=verbose)
	ref.makemorequads(verbose=verbose)
	
	identifications = []
	
	for ukn in ukns:
		
		if verbose:
			print 10*"#", "Processing %s" % (ukn)
		
		ukn = ImgCat(ukn, hdu=hdu)
		ukn.makecat(rerun=sexrerun, keepcat=sexkeepcat, verbose=verbose)
		ukn.makestarlist(skipsaturated=skipsaturated, n=n, verbose=verbose)
		if visu:
			ukn.showstars(verbose=verbose)

		idn = Identification(ref, ukn)
		idn.findtrans(verbose=verbose, r=r)
		idn.calcfluxratio(verbose=verbose)
		identifications.append(idn)
		
		if visu:
			ukn.showquads(verbose=verbose)
			idn.showmatch(verbose=verbose)
		
	if visu:
		ref.showquads(verbose=verbose)
		
	return identifications

###align

def affineremap(filepath, transform, shape, alifilepath=None, outdir = "alipy_out", makepng=False, hdu=0, verbose=True):
	"""
	Apply the simple affine transform to the image and saves the result as FITS, without using pyraf.
	
	:param filepath: FITS file to align
	:type filepath: string

	:param transform: as returned e.g. by alipy.ident()
	:type transform: SimpleTransform object
	
	:param shape: Output shape (width, height) 
	:type shape: tuple

	:param alifilepath: where to save the aligned image. If None, I will put it in the outdir directory.
	:type alifilepath: string
	
	:param makepng: If True I make a png of the aligned image as well.
	:type makepng: boolean

	:param hdu: The hdu of the fits file that you want me to use. 0 is primary. If multihdu, 1 is usually science.


	"""
	inv = transform.inverse()
	(matrix, offset) = inv.matrixform()
	#print matrix, offset
	
	data, hdr = fromfits(filepath, hdu = hdu, verbose = verbose)
	data = scipy.ndimage.interpolation.affine_transform(data, matrix, offset=offset, output_shape = shape)
	
	basename = os.path.splitext(os.path.basename(filepath))[0]
	
	if alifilepath == None:
		alifilepath = os.path.join(outdir, basename + "_affineremap.fits")
	else:	
		outdir = os.path.split(alifilepath)[0]
	if not os.path.isdir(outdir):
		os.makedirs(outdir)
	#add fits header about transform	
	hdr.add_history('Alighn.py: Did Affine remap to Alighn fits')
	tofits(alifilepath, data, hdr = hdr, verbose = verbose)
	
	
	if makepng:
		try:
			import f2n
		except ImportError:
			print "Couldn't import f2n -- install it !"
			return
		myimage = f2n.f2nimage(numpyarray=data, verbose=False)
		myimage.setzscale("auto", "auto")
		myimage.makepilimage("log", negative = False)
		myimage.writetitle(os.path.basename(alifilepath))
		if not os.path.isdir(outdir):
				os.makedirs(outdir)
		myimage.tonet(os.path.join(outdir, os.path.basename(alifilepath)+".png"))



def shape(filepath, hdu = 0, verbose=True):
	"""
	Returns the 2D shape (width, height) of a FITS image.
	
	:param hdu: The hdu of the fits file that you want me to use. 0 is primary. If multihdu, 1 is usually science.

	
	"""
	hdr = pyfits.getheader(filepath, hdu)
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
	
	pixelarray, hdr = pyfits.getdata(infilename, hdu, header=True)
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
		hdu = pyfits.PrimaryHDU(pixelarray.transpose())
	else: # this if else is probably not needed but anyway ...
		hdu = pyfits.PrimaryHDU(pixelarray.transpose(), hdr)

	hdu.writeto(outfilename)
	
	if verbose :
		print "Wrote %s" % outfilename

		

def irafalign(filepath, uknstarlist, refstarlist, shape, alifilepath=None, outdir = "alipy_out", makepng=False, hdu=0, verbose=True):
	"""
	Uses iraf geomap and gregister to align the image. Three steps :
	 * Write the matched source lists into an input file for geomap
	 * Compute a geomap transform from these stars. 
	 * Run gregister
	
	:param filepath: FITS file to be aligned
	:type filepath: string
	
	:param uknstarlist: A list of stars from the "unknown" image to be aligned, that matches to ...
	:type uknstarlist: list of Star objects
	:param refstarlist: ... the list of corresponding stars in the reference image.
	:type refstarlist: list of Star objects
	
	:param shape: Output shape (width, height) 
	:type shape: tuple

	:param alifilepath: where to save the aligned image. If None, I put it in the default directory.
	:type alifilepath: string
	
	:param makepng: If True I make a png of the aligned image as well.
	:type makepng: boolean

	:param hdu: The hdu of the fits file that you want me to use. 0 is primary. If multihdu, 1 is usually science.


	
	"""

	try:
		from pyraf import iraf
	except ImportError:
		print "Couldn't import pyraf !"
		return

	assert len(uknstarlist) == len(refstarlist)
	if len(uknstarlist) < 2:
		if verbose:
			print "Not enough stars for using geomap !"
		return
		
	basename = os.path.splitext(os.path.basename(filepath))[0]
	geomapinpath = basename + ".geomapin"
	geodatabasepath = basename + ".geodatabase"
	if os.path.isfile(geomapinpath):
		os.remove(geomapinpath)
	if os.path.isfile(geodatabasepath):
		os.remove(geodatabasepath)

	
	# Step 1, we write the geomap input.
	table = []	
	for (uknstar, refstar) in zip(uknstarlist, refstarlist):
		table.append([refstar.x, refstar.y, uknstar.x, uknstar.y])
	geomap = open(geomapinpath, "wb") # b needed for csv
	writer = csv.writer(geomap, delimiter="\t")
	writer.writerows(table)
	geomap.close()
	
	
	# Step 2, geomap
		
	iraf.unlearn(iraf.geomap)	
	iraf.geomap.fitgeom = "rscale"		# You can change this to : shift, xyscale, rotate, rscale
	iraf.geomap.function = "polynomial"	# Surface type
	iraf.geomap.maxiter = 3			# Maximum number of rejection iterations
	iraf.geomap.reject = 3.0		# Rejection limit in sigma units
	
	# other options you could specify :
	#(xxorder=                    2) Order of x fit in x
	#(xyorder=                    2) Order of x fit in y
	#(xxterms=                 half) X fit cross terms type
	#(yxorder=                    2) Order of y fit in x
	#(yyorder=                    2) Order of y fit in y
	#(yxterms=                 half) Y fit cross terms type
	#(calctyp=                 real) Computation type

	iraf.geomap.transfo = "broccoli"	# keep it
	iraf.geomap.interac = "no"		# keep it
	iraf.geomap.verbose = "yes"		# keep it
	#iraf.geomap.results = "bla.summary" # The optional results summary files
	
	geomapblabla = iraf.geomap(input=geomapinpath, database=geodatabasepath, xmin = 1, xmax = shape[0], ymin = 1, ymax = shape[1], Stdout=1)
	
	# We read this output ...
	for line in geomapblabla:
		if "X and Y scale:" in line:
			mapscale = line.split()[4:6]
		if "Xin and Yin fit rms:" in line:
			maprmss = line.split()[-2:]
		if "X and Y axis rotation:" in line:
			mapangles = line.split()[-4:-2]
		if "X and Y shift:" in line:
			mapshifts = line.split()[-4:-2]
	
	geomaprms = math.sqrt(float(maprmss[0])*float(maprmss[0]) + float(maprmss[1])*float(maprmss[1]))
	geomapangle = float(mapangles[0])# % 360.0
	geomapscale = 1.0/float(mapscale[0])
	
	if mapscale[0] != mapscale[1]:
		raise RuntimeError("Error reading geomap scale")
	if verbose:
		print "IRAF geomap : Rotation %+11.6f [deg], scale %8.6f, RMS %.3f [pixel]" % (geomapangle, geomapscale, geomaprms)
	
	# Step 3
	
	if alifilepath == None:
		alifilepath = os.path.join(outdir, basename + "_gregister.fits")
	else:	
		outdir = os.path.split(alifilepath)[0]
	if not os.path.isdir(outdir):
		os.makedirs(outdir)
	if os.path.isfile(alifilepath):
		os.remove(alifilepath)
	

	iraf.unlearn(iraf.gregister)
	iraf.gregister.geometry = "geometric"	# linear, distortion, geometric
	iraf.gregister.interpo = "spline3"	# linear, spline3
	iraf.gregister.boundary = "constant"	# padding with zero
	iraf.gregister.constant = 0.0
	iraf.gregister.fluxconserve = "yes"
	
	if verbose:
		print "IRAF gregister ..."

	regblabla = iraf.gregister(input = '%s[%s]' % (filepath, hdu), output = alifilepath, database = geodatabasepath, transform = "broccoli", Stdout=1)

	if verbose:
		print "IRAF gregister done !"
	
	if os.path.isfile(geomapinpath):
		os.remove(geomapinpath)
	if os.path.isfile(geodatabasepath):
		os.remove(geodatabasepath)

	
	if makepng:
		try:
			import f2n
		except ImportError:
			print "Couldn't import f2n -- install it !"
			return
		myimage = f2n.fromfits(alifilepath, verbose=False)
		myimage.setzscale("auto", "auto")
		myimage.makepilimage("log", negative = False)
		myimage.writetitle(os.path.basename(alifilepath))
		if not os.path.isdir(outdir):
				os.makedirs(outdir)
		myimage.tonet(os.path.join(outdir, os.path.basename(alifilepath)+".png"))

###imcat
class ImgCat:
	"""
	Represent an individual image and its associated catalog, starlist, quads etc.
	"""

	def __init__(self, filepath, hdu=0, cat=None):
		"""
		
		:param filepath: Path to the FITS file, or alternatively just a string to identify the image.
		:type filepath: string
		
		:param cat: Catalog generated by SExtractor (if available -- if not, we'll make our own)
		:type cat: asciidata catalog
		
		:param hdu: The hdu containing the science data from which I should build the catalog. 0 is primary. If multihdu, 1 is usually science.
		
		"""
		self.filepath = filepath
		
		(imgdir, filename) = os.path.split(filepath)
		(common, ext) = os.path.splitext(filename)
		self.name = common 
		
		self.hdu = hdu
		self.cat = cat
		self.starlist = []
		self.mindist = 0.0
		self.xlim = (0.0, 0.0) # Will be set using the catalog -- no need for the FITS image.
		self.ylim = (0.0, 0.0)

		self.quadlist = []
		self.quadlevel = 0 # encodes what kind of quads have already been computed
		
	
	def __str__(self):
		return "%20s: approx %4i x %4i, %4i stars, %4i quads, quadlevel %i" % (os.path.basename(self.filepath),
			self.xlim[1] - self.xlim[0], self.ylim[1] - self.ylim[0],
			len(self.starlist), len(self.quadlist), self.quadlevel)
	
	def makecat(self, rerun=True, keepcat=False, verbose=True):
		self.cat = pysex_run(self.filepath, conf_args={'DETECT_THRESH':3.0, 'ANALYSIS_THRESH':3.0, 'DETECT_MINAREA':10,
		'PIXEL_SCALE':1.0, 'SEEING_FWHM':2.0, "FILTER":"Y"},
		params=['X_IMAGE', 'Y_IMAGE', 'FLUX_AUTO', 'FWHM_IMAGE', 'FLAGS', 'ELONGATION', 'NUMBER', "EXT_NUMBER"],
		rerun=rerun, keepcat=keepcat, catdir="alipy_cats")

	
	def makestarlist(self, skipsaturated=False, n=200, verbose=True):
		if self.cat:
			if skipsaturated:
				maxflag = 3
			else:
				maxflag = 7
			self.starlist = sortstarlistbyflux(readsexcat(self.cat, hdu=self.hdu, maxflag=maxflag, verbose=verbose))[:n]
			(xmin, xmax, ymin, ymax) = area(self.starlist, border=0.01)
			self.xlim = (xmin, xmax)
			self.ylim = (ymin, ymax)
			
			# Given this starlists, what is a good minimal distance for stars in quads ?
 			self.mindist = min(min(xmax - xmin, ymax - ymin) / 10.0, 30.0)
 				
		else:
			raise RuntimeError("No cat : call makecat first !")
	
	
	def makemorequads(self, verbose=True):
		"""
		We add more quads, following the quadlevel.
		"""
		#if not add:
		#	self.quadlist = []
		if verbose:
			print "Making more quads, from quadlevel %i ..." % self.quadlevel
		if self.quadlevel == 0:
			self.quadlist.extend(makequads1(self.starlist, n=7, d=self.mindist, verbose=verbose))
		elif self.quadlevel == 1:
			self.quadlist.extend(makequads2(self.starlist, f=3, n=5, d=self.mindist, verbose=verbose))
		elif self.quadlevel == 2:
			self.quadlist.extend(makequads2(self.starlist, f=6, n=5, d=self.mindist, verbose=verbose))
		elif self.quadlevel == 3:
			self.quadlist.extend(makequads2(self.starlist, f=12, n=5, d=self.mindist, verbose=verbose))
		elif self.quadlevel == 4:
			self.quadlist.extend(makequads2(self.starlist, f=10, n=6, s=3, d=self.mindist, verbose=verbose))

		else:
			return False
		
		self.quadlist = removeduplicates(self.quadlist, verbose=verbose)
		self.quadlevel += 1
		return True	
		
	
	def showstars(self, verbose=True):
		"""
		Uses f2n to write a png image with circled stars.
		"""
		try:
			import f2n
		except ImportError:
			print "Couldn't import f2n -- install it !"
			return
				
		if verbose:
			print "Writing png ..."
		myimage = f2n.fromfits(self.filepath, verbose=False)
		#myimage.rebin(int(myimage.xb/1000.0))
		myimage.setzscale("auto", "auto")
		myimage.makepilimage("log", negative = False)
		#myimage.upsample()
		myimage.drawstarlist(self.starlist, r=8, autocolour="flux")
		myimage.writetitle(os.path.basename(self.filepath))
		#myimage.writeinfo(["This is a demo", "of some possibilities", "of f2n.py"], colour=(255,100,0))
		if not os.path.isdir("alipy_visu"):
				os.makedirs("alipy_visu")
		myimage.tonet(os.path.join("alipy_visu", self.name + "_stars.png"))

	
	
	def showquads(self, show=False, flux=True, verbose=True):
		"""
		Uses matplotlib to write/show the quads.
		"""
		if verbose:
			print "Plotting quads ..."
		
		import matplotlib.pyplot as plt
		#import matplotlib.patches
		#import matplotlib.collections
		
		plt.figure(figsize=(10, 10))
		
		if len(self.starlist) >= 2:
			a =  listtoarray(self.starlist, full=True)
			if flux:
				f = np.log10(a[:,2])
				fmax = np.max(f)
				fmin = np.min(f)
				f = 1.0 + 8.0 * (f-fmin)/(fmax-fmin)
				plt.scatter(a[:,0], a[:,1], s=f, color="black")
			else:
				plt.plot(a[:,0], a[:,1], marker=",", ls="none", color="black")
		
		if len(self.quadlist) != 0:		
			for quad in self.quadlist:
				polycorners =  listtoarray(quad.stars)
				polycorners = ccworder(polycorners)
				plt.fill(polycorners[:,0], polycorners[:,1], alpha=0.03, ec="none")
	
		plt.xlim(self.xlim)
		plt.ylim(self.ylim)
		plt.title(str(self))
		plt.xlabel("x")
		plt.ylabel("y")
		ax = plt.gca()
		ax.set_aspect('equal', 'datalim')
	
		if show:
			plt.show()
		else:
			if not os.path.isdir("alipy_visu"):
				os.makedirs("alipy_visu")
			plt.savefig(os.path.join("alipy_visu", self.name + "_quads.png"))





def ccworder(a):
	"""
	Sorting a coordinate array CCW to plot polygons ...
	"""
	ac = a - np.mean(a, 0)
	indices = np.argsort(np.arctan2(ac[:, 1], ac[:, 0]))
	return a[indices]

###pysex
def _check_files(conf_file, conf_args, verbose=True):
	if conf_file is None:
		os.system("sex -d > .pysex.sex")
		conf_file = '.pysex.sex'
	if not conf_args.has_key('FILTER_NAME') or not os.path.isfile(conf_args['FILTER_NAME']):
		if verbose:
			print 'No filter file found, using default filter'
		f = open('.pysex.conv', 'w')
		print>>f, """CONV NORM
# 3x3 ``all-ground'' convolution mask with FWHM = 2 pixels.
1 2 1
2 4 2
1 2 1"""
		f.close()
		conf_args['FILTER_NAME'] = '.pysex.conv'
	if not conf_args.has_key('STARNNW_NAME') or not os.path.isfile(conf_args['STARNNW_NAME']):
		if verbose:
			print 'No NNW file found, using default NNW config'
		f = open('.pysex.nnw', 'w')
		print>>f, """NNW
# Neural Network Weights for the SExtractor star/galaxy classifier (V1.3)
# inputs:	 9 for profile parameters + 1 for seeing.
# outputs:	  ``Stellarity index'' (0.0 to 1.0)
# Seeing FWHM range: from 0.025 to 5.5'' (images must have 1.5 < FWHM < 5 pixels)
# Optimized for Moffat profiles with 2<= beta <= 4.

 3 10 10  1

-1.56604e+00 -2.48265e+00 -1.44564e+00 -1.24675e+00 -9.44913e-01 -5.22453e-01  4.61342e-02	8.31957e-01	 2.15505e+00  2.64769e-01
 3.03477e+00  2.69561e+00  3.16188e+00	3.34497e+00	 3.51885e+00  3.65570e+00  3.74856e+00	3.84541e+00	 4.22811e+00  3.27734e+00

-3.22480e-01 -2.12804e+00  6.50750e-01 -1.11242e+00 -1.40683e+00 -1.55944e+00 -1.84558e+00 -1.18946e-01	 5.52395e-01 -4.36564e-01 -5.30052e+00
 4.62594e-01 -3.29127e+00  1.10950e+00 -6.01857e-01	 1.29492e-01  1.42290e+00  2.90741e+00	2.44058e+00 -9.19118e-01  8.42851e-01 -4.69824e+00
-2.57424e+00  8.96469e-01  8.34775e-01	2.18845e+00	 2.46526e+00  8.60878e-02 -6.88080e-01 -1.33623e-02	 9.30403e-02  1.64942e+00 -1.01231e+00
 4.81041e+00  1.53747e+00 -1.12216e+00 -3.16008e+00 -1.67404e+00 -1.75767e+00 -1.29310e+00	5.59549e-01	 8.08468e-01 -1.01592e-02 -7.54052e+00
 1.01933e+01 -2.09484e+01 -1.07426e+00	9.87912e-01	 6.05210e-01 -6.04535e-02 -5.87826e-01 -7.94117e-01 -4.89190e-01 -8.12710e-02 -2.07067e+01
-5.31793e+00  7.94240e+00 -4.64165e+00 -4.37436e+00 -1.55417e+00  7.54368e-01  1.09608e+00	1.45967e+00	 1.62946e+00 -1.01301e+00  1.13514e-01
 2.20336e-01  1.70056e+00 -5.20105e-01 -4.28330e-01	 1.57258e-03 -3.36502e-01 -8.18568e-02 -7.16163e+00	 8.23195e+00 -1.71561e-02 -1.13749e+01
 3.75075e+00  7.25399e+00 -1.75325e+00 -2.68814e+00 -3.71128e+00 -4.62933e+00 -2.13747e+00 -1.89186e-01	 1.29122e+00 -7.49380e-01  6.71712e-01
-8.41923e-01  4.64997e+00  5.65808e-01 -3.08277e-01 -1.01687e+00  1.73127e-01 -8.92130e-01	1.89044e+00 -2.75543e-01 -7.72828e-01  5.36745e-01
-3.65598e+00  7.56997e+00 -3.76373e+00 -1.74542e+00 -1.37540e-01 -5.55400e-01 -1.59195e-01	1.27910e-01	 1.91906e+00  1.42119e+00 -4.35502e+00

-1.70059e+00 -3.65695e+00  1.22367e+00 -5.74367e-01 -3.29571e+00  2.46316e+00  5.22353e+00	2.42038e+00	 1.22919e+00 -9.22250e-01 -2.32028e+00


 0.00000e+00 
 1.00000e+00"""
		f.close()
		conf_args['STARNNW_NAME'] = '.pysex.nnw'
		
	return conf_file, conf_args
	
def _setup(conf_file, params):
	try:
		shutil.copy(conf_file, '.pysex.sex')
	except:
		pass #already created in _check_files

	f=open('.pysex.param', 'w')
	print>>f, '\n'.join(params)
	f.close()
	
def _setup_img(image, name):
	if not type(image) == type(''):
		import pyfits
		pyfits.writeto(name, image)
		

def _get_cmd(img, img_ref, conf_args,sexcmd='sex'):
	ref = img_ref if img_ref is not None else ''
	cmd = ' '.join([sexcmd, ref, img, '-c .pysex.sex '])
	args = [''.join(['-', key, ' ', str(conf_args[key])]) for key in conf_args]
	cmd += ' '.join(args)
	return cmd


def _read_cat(path = '.pysex.cat'):
	cat = asciidata.open(path)
	return cat

def _cleanup():
	files = [f for f in os.listdir('.') if '.pysex.' in f]
	for f in files:
		os.remove(f)

#def run(image='', imageref='', params=[], conf_file=DEFAULT_CONF, conf_args={}):
def pysex_run(image='', imageref='', params=[], conf_file=None, conf_args={}, keepcat=True, rerun=False, catdir=None):
	"""
	Run sextractor on the given image with the given parameters.
	
	image: filename or numpy array
	imageref: optional, filename or numpy array of the the reference image
	params: list of catalog's parameters to be returned
	conf_file: optional, filename of the sextractor catalog to be used
	conf_args: optional, list of arguments to be passed to sextractor (overrides the parameters in the conf file)
	
	
	keepcat : should I keep the sex cats ?
	rerun : should I rerun sex even when a cat is already present ?
	catdir : where to put the cats (default : next to the images)
	
	
	Returns an asciidata catalog containing the sextractor output
	
	Usage exemple:
		import pysex
		cat = pysex.run(myimage, params=['X_IMAGE', 'Y_IMAGE', 'FLUX_APER'], conf_args={'PHOT_APERTURES':5})
		print cat['FLUX_APER']
	"""
	# Set sexcmd from global
        global sexcmd
	# Preparing permanent catalog filepath :
	(imgdir, filename) = os.path.split(image)
	(common, ext) = os.path.splitext(filename)
	catfilename = common + ".pysexcat" # Does not get deleted by _cleanup(), even if in working dir !
	if keepcat:
		if catdir:
			if not os.path.isdir(catdir):
				os.makedirs(catdir)
				#raise RuntimeError("Directory \"%s\" for pysex cats does not exist. Make it !" % (catdir))

	if catdir:	
		catpath = os.path.join(catdir, catfilename)
	else:
		catpath = os.path.join(imgdir, catfilename)
	
	# Checking if permanent catalog already exists :
	if rerun == False and type(image) == type(''):
		if os.path.exists(catpath):
			cat = _read_cat(catpath)
			return cat
	
	# Otherwise we run sex :
	conf_args['CATALOG_NAME'] = '.pysex.cat'
	conf_args['PARAMETERS_NAME'] = '.pysex.param'
	if 'VERBOSE_TYPE' in conf_args and conf_args['VERBOSE_TYPE']=='QUIET':
		verbose = False
	else: verbose = True 
	_cleanup()
	if not type(image) == type(''):
		import pyfits
		im_name = '.pysex.fits'
		pyfits.writeto(im_name, image.transpose())
	else: im_name = image
	if not type(imageref) == type(''):
		import pyfits
		imref_name = '.pysex.ref.fits'
		pyfits.writeto(imref_name, imageref.transpose())
	else: imref_name = imageref
	conf_file, conf_args = _check_files(conf_file, conf_args, verbose)
	_setup(conf_file, params)
	cmd = _get_cmd(im_name, imref_name, conf_args,sexcmd)
	res = os.system(cmd)
	if res:
		#try different cmd for sextractor
		sexcmd = 'sextractor'
		cmd = _get_cmd(im_name, imref_name, conf_args,sexcmd)
		res = os.system(cmd)
		if res:
			print "Error during sextractor execution!"
			_cleanup()
			return

	# Keeping the cat at a permanent location :
	if keepcat and type(image) == type(''):
		shutil.copy('.pysex.cat', catpath)
	
	# Returning the cat :		 
	cat = _read_cat()
	_cleanup()
	return cat

###quad
class Quad:
	"""
	A geometric "hash", or asterism, as used in Astrometry.net :
	http://adsabs.harvard.edu/cgi-bin/bib_query?arXiv:0910.2233
	It is made out of 4 stars, and it is shift / scale / rotation invariant
	"""
	
	def __init__(self, fourstars):
		"""
		fourstars is a list of four stars
		
		We make the following attributes :
		self.hash
		self.stars (in the order A, B, C, D)
		
		"""
		assert len(fourstars) == 4
		
		tests = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
		other = [(2,3), (1,3), (1,2), (0,3), (0,2), (0,1)]
		dists = np.array([fourstars[i].distance(fourstars[j]) for (i,j) in tests])
		assert np.min(dists) > 1.0

		maxindex = np.argmax(dists)
		(Ai, Bi) = tests[maxindex] # Indexes of stars A and B
		(Ci, Di) = other[maxindex] # Indexes of stars C and D
		A = fourstars[Ai]
		B = fourstars[Bi]
		C = fourstars[Ci]
		D = fourstars[Di]
		
		# We look for matrix transform [[a -b], [b a]] + [c d] that brings A and B to 00 11 :
		x = B.x - A.x
		y = B.y - A.y
		b = (x-y)/(x*x + y*y)
		a = (1.0/x) * (1.0 + b*y)
		c = b*A.y - a*A.x 
		d = - (b*A.x + a*A.y)
		
		t = SimpleTransform((a, b, c, d))
		
		# Test
		#print t.apply((A.x, A.y))
		#print t.apply((B.x, B.y))
		
		(xC, yC) = t.apply((C.x, C.y))
		(xD, yD) = t.apply((D.x, D.y))
		
		# Normal case
		self.hash = (xC, yC, xD, yD)
		
		# Break symmetries :
		testa = xC > xD
		testb = xC + xD > 1
		
		if testa and not testb: # we switch C and D
			#print "a"
			self.hash = (xD, yD, xC, yC)
			(C, D) = (D, C)
		
		if testb and not testa: # We switch A and B
			#print "b"
			self.hash = (1.0-xD, 1.0-yD, 1.0-xC, 1.0-yC)
			(A, B) = (B, A)
			(C, D) = (D, C)
			
		if testa and testb:
			#print "a + b"
			self.hash = (1.0-xC, 1.0-yC, 1.0-xD, 1.0-yD)
			(A, B) = (B, A)
	
		# Checks :
		assert self.hash[0] <= self.hash[2]
		assert self.hash[0] + self.hash[2] <= 1
		
		self.stars = [A, B, C, D] # Order might be different from the fourstars !
		
		
	def __str__(self):
		return "Hash : %6.3f %6.3f %6.3f %6.3f / IDs : (%s, %s, %s, %s)" % (
			self.hash[0], self.hash[1], self.hash[2], self.hash[3],
			self.stars[0].name, self.stars[1].name, self.stars[2].name, self.stars[3].name)


def mindist(fourstars):
	"""
	Function that tests if 4 stars are suitable to make a good quad...
	"""
	tests = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
	dists = np.array([fourstars[i].distance(fourstars[j]) for (i,j) in tests])
	return np.min(dists)



def makequads1(starlist, n=7, s=0, d=50.0, verbose=True):
	"""
	First trivial quad maker.
	Makes combis of the n brightest stars.
	
	:param n: number of stars to consider (brightest ones).
	:type n: int
	:param s: how many of the brightest stars should I skip ?
		This feature is useful to avoid building quads with nearly saturated stars that are not
		available in other exposures.
	:type s: int
	:param d: minimal distance between stars
	:type d: float
	
	"""
	quadlist = []
	sortedstars = sortstarlistbyflux(starlist)

	for fourstars in itertools.combinations(sortedstars[s:s+n], 4):
		if mindist(fourstars) > d:
				quadlist.append(Quad(fourstars))	
	
	if verbose:
		print "Made %4i quads from %4i stars (combi n=%i s=%i d=%.1f)" % (len(quadlist), len(starlist), n, s, d)
		
	return quadlist



	
def makequads2(starlist, f=5.0, n=6, s=0, d=50.0, verbose=True):
	"""
	Similar, but fxf in subareas roughly f times smaller than the full frame.
	s allows to skip the brightest stars in each region
	
	:param f: smallness of the subareas
	:type f: float
	:param n: number of stars to consider in each subarea
	:type n: int
	:param d: minimal distance between stars
	:type d: float
	:param s: number of brightest stars to skip in each subarea
	:type s: int
	
	"""
	quadlist = []
	sortedstars = sortstarlistbyflux(starlist)
	(xmin, xmax, ymin, ymax) =  area(sortedstars)
	
	r = 2.0 * max(xmax - xmin, ymax - ymin) / f
	
	for xc in np.linspace(xmin, xmax, f+2)[1:-1]:
		for yc in np.linspace(ymin, ymax, f+2)[1:-1]:
			cstar =  Star(x=xc, y=yc)
			das = cstar.distanceandsort(sortedstars)
			#closest = [s["star"] for s in das[0:4]]
			brightestwithinr = sortstarlistbyflux([element["star"] for element in das if element["dist"] <= r])[s:s+n]
			for fourstars in itertools.combinations(brightestwithinr, 4):
				if mindist(fourstars) > d:
					quadlist.append(Quad(fourstars))
			
	if verbose:
		print "Made %4i quads from %4i stars (combi sub f=%.1f n=%i s=%i d=%.1f)" % (len(quadlist), len(starlist), f, n, s, d)

	return quadlist





def removeduplicates(quadlist, verbose=True):
	"""
	Returns a quadlist without quads with identical hashes...
	"""
	# To avoid crash in lexsort if quadlist is too small :
	if len(quadlist) < 2:
		return quadlist
	hasharray = np.array([q.hash for q in quadlist])
	
	order = np.lexsort(hasharray.T)
	hasharray = hasharray[order]
	#diff = np.diff(hasharray, axis=0)
	diff = np.fabs(np.diff(hasharray, axis=0))
	#diff = np.sum(diff, axis=1)
	ui = np.ones(len(hasharray), 'bool')
	ui[1:] = (diff >= 0.000001).any(axis=1)
	#print hasharray[ui==False]
	if verbose:
		print "Removing %i/%i duplicates" % (len(quadlist) - np.sum(ui), len(quadlist))
	
	return [quad for (quad, u) in zip(quadlist, ui) if u == True] 
	


def proposecands(uknquadlist, refquadlist, n=5, verbose=True):
	"""
	Function that identifies similar quads between the unknown image and a reference.
	Returns a dict of (uknquad, refquad, dist, trans)
	"""
	# Nothing to do if the quadlists are empty ...
	if len(uknquadlist) == 0 or len(refquadlist) == 0:
		if verbose:
			print "No quads to propose ..."
		return []
	
	if verbose:
		print "Finding %i best candidates among %i x %i (ukn x ref)" % (n, len(uknquadlist), len(refquadlist))
	uknhashs = np.array([q.hash for q in uknquadlist])	
	refhashs = np.array([q.hash for q in refquadlist])
	
	# Brute force...
	dists = scipy.spatial.distance.cdist(refhashs, uknhashs)
	uknmindistindexes = np.argmin(dists, axis=0) # For each ukn, the index of the closest ref
	uknmindist = np.min(dists, axis=0) # The corresponding distances
	uknbestindexes = np.argsort(uknmindist)
	
	candlist = []
	for i in range(n):
	
		cand = {"uknquad": uknquadlist[uknbestindexes[i]], "refquad":refquadlist[uknmindistindexes[uknbestindexes[i]]],
			"dist":uknmindist[uknbestindexes[i]]}
					
		cand["trans"] = quadtrans(cand["uknquad"], cand["refquad"])
		
		candlist.append(cand)
		if verbose:
			print "Cand %2i (dist. %12.8f) : %s" % (i+1, cand["dist"], str(cand["trans"]))
	
	return candlist
	

def quadtrans(uknquad, refquad):
	"""
	Quickly return a transform estimated from the stars A and B of two quads.
	"""
	return fitstars(uknquad.stars[:2], refquad.stars[:2])
	
####star
"""
Overhaul of cosmouline's star module, for alipy2.
This module contains stuff for geometric matching algorithms.
"""

import sys, os
import math
import numpy as np
import operator # For sorting
import copy
import itertools
import scipy.linalg
import scipy.spatial




class Star:
	"""
	Simple class to represent a single source (usually stars, but not necessarily).
	In this module we often manipulate lists of such Star objects.
	"""

	def __init__(self, x=0.0, y=0.0, name="untitled", flux=-1.0, props={}, fwhm=-1.0, elon=-1.0):
		"""
		flux : Some "default" or "automatic" flux, might be a just good guess. Used for sorting etc.
		If you have several fluxes, colours, store them in the props dict.
		props : A placeholder dict to contain other properties of your choice (not required nor used by the methods).
		"""
		self.x = float(x)
		self.y = float(y)
		self.name = str(name)
		self.flux = float(flux)
		self.props = props
		self.fwhm = float(fwhm)
		self.elon = float(elon)
	
	def copy(self):
		return copy.deepcopy(self)
	
	def __getitem__(self, key) :
		"""
		Used for sorting list of stars.
		"""
		if key == 'flux':
			return self.flux
		if key == 'fwhm':
			return self.fwhm
		if key == 'elon':
			return self.elon
	
	def __str__(self):
		"""
		A string representation of a source.
		"""
		return "%10s : (%8.2f,%8.2f) | %12.2f | %5.2f %5.2f" % (self.name, self.x, self.y, self.flux, self.fwhm, self.elon)

	def coords(self, full=False):
		"""
		Returns the coords in form of an array.
		
		:param full: If True, I also include flux, fwhm, elon
		:type full: boolean

		"""
		if full:
			return np.array([self.x, self.y, self.flux, self.fwhm, self.elon])	
		else:
			return np.array([self.x, self.y])

	def distance(self, otherstar):
		"""
		Returns the distance between the two stars.
		"""
		return math.sqrt(np.sum((self.coords() - otherstar.coords())**2))

	def trigangle(self, otherstar):
		"""
		returns the "trigonometric" angle of the vector that goes from
		self to the otherstar, in degrees
		"""
		return math.atan2(otherstar.y - self.y, otherstar.x - self.x) * (180.0/math.pi) % 360.0
		
	def distanceandsort(self, otherstarlist):
		"""
		Returns a list of dicts(star, dist, origpos), sorted by distance to self.
		The 0th star is the closest.
		
		otherstarlist is not modified.
		"""
		import operator # for the sorting
		
		returnlist=[]
		for i, star in enumerate(otherstarlist):
			dist = self.distance(star)
			returnlist.append({'star':star, 'dist':dist, 'origpos':i})
		returnlist = sorted(returnlist, key=operator.itemgetter('dist')) # sort stars according to dist
		
		return returnlist

### And now some functions to manipulate list of such stars ###


def printlist(starlist):
	"""
	Prints the stars ...
	"""
	for source in starlist:
		print source

def listtoarray(starlist, full=False):
	"""
	Transforms the starlist into a 2D numpy array for fast manipulations.
	First index is star, second index is x or y
	
	:param full: If True, I also include flux, fwhm, elon
	:type full: boolean
	
	"""
	return np.array([star.coords(full=full) for star in starlist])
	
	
def area(starlist, border=0.01):
	"""
	Returns the area covered by the stars.
	Border is relative to max-min
	"""
	if len(starlist) == 0:
		return np.array([0, 1, 0, 1])

	if len(starlist) == 1:
		star = starlist[0]
		return np.array([star.x - 0.5, star.x + 0.5, star.y - 0.5, star.y + 0.5])

	a = listtoarray(starlist)
	(xmin, xmax) = (np.min(a[:,0]), np.max(a[:,0]))
	(ymin, ymax) = (np.min(a[:,1]), np.max(a[:,1]))
	xw = xmax - xmin
	yw = ymax - ymin
	xmin = xmin - border*xw
	xmax = xmax + border*xw
	ymin = ymin - border*yw
	ymax = ymax + border*yw
	return np.array([xmin, xmax, ymin, ymax])
	

def readmancat(mancatfilepath, verbose="True"):
	"""
	Reads a "manual" star catalog -- by manual, I mean "not written by sextractor".
	So this is typically a *short* file.
	
	Comment lines start with #, blank lines are ignored.
	The format of a data line is
	
	starname xpos ypos [flux]
	
	The data is returned as a list of star objects.
	"""
	
	if not os.path.isfile(mancatfilepath):	
		print "File does not exist :"
		print mancatfilepath
		print "Line format to write : starname xpos ypos [flux]"
		sys.exit(1)
		
	
	myfile = open(mancatfilepath, "r")
	lines = myfile.readlines()
	myfile.close
	
	table=[]
	knownnames = [] # We check for uniqueness of the names
	
	for i, line in enumerate(lines):
		if line[0] == '#' or len(line) < 4:
			continue
		elements = line.split()
		nbelements = len(elements)
		
		if nbelements != 3 and nbelements != 4:
			print "Format error on line", i+1, "of :"
			print mancatfilepath
			print "The line looks like this :"
			print line
			print "... but we want : starname xpos ypos [flux]"
			sys.exit(1)
		
		name = elements[0]
		x = float(elements[1])
		y = float(elements[2])
		if nbelements == 4:
			flux = float(elements[3])
		else:
			flux = -1.0	
		
		if name in knownnames:
			print "Error in %s" % (mancatfilepath)
			print "The name '%s' (line %i) is already taken." % (name, i+1)
			print "This is insane, bye !"
			sys.exit(1)
		knownnames.append(name)
		
		#table.append({"name":name, "x":x, "y":y, "flux":flux})
		table.append(Star(x=x, y=y, name=name, flux=flux))
	

	if verbose: print "I've read", len(table), "sources from", os.path.split(mancatfilepath)[1]
	return table


def readsexcat(sexcat, hdu=0, verbose=True, maxflag = 3, posflux = True, minfwhm=2.0, propfields=[]):
	"""
	sexcat is either a string (path to a file), or directly an asciidata catalog object as returned by pysex
	
	:param hdu: The hdu containing the science data from which I should build the catalog. 0 is primary. If multihdu, 1 is usually science.
		
	We read a sextractor catalog with astroasciidata and return a list of stars.
	Minimal fields that must be present in the catalog :

		* NUMBER
		* EXT_NUMBER
		* X_IMAGE
		* Y_IMAGE
		* FWHM_IMAGE
		* ELONGATION
		* FLUX_AUTO
		* FLAGS
		
	maxflag : maximum value of the FLAGS that you still want to keep. Sources with higher values will be skipped.
		* FLAGS == 0 : all is fine
		* FLAGS == 2 : the flux is blended with another one; further info in the sextractor manual.
		* FLAGS == 4	At least one pixel of the object is saturated (or very close to)
		* FLAGS == 8	The object is truncated (too close to an image boundary)
		* FLAGS is the sum of these ...
	
	posflux : if True, only stars with positive FLUX_AUTO are included.
	
	propfields : list of FIELD NAMES to be added to the props of the stars.
	
	I will always add FLAGS as a propfield by default.
	
	"""
	returnlist = []
	
	if isinstance(sexcat, str):
	
		import asciidata
		if not os.path.isfile(sexcat):
			print "Sextractor catalog does not exist :"
			print sexcat	
			sys.exit(1)
	
		if verbose : 
			print "Reading %s " % (os.path.split(sexcat)[1])
		mycat = asciidata.open(sexcat)
	
	else: # then it's already a asciidata object
		mycat = sexcat
		
	# We check for the presence of required fields :
	minimalfields = ["NUMBER", "X_IMAGE", "Y_IMAGE", "FWHM_IMAGE", "ELONGATION", "FLUX_AUTO", "FLAGS", "EXT_NUMBER"]
	minimalfields.extend(propfields)
	availablefields = [col.colname for col in mycat]
	for field in minimalfields:
		if field not in availablefields:
			print "Field %s not available in your catalog file !" % (field)
			sys.exit(1)
	
	if verbose : 
		print "Number of sources in catalog : %i" % (mycat.nrows)
		
	propfields.append("FLAGS")
	propfields = list(set(propfields))
		
	if mycat.nrows == 0:
		if verbose :
			print "No stars in the catalog :-("
	else :
		for i, num in enumerate(mycat['NUMBER']) :
			if mycat['FLAGS'][i] > maxflag :
				continue
			'''if mycat['EXT_NUMBER'][i] != hdu :
				continue'''
			flux = mycat['FLUX_AUTO'][i]
			if posflux and (flux < 0.0) :
				continue
			fwhm = mycat['FWHM_IMAGE'][i]
			if float(fwhm) <= minfwhm:
				continue
			
			props = dict([[propfield, mycat[propfield][i]] for propfield in propfields])
			
			newstar = Star(x = mycat['X_IMAGE'][i], y = mycat['Y_IMAGE'][i], name = str(num), flux=flux,
					props = props, fwhm = mycat['FWHM_IMAGE'][i], elon = mycat['ELONGATION'][i])
			
			returnlist.append(newstar)
	
	if verbose:
		print "I've selected %i sources" % (len(returnlist))
		
	return returnlist

def findstar(starlist, nametofind):
	"""
	Returns a list of stars for which name == nametofind
	"""
	foundstars = []
	for source in starlist:
		if source.name == nametofind:
			foundstars.append(source)
	return foundstars

def sortstarlistbyflux(starlist):
	"""
	We sort starlist according to flux : highest flux first !
	"""
	sortedstarlist = sorted(starlist, key=operator.itemgetter('flux'))
	sortedstarlist.reverse()
	return sortedstarlist

def sortstarlistby(starlist, measure):
	"""
	We sort starlist according to measure : lowest first !
	Where measure is one of flux, fwhm, elon
	"""
	sortedstarlist = sorted(starlist, key=operator.itemgetter(measure))
	return sortedstarlist









class SimpleTransform:
	"""
	Represents an affine transformation consisting of rotation, isotropic scaling, and shift.
	[x', y'] = [[a -b], [b a]] * [x, y] + [c d]
	"""
	
	def __init__(self, v = (1, 0, 0, 0)):
		"""
		v = (a, b, c, d)
		"""
		self.v = np.asarray(v)
	
	def getscaling(self):
		return math.sqrt(self.v[0]*self.v[0] + self.v[1]*self.v[1])
		
	def getrotation(self):
		"""
		The CCW rotation angle, in degrees
		"""
		return math.atan2(self.v[1], self.v[0]) * (180.0/math.pi)# % 360.0
	
	def __str__(self):
		return "Rotation %+11.6f [deg], scale %8.6f" % (self.getrotation(), self.getscaling())
	
	
	def inverse(self):
		"""
		Returns the inverse transform !
		"""
		
		# To represent affine transformations with matrices, we can use homogeneous coordinates.
		homo = np.array([
		[self.v[0], -self.v[1], self.v[2]],
		[self.v[1],  self.v[0], self.v[3]],
		[0.0, 0.0, 1.0]
		])
		
		inv = scipy.linalg.inv(homo)
		#print inv
		
		return SimpleTransform((inv[0,0], inv[1,0], inv[0,2], inv[1,2]))
		
		
	
	def matrixform(self):
		"""
		Special output for scipy.ndimage.interpolation.affine_transform
		Returns (matrix, offset)
		"""
		
		return (np.array([[self.v[0], -self.v[1]], [self.v[1], self.v[0]]]), self.v[2:4])
		
	
	def apply(self, (x, y)):
		"""
		Applies the transform to a point (x, y)
		"""
		xn = self.v[0]*x -self.v[1]*y + self.v[2]
		yn = self.v[1]*x +self.v[0]*y + self.v[3]
		return (xn, yn)
		
	def applystar(self, star):
		transstar = star.copy()
		(transstar.x, transstar.y) = self.apply((transstar.x, transstar.y))
		return transstar
	
	def applystarlist(self, starlist):
		return [self.applystar(star) for star in starlist]
	
	
def fitstars(uknstars, refstars, verbose=True):
	"""
	I return the transform that puts the unknown stars (uknstars) onto the refstars.
	If you supply only two stars, this is using linalg.solve() -- perfect solution.
	If you supply more stars, we use linear least squares, i.e. minimize the 2D error.
	
	Formalism inspired by :
	http://math.stackexchange.com/questions/77462/
	"""
	
	assert len(uknstars) == len(refstars)
	if len(uknstars) < 2:
		if verbose:
			print "Sorry I cannot fit a transform on less than 2 stars."
		return None
	
	# ukn * x = ref
	# x is the transform (a, b, c, d)
	
	ref = np.hstack(listtoarray(refstars)) # a 1D vector of lenth 2n
	
	uknlist = []
	for star in uknstars:
		uknlist.append([star.x, -star.y, 1, 0])
		uknlist.append([star.y, star.x, 0, 1])
	ukn = np.vstack(np.array(uknlist)) # a matrix
	
	if len(uknstars) == 2:
		trans = scipy.linalg.solve(ukn, ref)
	else:
		trans = scipy.linalg.lstsq(ukn, ref)[0]
	
	return SimpleTransform(np.asarray(trans))
def identify(uknstars, refstars, trans=None, r=5.0, verbose=True, getstars=False):
	"""
	Allows to:
	 * get the number or matches, i.e. evaluate the quality of the trans
	 * get corresponding stars from both lists (without the transform applied)
	
	:param getstars: If True, I return two lists of corresponding stars, instead of just the number of matching stars
	:type getstars: boolean

	Inspired by the "formpairs" of alipy 1.0 ...

	"""
	
	if trans != None:
		ukn = listtoarray(trans.applystarlist(uknstars))
	else:
		ukn = listtoarray(uknstars)
	ref = listtoarray(refstars)
	
	dists = scipy.spatial.distance.cdist(ukn, ref) # Big table of distances between ukn and ref
	mindists = np.min(dists, axis=1) # For each ukn, the minimal distance
	minok = mindists <= r # booleans for each ukn
 	minokindexes = np.argwhere(minok).flatten() # indexes of uknstars with matches
	
	if verbose:
		print "%i/%i stars with distance < r = %.1f (mean %.1f, median %.1f, std %.1f)" % (np.sum(minok), len(uknstars), r, 
			np.mean(mindists[minok]), np.median(mindists[minok]), np.std(mindists[minok]))
	
	matchuknstars = []
	matchrefstars = []
	
	for i in minokindexes: # we look for the second nearest ...
		sortedrefs = np.argsort(dists[i,:])
		firstdist = dists[i,sortedrefs[0]] 
		seconddist = dists[i,sortedrefs[1]]
		if seconddist > 2.0*firstdist: # Then the situation is clear, we keep it.
			matchuknstars.append(uknstars[i])
			matchrefstars.append(refstars[sortedrefs[0]])
		else:
			pass # Then there is a companion, we skip it.
	
	if verbose:
		print "Filtered for companions, keeping %i/%i matches" % (len(matchuknstars), np.sum(minok))
	
	if getstars==True:
		return (matchuknstars, matchrefstars)
	else:
		return len(matchuknstars)

###run from command line
if __name__=='__main__':
    #steps for data reduction

    #combine bias

    #combine darks

    #make master dark

    #calibrate each flat with master dark

    #transform flats so means and std are simmilar

    #combine flats to make master dark

    #calibrate each image with master dark and master flat

    #align images

    #combine images 
    
    #example on how to make master flat
    
    '''
    def make_master_flat(dark, bias=None, cal=False, path=None, combine_type='mean',
                     outdir=None, Filter='FILTER'):
    ''''''(dict(ndarray), dict(ndarray), str, str, str, str) -> dict(ndarray)

    Sky flats may have variable brightness because of the dimming of the
    sky. This program helps to better calibrate these by doing the calibration
    on the individual images and then normalizing the flats so they can better
    be averaged together.

    Takes in a dark and bias, unless dark has been calibrated, then cal=True 
    and no bias is required.

    Returns a calibrated, normalized and combined flat field sorted by Filter
    option.
    ''''''
    #gui get flats
    if path is None:
        path = gui_getdir(title='Please Select Flats dir')
        if not path:
            raise ValueError('Must specify directory where files are.')
    if not path.endswith('/'):
        path += '/'
    if outdir is None:
        outdir = gui_getdir(title='Please Select Save Directory')
    #load path to flats
    fits_path = sorted(glob(path+'*'))
    fits_path = get_fits_type(fits_path,'flat')
    filters = {}
    #sort flats by filter
    for i in fits_path:
        filt = str(fits.getval(i,Filter))
        if not filt in filters.keys():
            filters[filt] = [i]
        else:
            filters[filt].append(i)
    ######calibrate individual images
    #create temp dir to store calbirated flats
    if not os.path.isdir('temp'):
        os.mkdir('temp')
    out, hdr = {}, {}
    for i in filters.keys():
        #hold individual flats and their fits headers
        temp_flat, temp_hdr = [], []
        for k,j in enumerate(filters[i]):
            #load flat
            temp = fromfits(j,verbose=False)
            temp_flat.append(temp[0]) #assgin
            temp_hdr.append(temp[1]) #assign header
            #try to select correct dark
            dark_key = (str(temp_hdr[-1]['SET-TEMP']) + '_' + 
                        str(temp_hdr[-1]['EXPTIME']) + '_')
            if not dark.has_key(dark_key):
                #fix later
                raise KeyError('Dark must have "%s" key'%dark_key)
            if cal: #if dark is a master dark
                temp_flat[-1] -= dark[dark_key] #dark subtract
                #put dark subtraction into fits header
                temp_hdr[-1].add_history('Dark subtracted')
            else:
                #if dark not master use bias
                temp_flat[-1] -= (dark[dark_key] + bias[bias.keys()[0]])
                temp_hdr[-1].add_history('Dark and Bias subtracted')
            #save calibrated image
            tofits('temp/' + j[j.rfind('/')+1:], temp_flat[-1], 
                   temp_hdr[-1], False)
            filters[i][k] = 'temp/' + j[j.rfind('/')+1:]
        #make master for filter hdr
        hdr[i] = temp_hdr[-1]
        #combine
        temp_flat = nu.dstack(temp_flat) #turn list to ndarray
        for k in range(temp_flat.shape[2]):
            #scale flats so can be combined
            mean = nu.mean(temp_flat[:,:,k])
            std = nu.std(temp_flat[:,:,k])
            temp_flat[:,:,k] = (temp_flat[:,:,k] - mean)/std
        #different combine types
        if combine_type == 'mean':
            out[i] = nu.mean(temp_flat,2)
            hdr[i].add_history('Mean combine')
        elif combine_type == 'median':
            out[i] = nu.median(temp_flat,2)
            hdr[i].add_history('Median combine')
        elif combine_type == 'sigmaclip':
            out[i] = Sigmaclip(temp_flat, axis=2)
            hdr[i].add_history('Sigma clip combine')
        elif combine_type == 'sum':
            #crash for now
            out[i] = temp_flat[:,:,0] + nu.nan
            hdr[i].add_history('Summed Images')
        else:
            raise ValueError('Do not even know how this error came up')
        #normalize flat
        out[i] -=  out[i].min()
        out[i] /= out[i]/out[i].max()
        hdr[i].add_history('Normalized')
    #save master flat as fits?
    if outdir is None:
        outdir = path
    else:
        if not outdir.endswith('/'):
            outdir += '/'
    for i in out.keys():
        basename = os.path.split(filters[i][0])[-1]
        basename = os.path.splitext(basename)[0]
        tofits(outdir+basename+'_%s.fits'%combine_type.lower(), out[i],
               hdr[i],verbose=False)
    return out, hdr
    '''
