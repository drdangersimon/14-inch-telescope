'''Reduices images from 14 inch for use with science
#Thuso Simon 8 Mar 2013'''
import tkFileDialog as tk
import numpy as np
import pylab as lab
import pyfits as fits
#import set_cord 
from glob import glob
import os,csv,math,shutil,sys
import asciidata,operator,copy,itertools
import scipy.spatial as spatial
import scipy.ndimage as ndimage #can use for filters (high/low pass)
import align
import utilities as util
import organize_fits as org_fits
from scipy.stats import sigmaclip

###main program
def get_flats(path=None, combine_type='median', outdir=None,
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
        
    else:
        # make dir if don't exisits
        if  not os.path.exists(outdir):
            os.mkdir(outdir)
    #load path to flats
    fits_path = org_fits.get_filelist(path)
    fits_path = util.get_fits_type(fits_path,'flat')
    filters = {}
    #sort flats by filter
    for i in fits_path:
        filt = str(fits.getval(i,Filter))
        if not filt in filters.keys():
            filters[filt] = [i]
        else:
            filters[filt].append(i)
    
    out,hdr = {},{}
    #save as fits?
    if outdir is None:
        outdir = path
    else:
        if not outdir.endswith('/'):
            outdir += '/'
    for i in filters:
        # combine flats for each filter
        out[i], hdr[i] = comm(filters[i])
        # Normailize by mean
        out[i] /= out[i].mean()
        hdr[i].add_history('Normalized') #nomalizing the flats

        basename = os.path.split(filters[i][0])[-1]
        basename = os.path.splitext(basename)[0]
        util.tofits(outdir+basename+'_%s.fits'%combine_type.lower(), out[i],
               hdr[i],verbose=False)
    return out, hdr

def get_darks(path=None, combine_type='median', outdir=None,
              Filter=('SET-TEMP','EXPTIME')):
    '''(str,str,str,tuple(str) or str) -> dict(ndarry), dict(asciidata)

    Opens dark directory, combines all *.fits or *.fit files in the 
    directory, sorts by Filter options (can have multiple) and out 
    puts fits file to outdir and also out ndarray of combined fits 
    files and fits header with modified history.  Combine types can 
    include: "sigmaclip","median","sum" and "sigmaclip."
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
    fits_path = util.get_fits_type(fits_path,'dark')
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
        # make dir if don't exisits
        if  not os.path.exists(outdir):
            os.mkdir(outdir)
    for i in out.keys():
        basename = os.path.split(filters[i][0])[-1]
        basename = os.path.splitext(basename)[0]
        util.tofits(outdir+basename+'_%s.fits'%combine_type.lower(), out[i],
               hdr[i],verbose=False)
    return out,hdr



def get_bias(path=None, combine_type='median', outdir=None, Filter='SET-TEMP'):
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
    fits_path = util.get_fits_type(fits_path,'bias')
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
        # make dir if don't exisits
        if  not os.path.exists(outdir):
            os.mkdir(outdir)
 
    for i in out.keys():
        basename = os.path.split(filters[i][0])[-1]
        basename = os.path.splitext(basename)[0]
        util.tofits(outdir+basename+'_%s.fits'%combine_type.lower(), out[i],
               hdr[i],verbose=False)
    return out,hdr

  
def stack_images(path=None, combine_type='median', outdir=None):
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
    light_path = util.get_fits_type(light_path,'light')
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
        # make dir if don't exisits
        if not os.path.exists(outdir):
            os.mkdir(outdir)

    for i in out.keys():
        basename = os.path.split(filters[i][0])[-1]
        basename = os.path.splitext(basename)[0]
        util.tofits(outdir+basename+'_%s.fits'%combine_type.lower(), out[i],
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
    light_path = util.get_fits_type(light_path,'light')
    #start allignment
    ident = align.ident_run(light_path[0],light_path)
    #see if any images succesfully got aligned
    Shape = util.shape(light_path[0],verbose=False)
    #save algned fits to outdir
    for id in ident:
        if id.ok:
            align.affineremap(id.ukn.filepath, id.trans, 
                        Shape,outdir=outdir)

####supporting programs
def Normalize(a):
    '''(ndarry) -> ndarry

    Normalizes an array so that all values are in [0,1]. 
    >>> a = numpy.array([-1.,0.,1.],dtype=float)
    >>> Normalize(a)
    array([ 0. ,  0.5,  1. ])
    '''
    a = np.asarray(a)
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
    path = tk.askdirectory(initialdir=initdir, title=title)
    if not path.endswith( '/' ):
      path += '/'
    return path



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
        temp = np.zeros(shape)
        for i,j in enumerate(file_list):
            temp[:,:,i],hdr = fits.fromfits(j,verbose=False)
        out = Sigmaclip(temp,axis=2)
    else:
        #slow ram but takes longer
        shape = (fits.getval(file_list[0],'NAXIS1'),
                 fits.getval(file_list[0],'NAXIS2'))
        out = np.zeros(shape)
        temp = np.zeros((shape[0],len(file_list)))
        for i in xrange(shape[1]):
            for j,k in enumerate(file_list):
                temp[:,j] = util.fromfits(k,verbose=False)[0][:,i]
            out[:,i] = Sigmaclip(temp,axis=1)
        temp, hdr = util.fromfits(k,verbose=False)
    hdr.add_history('Sigmaclip combine')
    return out, hdr

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
        temp = np.zeros(shape)
        for i,j in enumerate(file_list):
            temp[:,:,i],hdr = util.fromfits(j,verbose=False)
        out = np.median(temp,2)
    else:
        #low ram but takes longer
        shape = (fits.getval(file_list[0],'NAXIS1'),
                 fits.getval(file_list[0],'NAXIS2'))
        out = np.zeros(shape)
        temp = np.zeros((shape[0],len(file_list)))
        for i in xrange(shape[1]):
            for j,k in enumerate(file_list):
                temp[:,j] = util.fromfits(k, verbose=False)[0][:,i]
            out[:,i] = np.median(temp,1)
        temp,hdr = util.fromfits(k, verbose=False)
    hdr.add_history('Medium combine')
    return out, hdr

def combine_sum(file_list):
    '''(str) -> ndarray, acsciidata

    Puts list of fits files into correct order for use with
    sum function. Adds to fits header histrory.

    '''
    #sums images together
    out = False
    for i in file_list:
        if np.any(out):
            try:
                out += util.fromfits(i,verbose=False)[0]
            except ValueError:
                print i
                raise
        else:
            out, hdr = util.fromfits(i,verbose=False)
    hdr.add_history('Summed from %i images'%len(file_list))
    return out,hdr

def combine_SDmask(file_list, num=70):
    '''(str, int) -> ndarray, acsciidata

    Puts list of fits files into correct order for use with
    SD masking function. Adds to fits header histrory.

    '''
    raise NotImplementedError

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
        return np.mean(sigmaclip(c)[0])
    #create masked array
    c_mask = np.ma.masked_array(c,np.isnan(c))
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
               c_mask[indexer].mask = np.logical_and(
                   c_mask[indexer].squeeze() > critlower, 
                   c_mask[indexer].squeeze()<critupper) == False
           delta = size - c_mask.mask.sum()
    return c_mask.mean(axis).data





    
