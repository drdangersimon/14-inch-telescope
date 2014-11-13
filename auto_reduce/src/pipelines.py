import tkFileDialog as tk
import numpy as np
import pylab as plt
import pyfits  as fits
from glob import glob
import os,csv,math,shutil,sys
import reduice
import organize_fits as org_fits
import utilities as util
import tkMessageBox
import ipdb

def SubtractBias(path=None, outdir=None):
    '''(str,str) -> None 
    Opens the various directories(Dark,Flat and Light) and prepares  
    subtracts the master bais form all the fits files and stores them
    in Biased dir'''
    # get files if no path is passed then gui will come up
    if path == None:
        path = reduice.gui_getdir(title='Please Select FITS Directory')
    # get biases and fits header list
    bias_dic,bias_hdr = reduice.get_bias(os.path.join(path,'Bias'),outdir=os.path.join(path,'Reduced'))
    if bias_dic == {}:
        return
    #decoupling the tuple and dictionary to get the indiviual bias and hdr arrays 
    bias = bias_dic.values()[0]
    hdr = bias_hdr.values()[0]
    dim_bias = util.shape(hdr)
    # gets all the files in a dir and all its sub dirs
    files = org_fits.get_filelist(path)
    path += 'Reduced/Biased'
    #Loops through all the files in the dir
    for fit in files:
        #rejects all bias files and the non-fits files
		if not util.fitype(fit,'Bias'): 
			#print 'not afits'+fit
			continue
		#gets the array and header of the fit 
		fits_array, hdr_fit = util.fromfits(fit, verbose=False )

		dim_fits = util.shape(hdr_fit)

		#checks if the bias and fits have the same dimension
		if dim_fits == dim_bias:
			new_array = fits_array - bias
		else:
			print 'arrays not the same sizes'
			continue
			
		util.savefits('bias subtracted', path, fit, new_array, hdr_fit)
			
def RemoveDarks(path=None):
    ''' Removes master dark from fits from the flatfielded dir and stores them in Darked dir'''
    if path==None:
        path = reduice.gui_getdir(title='Please Select FITS Directory')
    #gets darks
    dark_dic, hdr = reduice.get_darks(path=os.path.join(path,'Darks'),
                                      outdir=os.path.join(path, 'Reduced/'))
    # 
    light_dir = glob(os.path.join(path,'Light/*/*'))
    outpath = os.path.join(path, 'Reduced/Darked')
    #decompling the dark arrays and header
    darks = dark_dic.values()
    hdrs = hdr.values()
    #iterates throught the darks
    for i in xrange(len(darks)):
        #ipdb.set_trace()
        # exposure time of the ith dark
        dark_exptime = hdrs[i]['EXPTIME']
        # array of the ith dark
        dark_array = darks[i]
        # dimension of the ith dark
        dim_dark = util.shape(hdrs[i]) 
        #iterates through the biased fits
        for fit in light_dir:
            #checks if fit is a fits, also rejects darks
            if not util.fitype(fit,'Dark'): 
                print '%s is not a fits file'%fit
                continue
            #exposure time of the fit
            light_exptime = fits.getval(fit,'EXPTIME') 
            # the ratio of expected time of light and dark
            factor = light_exptime / dark_exptime 
            fits_array,	hdr = util.fromfits( fit ,verbose=False)
            dim_fits = util.shape(hdr) #dimension of fits
            #checks dimensions
            if dim_fits == dim_dark:
                new_array = fits_array - dark_array * factor
            else: 
                print 'arrays not the same sizes'
            util.savefits('dark removed', outpath , fit, new_array, hdr)



def RemoveFlat(path=None):
    '''Divides the master flat from the fits and stores them in Flatfielded'''
    if path==None:
        path = reduice.gui_getdir(title='Please Select fits Directory')
    flats_dic, hdr = reduice.get_flats(path= os.path.join(path,'Flats'),
                                       outdir=os.path.join(path,'Reduced'))
    lights = glob(os.path.join(path, 'Reduced/Darked/*'))
    if len(lights) == 0:
        # try bias dir
        lights = glob(os.path.join(path, 'Reduced/Biased/*'))
        if len(lights) == 0:
            # no lights
            print 'Calibration not sucessful'
		
	# removes flat only on lights
    lights = util.get_fits_type(lights,'light') 
    flats = flats_dic.values()  
    flit = flats_dic.keys()  
    hdrs = hdr.values()
    outpath = os.path.join(path, 'Reduced/Flatfielded')
    ipdb.set_trace()
	#iterating through the different filter flats
    for i in xrange(len(flats)):
        
        Filter = flit[i]
        dim_flat = util.get_dim(hdrs[i])
        #iterating through the light files(biased or darked)
        for light in lights:
            light_filt = str(fits.getval(light,'FILTER'))
            #checks if the filters are the same   
            if Filter == light_filt:
                flat_arry = flats[i]
                light_arry ,hdr = util.fromfits(light,verbose=False)
                dim_fits = util.shape(hdr)
                if dim_fits == dim_flat:
                    light_arry = flat_arry.mean()*light_arry/flat_arry
                else:
                    print 'arrays not the same sizes'
                util.savefits('flat divided',outpath, light, light_arry, hdr)
				

def run(path=None):
    '''Calibrates data and sorts into files'''
    if path is None:
        path = reduice.gui_getdir(title='Please Select Fits Directory')
	if not path:
		raise ValueError('Must specify directory where files are.')
	if not path.endswith('/'):
		path += '/'
	org_fits.check_ccd_type(path)
	paths = org_fits.get_dir(path)
	for path in paths:
		org_fits.sort_fits(path)
		org_fits.products(path)
		print '****************Bias**************'
		SubtractBias(path=path)
		print '****************Dark**************'
		RemoveDarks(path)
		print '****************Flat**************'
		RemoveFlat(path=path)
		print '****************Done**************'
