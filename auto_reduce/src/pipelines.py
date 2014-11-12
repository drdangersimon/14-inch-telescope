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

def SubtractBias(path=None,outdir=None):
	'''(str,str) -> None 
   Opens the various directories(Dark,Flat and Light) and prepares  
   subtracts the master bais form all the fits files and stores them in Biased dir'''
	#get files if no path is passed then gui will come up
	if path == None:
	        path = gui_getdir(title='Please Select FITS Directory')
	#get biases and fits header list
    	bias_dic,bias_hdr = get_bias(path+'/Bias',outdir=path+'/Reduced')
    	if bias_dic=={}:
		return
	#decoupling the tuple and dictionary to get the indiviual bias and hdr arrays 
    	bias 	      	  = bias_dic.values()[0]
    	hdr               = bias_hdr.values()[0]

    	dim_bias         = get_dim(hdr)
    	         
 	#gets all the files in a dir and all its sub dirs
    	files 		  = divide.get_filelist(path)
	path		 += 'Reduced/Biased'
 
	#Loops through all the files in the dir
	for fit in files:
		if not fitype(fit,'Bias'): #rejects all bias files and the non-fits files
			#print 'not afits'+fit
			continue
		#gets the array and header of the fit 
		fits_array, hdr_fit = fromfits( fit,verbose=False )

		dim_fits       = get_dim(hdr_fit)

		#checks if the bias and fits have the same dimension
		if dim_fits == dim_bias:
			new_array       = fits_array - bias
		else:
			print 'arrays not the same sizes'
			continue
			
		savefits('bias subtracted',path,fit, new_array,hdr_fit)
			
def RemoveDarks(path=None):
	''' Removes master dark from fits from the flatfielded dir and stores them in Darked dir'''


	if path==None:
		path = gui_getdir(title='Please Select FITS Directory')
	#gets darks
	dark_dic,hdr        = get_darks(path=path+'/Darks', outdir=path+'/Reduced')

	#gets all the fits which biased hased been removed
	light_dir=glob(path+'/Reduced/Biased/*')
	path	    +='Reduced/Darked'

	#decompling the dark arrays and header
	darks = dark_dic.values()
	hdrs   = hdr.values()

	#iterates throught the darks
	for i in xrange(len(darks)):

		dark_exptime = hdrs[i]['EXPTIME'] #exposure time of the ith dark
		dark_array   = darks[i] #array of the ith dark
		dim_dark    = get_dim(hdrs[i]) #dimension of the ith dark
  		
  		#iterates through the biased fits
		for fit in light_dir:
			#checks if fit is a fits, also rejects darks
			if not fitype(fit,'Dark'): 
				print 'not afits'+fit
				continue

			light_exptime = pyfits.getval(fit,'EXPTIME') #exposure time of the fit

			factor = light_exptime / dark_exptime # the ratio of expected time of light and dark
			fits_array,	hdr = fromfits( fit ,verbose=False)
			dim_fits       = get_dim(hdr) #dimension of fits
	
			#checks dimensions
			if dim_fits == dim_dark:
				new_array       = fits_array - dark_array*factor

			else: 
			 print 'arrays not the same sizes'

			savefits('dark removed', path , fit, new_array,hdr)



def RemoveFlat(path=None):
	'''Divides the master flat from the fits and stores them in Flatfielded'''

	if path==None:
		path= gui_getdir(title='Please Select fits Directory')
 
	flats_dic,hdr =get_flats(path=path+'/Flats', outdir=path+'/Reduced/')
	lights=glob(path+'/Reduced/Darked/*')

	if len(lights) == 0: 
	#no darks
		print 'There are no darks. Calibration may not be accurate.'
		#use bias
		lights=glob(path+'/Reduced/Biased/*') 

	#removes flat only on lights
	lights=get_fits_type(lights,'light') 
	flats      = flats_dic.values()  
	flit       = flats_dic.keys()  
	hdrs        = hdr.values()
	path      += 'Reduced/Flatfielded'

	#iterating through the different filter flats
	for i in xrange(len(flats)):
		Filter     = flit[i]
		dim_flat   = get_dim(hdrs[i])
  
		#iterating through the light files(biased or darked)
		for light in lights:
			
			light_filt=str(pyfits.getval(light,'FILTER'))
			#checks if the filters are the same   
			if Filter == light_filt:
				flat_arry        = flats[i]
   
				light_arry ,hdr = fromfits(light,verbose=False)
    
				dim_fits       = get_dim(hdr)
    
				if dim_fits==dim_flat:
					light_arry     /= (flat_arry)
					light_arry	    *= flat_arry.max() 
				else:
					print 'arrays not the same sizes'

				savefits('flat divided',path,light,light_arry,hdr)
				
  
def savefits(hsty,path,fit,arry,hdr):
	'''(str,str,str,str)->None
	The method saves fits into their new dirs'''

	try:
	 new_path=fit[fit.rfind('/'):]
	 hdr.add_history(hsty)
	 tofits(path + new_path ,arry , hdr)
	except:
	 print "No such file or directory: %s from %s"%(path, fit)
	
def get_dim(hdr):
	'''Returns the dimensions of fit
	str -> tuple'''
	return (hdr['NAXIS1'],hdr['NAXIS2'])


def main():
	path      = gui_getdir( title = 'Please Select Fits Directory')
	if not path:
		raise ValueError( 'Must specify directory where files are.' )
	if not path.endswith( '/' ):
		path      += '/'

	divide.check_ccd_type(path)
	paths = divide.get_dir(path)
	for path in paths:
		print path
		divide.div(path)
		divide.products(path)
		print '****************Bias**************'
		SubtractBias(path=path)
		print '****************Dark**************'
		RemoveDarks(path)
		print '****************Flat**************'
		RemoveFlat(path=path)
		print '****************Done**************'

if __name__ == "__main__":
	main()
 

 
 
