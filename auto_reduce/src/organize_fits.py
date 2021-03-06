import os,csv,math,shutil,sys
import pyfits as fits
from glob import glob
import shutil
import utilities as util
import reduice


'''This program creates all the directories needed for the reduction process'''

def check_ccd_type(path=None):
	'''This method checks if we are dealing with spectal or photometry 
	data and splits them accordingly'''
	
	if path == None:
  		path = reduice.gui_getdir(title='Please Select raw fits Directory')
	
	if not os.path.exists(path+'Spectral/'):
  		os.makedirs(path+ 'Spectral' )
	if not os.path.exists(path+'Photometry/'):
  		os.makedirs(path+ 'Photometry' )
	
	#getting all the files in the main dir
 	files = get_filelist(path)
	#iterating through each file
	for fit in files:
		if not util.fitype(fit): #Checking if fit is a fits file
			continue
		ccdtype = fits.getval(fit,'INSTRUME') #instrument used to make fit
		#moving to appropriate dir
		if ccdtype == 'SBIG ST-7/7E/7XE':
			shutil.move(fit, os.path.join(path, 'Spectral'))
		else:
			shutil.move(fit,os.path.join(path, 'Photometry'))
	
	clean(path)

def get_dir(path):
	'''Returns the a list of the directories that needs to be 	reduced(spectral, photometry or both 
	str-> array'''
	arry = []
	if os.path.exists(os.path.join(path,'Spectral/')):
  		arry.append(os.path.join(path, 'Spectral/'))
	if os.path.exists(os.path.join(path,'Photometry/')):
  		arry.append(os.path.join(path, 'Photometry/'))
	
	return arry

			
def sort_fits(path=None):
    ''' This method sorts fits into directories according to the type of fits they are(bias, darks, flats and lights)'''
 
    if path==None:
        path=reduice.gui_getdir(title='Please Select raw fits Directory')

    keyword = ['bias','dark','flat','light']
 
    #creating the different dir to sort fits
    if not os.path.exists(os.path.join(path,'Bias')):
        os.makedirs(os.path.join(path, 'Bias' ))

    if not os.path.exists(os.path.join(path,'Darks/')):
        os.makedirs(os.path.join(path, 'Darks' ))

    if not os.path.exists(os.path.join(path,'Flats/')):
        os.makedirs(os.path.join(path,'Flats'))

    if not os.path.exists(os.path.join(path,'Light/')):
        os.makedirs(os.path.join(path,'Light'))
 

    files = get_filelist(path)
 
    for fit in files:
        #remove non light images
        if not (fit.endswith('.fit') or fit.endswith('.fits')):
            continue
        #getting the fit type eg(flat)
        imtype = fits.getval(fit,'IMAGETYP').lower()
        #moving fit to approapriate dir
        try:
            if imtype.startswith(keyword[0]):
                shutil.move(fit, os.path.join(path, 'Bias/'))
            elif imtype.startswith(keyword[1]):
                shutil.move(fit, os.path.join(path, 'Darks/'))
            elif imtype.startswith(keyword[2]):
                shutil.move(fit,os.path.join(path, 'Flats/'))
            elif imtype.startswith(keyword[3]):
                shutil.move(fit, os.path.join(path, 'Light/'))

        except shutil.Error as e:
            if e.message == "Destination path '%s' already exists"%fit:
                print "A file with the same name has already been sorted"

    sort_filters(os.path.join(path, 'Light/'))
    sort_filters(os.path.join(path, 'Flats/'))
    clean(path)
    #products(path)

def sort_filters(path=None):
	'''Sorts fits files in a directory into folders according to their filter ''' 
	if path == None:
  		path = reduice.gui_getdir(title='Please Select raw fits Directory')
	#getting all the files in the path dir
	files = get_filelist(path)
	#iterating through these files
	for fit in files:
		#removing all the non fits eg logfiles 
		if not util.fitype(fit):
			print fit+'  is not a fits'
			continue #move on if not fit
		#get the filter of the fit
		filt = fits.getval(fit,'FILTER')
		tempath = os.path.join(path, filt)
		#if no filter dir, create one
    		if not os.path.exists(tempath):
     			os.makedirs(tempath)

		try: #move the fit to its filter folder
    			shutil.move(fit, tempath)
		except shutil.Error as e:
   			if e.message == "Destination path '%s' already exists"%fit:
   				print "A file with the same name has already been"
		

def clean(path=None):
    '''(str)->  null removes empty dir '''
    if path == None:
        path = reduice.gui_getdir(title='Please Select raw fits Directory')
	#remove copy python files created during the process
	remove_pyfiles(path)
	#getting all files including dir
    files = glob(path+'/*')
	#interates through all content of path
    for f in files:
		#ignores files 
        if os.path.isdir(f):
            #if the dir is empty remove it
            if len(os.listdir(f)) == 0: 
                shutil.rmtree(f)
            else:
                #if its a dir then recursivly cleans it
				clean(f)

def remove_pyfiles(path=None):
    '''This method looks for python copy files that are somehow
    copied to the fits dir during the
    '''
    if path == None:
        path = reduice.gui_getdir(title='Please Select raw fits Directory')
	#finds all the files in path
	files = get_filelist(path)
	
	for f in files:
		#checks for python files and removes them
		if f.endswith('.py') or f.endswith('.pyc') or f.endswith('.py~'):
			os.remove(f)


def get_filelist(path=None):
	'''(str)--->(list)
	recursivly gets the filenames in a dir and its subdirs	
	''' 
	flist=[]
	for root, subdir, files in os.walk(path):
		for f in files:
			flist.append(os.path.join(root,f))
	return flist


#What about: Assuming any reduced files that already exist there will be overwritten
def products(outdir=None):
    '''Creates the reduced directory where the products will be saved. 
    fits with removed bias are in the Biased dir, those with darks removed are in the Darked dir
    and those with removed flat are saved in the Flatfielded directory'''
 
    if outdir == None:
        outdir = reduice.gui_getdir(title='Please Select save Directory')
    outdir += '/Reduced'
    if not os.path.exists(outdir): 
        os.makedirs(outdir)
    if not os.path.exists(os.path.join(outdir,'Flatfielded')):  
        os.makedirs(os.path.join(outdir,'Flatfielded'))
    if not os.path.exists(os.path.join(outdir,'Darked')):
        os.makedirs(os.path.join(outdir,'Darked'))
    if not os.path.exists(os.path.join(outdir,'Biased')):
        os.makedirs(os.path.join(outdir,'Biased'))


