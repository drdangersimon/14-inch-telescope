#set wcs coordinates of fits image
#Thuso Simon 8 Mar 2013

import os,sys
import subprocess as sub
import numpy as nu
import multiprocessing as M
from glob import glob
from time import time
import astropy.coordinates as coord

#######BEFOR RUNNING#####
'''program uses astrometry.net to register images. Make sure you have all the
programs needed to run it.'''

solve_cmd = 'solve-field'

def set_wcs_headers(fits_path,ra,dec,pix_scale):

    '''uses astromomtry.net to solve wcs coordinates overwrite files if success
    full.

    Assumes pix_scale is in arcseconds and ra and dec are degrees

    Returns bool of status of wcs'''
    #if ra and dec are strings assumes put as hh:mm:ss etc...
    if type(ra) is str:
        coord.FK5Coordinates
    args = ' '
    args += '--scale-units arcsecperpix '
    args += '--scale-low %f --scale-high %f '%(pix_scale-pix_scale*.1,
                                                pix_scale+pix_scale*.1)
    args += '--no-plots --fits-image --overwrite --cpulimit 500 '
    args += '-ra %f --dec %f --radius 1 '%(ra,dec)
    args += fits_path
    
    txt = os.popen(args)
    
    return txt
    

def set_dir_wcs(direc,ra,dec,pix_scale,radus,multi_pros=True):
    '''Fixes a whole directory of fits files.

    Assumes all files are located close to each other, so will use pevous info for
    next fit. '''
    pass
