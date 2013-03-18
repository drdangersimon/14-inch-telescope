# does photometry using ds9 as gui interface
#Thuso Simon March 18 2013

import numpy as nu
import ds9
import pylab as lab
import tkFileDialog as tk


class Phot:
    '''Class for photmetry of a fits image. Has  functions to help determine
    the aperatre radius for each star. Returns region data from ds9 as numpy 
    array.'''
    
    def __init__(self,file_path=None):
        #if no path use gui to select image
        if file_path is None:
            file_path = tk.askopenfilename('*.fits' or '*.fit')
