# does photometry using ds9 as gui interface
#Thuso Simon March 18 2013

import numpy as nu
import ds9
import pylab as lab
import tkFileDialog as tk
import pyfits as fits
import gui_reduice as gui

class Phot(object):
    '''Class for photmetry of a fits image. Has  functions to help determine
    the aperatre radius for each star. Returns region data from ds9 as numpy 
    array.'''
    
    def __init__(self,file_path=None):
        #if no path use gui to select image
        if file_path is None:
            file_path = tk.askopenfilename(filetypes=[('Fits Files',
                                                       ('*.fit','*.fits'))])
        self.data = fits.open(file_path)
        self.file_path = file_path
        self.XYinfo = []
        self.photo = []
        #start ds9
        self.ds9_class = ds9.ds9()
        self.ds9_class.set_pyfits(self.data)
        #set viewing and other options to ds9
        self.ds9_class.set('scale log')
        self.ds9_class.set('magnifier to fit')
        #make limits
        lower = self.data[0].data.mean() - self.data[0].data.std()
        upper = self.data[0].data.mean() + 3*self.data[0].data.std()
        self.ds9_class.set('scale limits %f %f'%(lower,upper))
        #make catolog of stars
        self._image = gui.ImgCat(self.file_path)
        self._image.makecat(verbose=False)
        self._image.makestarlist(verbose=False)
        for i in self._image.starlist:
            self.XYinfo.append([i.x, i.y])
            self.photo.append([i.flux, i.fwhm])
            #plot cat to ds9
            self.ds9_class.set('regions', 'physical;circle(%f %f %.0f)'%(
                    i.x, i.y,i.fwhm*3))
        
    def update_region(self):
        '''Updates python regions from ds9'''
        raw_region = self.ds9_class.get('regions').splitlines()
        self.XYinfo = []
        self.photo = []
        for i in raw_region:
            if i.startswith('circle'):
                x,y,fwhm = eval(i.strip('circle'))
            elif i.startswith('ellipse'):
                x,y,fwhm =  eval(i.strip('ellipse'))
            self.XYinfo.append([x + 0,y+0])
            self.photo.append([-1, fwhm + 0])
            
    
    def save_region(self):
        pass

    def replot_region(self,xyinfo=None):
        '''replots regions on ds9'''
        #delet current region
        self.ds9_class.set('regions delete all')
        #plot
        if xylist is not None:
            self.XYinfo = xyinfo
            self.photo = [-1,-1] *len(xyinfo)
        for i in self.XYinfo:
            self.ds9_class.set('regions', 'physical;circle(%f %f %.0f)'%(
                    i[0], i[1],5*3))
        
    def curve_of_growth(self,xylist=None):
        '''Makes a curve of growth for current region. Helps find radius to use
        for photometry. Can either import own list of star possitions or use
        list made from current instance.'''
        if xylist is not None:
            self.XYinfo = xylist
            self.photo = [-1,-1] *len(xylist)

        #iterate through star list and create curve of growth plot
        rad = 20
        lab.ion()
        fig = lab.figure()
        plt = fig.add_subplot(111)
        plt.set_title('Curve of Growth')
        plt.set_ylabel('Counts')
        plt.set_xlabel('Radius (pixels)')
        x = nu.arange(1,rad)
        y = []
        bkrnd = []
        data = self.data[0].data
        for i in range(len(self.XYinfo)):
            totsum = 0
            for j in x:
                tempsum = data[i[0]
                totsum += data[i[0]

    def _get_20_brightest(self, N=20):
        '''plots and gets N brightest stars in image'''
        bright_list = nu.zeros((N,3))
        temp = []
        raw_region = self._image.starlist
        for i in raw_region:
            #get region info
            x,y,fwhm = i.x, i.y, i.flux
            #put in bright list
            if fwhm > bright_list[-0,-1]:
                bright_list[-0] = nu.copy([x,y,fwhm])
                bright_list = bright_list[bright_list[:,-1].argsort()]
        #plot new bright_list region file
        #plot brightest
        self.replot_region(bright_list)
        
        return bright_list


    def photometry(self,xylist):
        '''returns photmetry from stars'''
        pass

    def light_curve(self,files,xylist):
        '''Makes light curves for different stars'''
        pass

    def make_cat(self):
        '''Make catalog of all stars in an image.'''
        pass
