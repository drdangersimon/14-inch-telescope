#set wcs coordinates of fits image
#Thuso Simon 8 Mar 2013

import os,sys
import pywcs
import subprocess as sub
import numpy as nu
import pyfits as fits
import multiprocessing as M
from time import time


dir_wcs = '/home/thuso/14_inch/git_repo/code/wcstools-3.8.7/'
sextractor_cmd = 'sex'

def set_wcs_headers(fits_path,ra,dec,pix_scale,xy_center=None):

    #uses sextractor to find stars
    #callibrates from there
    #run sextractor on image useing 14inch config

    #run imwcs to find coordinates
    #look at match ratio > .8 try 3 feild around input ra and dec and all 
    #rotations
    #copy to temp image
    os.popen('cp %s %i.fit'%(fits_path,os.getpid()))
    new_fits_path = '%i.fit'%os.getpid()
    #make field
    max_axis = max(fits.getheader(fits_path)['NAXIS1'],
                   fits.getheader(fits_path)['NAXIS2'])
    #convert to degrees
    field = max_axis * pix_scale /3600./2
    ra_fields = nu.linspace(ra - field, ra + field ,5)
    dec_fields = nu.linspace(dec - field, dec + field,5)
    rotaion = nu.arange(0,90,10)
    tot_iter = len(ra_fields)*len(dec_fields)*len(rotaion)*2
    i = 0
    stars_matched = 0
    tot_stars =1
    imwcscmd = dir_wcs + 'bin/imwcs'
    imrotcmd = dir_wcs + 'bin/imrot'
    best = [0,-9999,-9999, 400,True] #nstars,ra,dec,rot
    for ROT in rotaion:
        for xr_mirror in ['-m','-l']:
            #make new temp image
            os.popen('cp %s %i.fit'%(fits_path,os.getpid()))
            #rotatae or mirror
            new_fits = sub.Popen([imrotcmd,xr_mirror,'-r',str(ROT), 
                             new_fits_path],stdout=sub.PIPE).communicate()[0]
            #extract sources to 14inch.sex
            txt = sub.Popen([sextractor_cmd,new_fits,'-c', 
                             '14inch.sex'],stderr=sub.PIPE).communicate()
            #sort 100 brightest stars and put in pid.cat file 3 coulmn is flux
            temp_cat = nu.loadtxt('14inch.cat')
            temp_cat = temp_cat[temp_cat[:,2].argsort()][-100:]
            nu.savetxt('%i.cat'%os.getpid(),temp_cat[-1:0:-1],fmt='%.3f',delimiter='    ')
            #move around ra and dec fields
            for RA in ra_fields:
                for DEC in  dec_fields:
                    ra_dec = str(RA)+' '+str(DEC)
                    arg = imwcscmd +' -j '+ra_dec 
                    arg += ' -c '+' ucac2 ' #+ ' -d ' + ' %i.cat '%os.getpid()
                    if nu.any(xy_center):
                        arg += ' -x %.2f %.2f'%(xy_center[0],xy_center[1])
                    arg += ' -t ' +' 20 '+' -s '+ ' 0.5 ' #+ ' -m '+' 14 '
                    arg += ' -q ' +' ips ' + ' -p ' +str(pix_scale)
                    arg += ' -v '+new_fits
                    txt = sub.Popen(arg,stderr=sub.PIPE,stdout=sub.PIPE,
                                    shell=True).communicate()
                    i +=1.
                    if txt[0]:
                        temp = txt[0].split()
                        try:
                            stars_matched = int(temp[temp.index('nmatch=') +1])
                            tot_stars = int(temp[temp.index('nstars=') +1])
                        except:
                            continue
                    #check if best
                        if i%10 == 1:
                            print 'nmatch %i nstars %i %2.1f percent done'%(stars_matched,tot_stars,i/tot_iter*100)
                        sys.stdout.flush()
                    
                        '''if (stars_matched/50. > .5 and 
                        len(M.active_children()) <= M.cpu_count()):
                         mult_know.append(M.Process(target=set_wcs_headers,
                                                    args=(fits_path,RA,DEC,pix_scale)))
                         mult_know[-1].start()
                         print 'starting new processes' '''
                    if best[0] < stars_matched:
                        best = [stars_matched + 0,RA +0, 
                                DEC + 0, ROT + 0, xr_mirror+'']
                        print best
                    elif best[0] == stars_matched:
                        best.append([stars_matched + 0,RA +0, 
                                DEC + 0, ROT + 0, xr_mirror+''])
                        print best[-1]
                    if stars_matched/float(tot_stars) > .8:
                        break
    #write best out
    new_fits = sub.Popen([imrotcmd,best[4],'-r',str(best[3]), 
                          new_fits_path],stdout=sub.PIPE).communicate()[0]
    txt = sub.Popen([sextractor_cmd,new_fits,'-c', 
                             '14inch.sex'],stderr=sub.PIPE).communicate()
    #sort 100 brightest stars and put in pid.cat file 3 coulmn is flux
    temp_cat = nu.loadtxt('14inch.cat')
    temp_cat = temp_cat[temp_cat[:,2].argsort()][-100:]
    nu.savetxt('%i.cat'%os.getpid(),temp_cat[-1:0:-1],fmt='%.3f',delimiter='    ')
    ra_dec = str(best[1])+' '+str(best[2])
    ra_dec = str(RA)+' '+str(DEC)
    arg = imwcscmd +' -j '+ra_dec 
    arg += ' -c '+' ucac2 ' #+ ' -d ' + ' %i.cat '%os.getpid()
    if nu.any(xy_center):
        arg += ' -x %.2f %.2f'%(xy_center[0],xy_center[1])
    arg += ' -t ' +' 20 '+' -s '+ ' 0.5 ' #+ ' -m '+' 15 '
    arg += ' -q ' +' ips ' + ' -p ' +str(pix_scale)
    arg += ' -vo best_' + fits_path + ' '+ new_fits
    txt = sub.Popen(arg,stderr=sub.PIPE,stdout=sub.PIPE,
                                shell=True).communicate()
    os.popen('rm -f %i*.fit'%os.getpid())
    #return best fit position
    return best
    
def match_mcmc(fits_path,ra,dec,pix_scale,time_limit=600):
    '''uses mcmc to find best position. Uses multiprocesssing
    Time limit is in seconds'''
    #initalize params
    axis = [fits.getheader(fits_path)['NAXIS1'],
            fits.getheader(fits_path)['NAXIS2']]
    #everything is in degrees except pix_scale
    active_param = nu.array([ra,dec,0.,0]) #[ra,dec,rot,xmirror]
    sigma = nu.array([axis[0]*pix_scale*.1,axis[1]*pix_scale*.1,36.,.2])
    txt = sub.Popen([sextractor_cmd,fits_path,'-c', '14inch.sex'],stderr=sub.PIPE).communicate()
    if txt[0]:
        #errors
        print 'Error in input'
        print txt[1]
        raise SystemError
    #output is to 14inch.cat
    #sort 100 brightest stars and put in pid.cat file 3 coulmn is flux
    temp_cat = nu.loadtxt('14inch.cat')
    nu.savetxt('%i.cat'%os.getpid(),temp_cat[temp_cat[:,2].argsort()][-100:],fmt='%.3f',delimiter='    ')
    imwcscmd = dir_wcs + 'bin/imwcs'
    temp = run_imwcs(imwcscmd,fits_path,active_param,pix_scale)
    chi = nu.array([temp[0]/float(temp[1]),0.])
    chi_best = temp[0]/float(temp[1])
    param_best = nu.copy(active_param)
    param_old = nu.copy(active_param)
    T = time()
    while time() - T <= time_limit:
        active_param = nu.random.rand(len(active_param))*sigma + active_param
        #check param values [ra,dec,rot,xmirror]
        if active_param[0] > 360:
            active_param[0] -= 360
        elif active_param[0] < 0:
            active_param[0] += 360
        #dec
        if abs(active_param[1]) > 90:
            active_param[1] = nu.sign(active_param[1])*90
        #rot
        if active_param[2] > 360:
            active_param[2] -= 360
        elif active_param[2] < 0:
            active_param[2] += 360
        #mirror
        if active_param[3] > 0:
            active_param[3] = 1
        else:
            active_param[3] = 0
        #run imwcs
        temp = run_imwcs(imwcscmd,fits_path,active_param,pix_scale)
        chi[1] = temp[0]/float(temp[1])
        if chi[1] > chi[0]:
            #accept
            chi[0] = nu.copy(chi[1])
            param_old = active_param.copy()
            #check if best
            if chi_best < chi[1]:
                chi_best = nu.copy(chi[1])
                param_best = active_param.copy()
                print 'best match found ratio %0.2f at ra,dec,rot %3.1f,%2.1f,%2.0f'%(chi_best,param_best[0],param_best[1],param_best[2])
        else:
            active_param = param_old.copy()
        if chi_best > .8:
            break

    return param_best, chi_best


def run_imwcs(imwcscmd,fits_path,param,pix_scale,save=False):
    '''create imwcs aguments'''
    ra,dec,rot,mirror = param
    ra_dec = str(ra)+' '+str(dec)
    arg = imwcscmd + ' -a '+ str(rot)+' -j '+ra_dec
    arg += ' -c '+' ucac2 ' + ' -d ' + ' %i.cat '%os.getpid()
    arg += ' -t ' +' 30 '+' -s '+ ' 0.5 ' + ' -m '+' 15 '
    arg += ' -q ' +' ips ' + ' -p ' +str(pix_scale)
    if mirror:
        arg += ' -l '
    if save:
        arg += ' -o ' + ' %i.fit '%os.getpid()
    arg += ' -v ' + fits_path
    txt = sub.Popen(arg,stderr=sub.PIPE,stdout=sub.PIPE,
                    shell=True).communicate()
    if txt[0]:
        temp = txt[0].split()
        try:
            stars_matched = int(temp[temp.index('nmatch=') +1])
            tot_stars = int(temp[temp.index('nstars=') +1])
        except:
            return 0,50
    else:
        return 0,50
    return stars_matched,tot_stars


def coordfind(fits_file, raguess,decguess,area):
    #uses sextractor to find center of feild
    
    #find catolog dir

    #set ra and dec headers if none in fits header

    #find sources in image

    #take only 100 brightest starts and move to star list file

    #use imwcs with star list to match start to cat

    #if match is <.8 then try again with ra and dec offset

    #return found ra and dec

    pass
