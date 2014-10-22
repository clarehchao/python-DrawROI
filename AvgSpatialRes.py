#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from ImVolTool import xyz2vol as xv
from ImVolTool import ImVolFit as ivfit
import Annotate as ant
import pandas as ps
import re
#import Tkinter, tkFileDialog # user-select directory diaglog

"""
Author: Shih-ying (Clare) Huang
Date: 10/10/2014
Note:
 **** The correct way to compute FWHM of a Gaussian ***
- Fit the Normlized ydata (max of ydata = 1) with a set of xdata (0,1,2,...etc.)
- compute the fwhm from the fit result and scale it by the appropriate pixel size
To fix:
- NEED TO FIX THE INCONSISTENCY IN PARSING THE FILE DIRECTORY WITH DIFFERENT PATTERN
- 10/14/2014: generalize the string parsing for fdir to save .csv files, but when compiling dicom images, the filename parsing still remain case specific,
              e.g. xxxxxxIxxx.DCM (scanner-specific recon) vs. xxxxxslxxxxx.dcm (GE toolbox recon from Jaewon)
"""

def GetDcmVol(fdir,isscrecon=True):
    # get the dicom images into a volumn
    dxyz,nxyz,vol = xv.Dicom2Vol(fdir,isscrecon)
    
    # find the center of mass
    xc,yc,zc = ivfit.FindCofMass3D(vol,0.9)
    
    return xc,yc,zc,dxyz,nxyz,vol


def main():
    fdir = sys.argv[1]
    imdir = int(sys.argv[2])
    Nslice = int(sys.argv[3])
    soffset = int(sys.argv[4])
    isscrecon = int(sys.argv[5])
    
    # Get a DICOM volume
    if isscrecon == 0:
        xc,yc,zc,dxyz,nxyz,vol = GetDcmVol(fdir,False)
    else:
        xc,yc,zc,dxyz,nxyz,vol = GetDcmVol(fdir)
    print xc,yc,zc,nxyz
    
    # Get an image in a given direction and show the image
    if imdir == 2:
        imindx = xc-10
    elif imdir == 0:
        imindx = zc
    else:
        imindx = yc

    theIM = ivfit.Get2DPlane(vol,imdir,imindx+soffset)
    plt.imshow(theIM,cmap='hot',interpolation='bicubic')
    print theIM.shape
    
    # Select a ROI in the image
    a = ant.Annotate()
    plt.show()
    
    # plot the ROI image
    rx0,ry0,rx1,ry1=[int(val) for val in [a.x0,a.y0,a.x1,a.y1]]
    theROI = theIM[ry0:ry1,rx0:rx1]
    plt.imshow(theROI,cmap='hot',interpolation='bicubic')

    # get the correct pixel dimension
    thedxyz = np.take(dxyz,[2,1,0])
    theiax = np.arange(len(thedxyz))
    oh = theiax[theiax != imdir]
    thedxy = np.take(thedxyz,oh)
    print thedxy
    statscombo1 = np.empty((Nslice+2,5))
    statscombo2 = np.empty((Nslice+2,5))

    # determine the slice # to look at
    halfNslice = int(math.floor(Nslice/2.))
    imindx0 = imindx+soffset
    if Nslice % 2. == 0:
        lnslice = np.arange(imindx0-halfNslice,imindx0+halfNslice,1)
    else:
        lnslice = np.arange(imindx0-halfNslice,imindx0+halfNslice+1,1)

    for ii in range(len(lnslice)):
        if ii == 0:
            ioi = theROI
        else:
            theIM = ivfit.Get2DPlane(vol,imdir,lnslice[ii])
            ioi = theIM[ry0:ry1,rx0:rx1]

        for ldir in range(2):
            # Get the line spread function of the ROI image
            xdata,ydata,demoim = ivfit.ShowLineProfile(ioi,ldir)
            plt.imshow(demoim,cmap='hot',interpolation='bicubic')
            xfit,yfit,A,x0,sigma,r2,fwhm = ivfit.GaussFit_lsq(xdata,ydata,thedxy[ldir],isplot=True)
            print A,x0,sigma,'Gaussian fit r2 = {}, FWHM = {} mm'.format(r2,fwhm)
            if ldir == 0:
                statscombo2[ii,:] = [A,x0,sigma,r2,fwhm]
            else:
                statscombo1[ii,:] = [A,x0,sigma,r2,fwhm]
    # append the mean and stdev
    statscombo1[-2] = np.mean(statscombo1[:-2],axis=0,dtype=np.float64)
    statscombo1[-1] = np.std(statscombo1[:-2],axis=0,dtype=np.float64)
    statscombo2[-2] = np.mean(statscombo2[:-2],axis=0,dtype=np.float64)
    statscombo2[-1] = np.std(statscombo2[:-2],axis=0,dtype=np.float64)
    statsall = np.hstack((statscombo1,statscombo2))
    
    # get all the info into a data frame
    rowname = [str(i) for i in lnslice] + ['mean','stdev']
    #print rowname
    if imdir == 0:
        colname = ['A','x0','sigma','R2','X-dir FWHM (mm)','A','x0','sigma','R2','Y-dir FWHM (mm)']
    elif imdir == 2:
        colname = ['A','x0','sigma','R2','Y-dir FWHM (mm)','A','x0','sigma','R2','Z-dir FWHM (mm)']
    df = ps.DataFrame(statsall,index=rowname,columns=colname)
    wfdir = '/Users/shuang/Documents/Proj_PETMR/Data/Measurement_SpatRes'
    froot = '/Users/shuang/Documents/Proj_PETMR/IM'
    tmp = re.findall(r'{}/(.+)*'.format(froot),fdir)
    if tmp:  # make sure the fdir string can be parsed
        tag = tmp[0].replace('/','_')
        file = '{}/SpaRes_{}.csv'.format(wfdir,tag)
        df.to_csv(file)

if __name__ == '__main__':
    main()