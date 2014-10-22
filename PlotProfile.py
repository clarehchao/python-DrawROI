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

def GetDcmVol(fdir,isscrecon=True):
    # get the dicom images into a volumn
    dxyz,nxyz,vol = xv.Dicom2Vol(fdir,isscrecon)
    
    # find the center of mass
    xc,yc,zc = ivfit.FindCofMass3D(vol,0.9)
    
    return xc,yc,zc,dxyz,nxyz,vol


def main():
    fdir = sys.argv[1]
    imdir = int(sys.argv[2])
    soffset = int(sys.argv[3])
    isscrecon = int(sys.argv[4])
    
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

    # get the line profile and fit a Gaussian via leastsq
    #wfdir = '/Users/shuang/Documents/Proj_PETMR/Data/Measurement_SpatRes'
    wfdir = '/Users/shuang/Documents/UCSF/Conference/IEEE2014/Poster'

    # get file tag
    froot = '/Users/shuang/Documents/Proj_PETMR/IM'
    tmp = re.findall(r'{}/(.+)*'.format(froot),fdir)
    if tmp:  # make sure the fdir string can be parsed
        tag = tmp[0].replace('/','_')

    # save the original ROI image
    fig = plt.figure(1,figsize=(10,10))
    ax = fig.add_subplot(111)
    plt.imshow(theROI,cmap='hot',interpolation='bicubic')
    plt.xticks([])
    plt.yticks([])
    imname = '{}/theROI_{}.pdf'.format(wfdir,tag)
    print imname
    plt.savefig(imname)

    for ldir in range(2):
        # Get the line spread function of the ROI image
        xdata,ydata,demoim = ivfit.ShowLineProfile(theROI,ldir)
        nydata = ydata/np.amax(ydata)
        fig = plt.figure(2,figsize=(8,8))
        ax = fig.add_subplot(111)
        plt.imshow(demoim,cmap='hot',interpolation='bicubic')
        imname = '{}/demoim_{}_imdir{:d}_ldir{:d}.pdf'.format(wfdir,tag,imdir,ldir)
        print imname
        plt.savefig(imname)
        xfit,yfit,A,x0,sigma,r2,fwhm = ivfit.GaussFit_lsq(xdata,nydata,thedxy[ldir])
        print A,x0,sigma,'Gaussian fit r2 = {}, FWHM = {} mm'.format(r2,fwhm)
        print xfit,yfit
        
        #wfname1 = '{}/LSP_{}_imdir{:d}_ldir{:d}.txt'.format(wfdir,tag,imdir,ldir)
        #np.savetxt(wfname1,(xdata[:,np.newaxis],ydata[:,np.newaxis]),delimiter='\t')
        #wfname2 = '{}/fitLSP_{}_imdir{:d}_ldir{:d}.txt'.format(wfdir,tag,imdir,ldir)
        #np.savetxt(wfname2,(xfit,yfit),delimiter='\t')
        # plot the profile
        fig = plt.figure(3,figsize=(11,8))
        ax = fig.add_subplot(111)
        plt.rc('lines', linewidth=3)
        plt.rc('axes',linewidth=3)
        plt.tick_params(labelsize=20)
        pixdim = 10*dxyz[0]
        p1 = plt.plot(pixdim*xdata,nydata,linestyle='None',color='b',marker='o',markersize=6.0,label='Original Data')
        p2 = plt.plot(pixdim*xfit,yfit,color='r',marker='None',label='Fitted Data')
        axes = plt.gca()
        axes.set_ylim([0,np.amax(yfit)])
        plt.xlabel('Distance (mm)',fontsize=30)
        plt.ylabel('Relative intensity',fontsize=30)
        plt.legend(fontsize=30,loc='best')
            
        figname = '{}/LSP_{}_imdir{:d}_ldir{:d}.pdf'.format(wfdir,tag,imdir,ldir)
        print figname
        plt.savefig(figname)
        plt.show()

if __name__ == '__main__':
    main()