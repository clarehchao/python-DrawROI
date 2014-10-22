#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import matplotlib.pyplot as plt
import Tkinter, tkFileDialog # user-select directory diaglog
import numpy as np
from pylab import *
import time
import math
from ImVolTool import xyz2vol as xv
import Annotate as ant

def FindCofMass(im,isplot=False):
    """
    determine the center of mass for a given 2D image
    i.e.X_center = sum(ix[...]*vol[...])/sum(vol)
        same for Y_center
    """
    check = np.zeros(im.shape)
    iy,ix = np.where(im > 0)
    wim = im[im>0]
    wimmax = np.max(wim)
    check[im>0.8*wimmax] = 1
    
    if(isplot):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        imshow(check,cmap='gray')
        plt.show()
    
    #print wim.shape,np.max(wim)
    sumwim = sum(wim)
    X_center = round(sum(ix*wim)/sumwim)
    Y_center = round(sum(iy*wim)/sumwim)
    
    return X_center,Y_center
    
def ShowLineProfile(im,idir,indx,pixdim,isplot=False):
    """
    Plot a line profile through a given image dimension at indx1, indx2
    """
    xdata = np.arange(0,pixdim*len(ydata),pixdim)
    if idir == 0:
        ydata = im[:,indx]
    else:
        ydata = im[indx,:]

    if isplot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xdata,ydata,'ro')
        plt.show()
        #plt.plot(xdata,ydata,'ro')
    return xdata, ydata
    
    
def GetR2(data,fitdata):
    fres = np.sum((data-fitdata)**2)
    ftot = np.sum((data-data.mean())**2)
    R2 = 1 - fres/ftot
    return R2
    
def gauss(x,*p):
    A,mu,sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def PlotGaussFit(xdata,ydata,f0):
    coeff,var_matrix = curve_fit(gauss,xdata,ydata,p0=f0,maxfev=1000000)
    ydata_fit = gauss(xdata,*coeff)
    R2 = GetR2(ydata,ydata_fit)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(xdata,ydata,'ro',label='Original data')
    ax.plot(xdata,ydata_fit,label='Fitted data')
    plt.show()
    
    # load the output data
    outdata = np.zeros(len(coeff)+1)
    outdata[:-1] = coeff
    outdata[-1] = R2
    return outdata
    
def GaussFit(xdata,ydata,isplot=False):
    # clean up the data first
    A = ydata.max()
    ydata[ydata < 0.2*A] = 0.
    # for boolean logica operation (element wise) do: eh = np.all([(cond1),(cond2)],axis=0)
    
    # Fit the data to a Gaussian function
    bigx = xdata
    x0 = sum(bigx*ydata)/sum(ydata)
    sigma = sqrt(abs(sum((bigx-x0)**2*ydata)/sum(ydata)))
    A = ydata.max()
    fitydata = lambda x: A*exp(-(x-x0)**2/(2*sigma**2))
    
    # get the r2 of the fit and fwhm
    r2 = GetR2(ydata,fitydata)
    fwhm = 2*sqrt(2*log(2.))*sigma
    
    # plot the data
    if isplot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xdata,ydata,'ro',label='Original data')
        ax.plot(xdata,fitdata(xdata),label='Fitted data')
        plt.show()
    return A,x0,sigma,r2,fwhm

def GetDicomVol(fdir):
    # get the dicom images into a volumn
    dxyz,nxyz,vol = xv.Dicom2Vol(fdir)
    
    # find the center of mass
    iz,iy,ix = np.where(vol > 0.9*volmax)
    sumwim = sum(vol[vol > 0.9*volmax])
    X_center = round(sum(ix*wim)/sumwim)
    Y_center = round(sum(iy*wim)/sumwim)
    Z_center = round(sum(iz*wim)/sumwim)
    
    # find the file with with slice = Z_center
    #thefname = '%s/%sI%d.DCM' % (fdir,t3,int(Z_center))
    #theim0 = vol[int(Z_center)]
    #theim1 = np.amax(vol,axis=2)
    theim0 = vol[:,:,int(X_center+10)]
    theim1 = np.amax(vol,axis=2)
    #theim1 = vol[:,68,:]
    #print theim0.shape,theim1.shape
    #fig=plt.figure()
    #ax=fig.add_subplot(111)
    #ax.imshow(theim1,cmap='gray',interpolation='nearest')
    #ax.set_aspect('equal')
    #plt.show()
    
    return X_center,Y_center,Z_center,dxyz,nxyz,theim0,theim1,vol
    

def main():
    idir = int(sys.argv[1])
    #root = Tkinter.Tk()
    #fdir = tkFileDialog.askdirectory(parent=root,initialdir='/Users/shuang/Documents/IM',title='Please select a directory')
    #print fdir
    #fdir = '/Users/shuang/Documents/IM/CBPTMR/PHANTOM/E313/402'
    fdir = '/Users/shuang/Documents/IM/CBPET2/D078140/E12092/402'
    xc,yc,zc,dxyz,nxyz,im0,im1,vol = FindDCVolCOF(fdir)
    #xc,yc,zc,imrow,imcol,pixdim,sthick,thefname,im0,im1,vol = FindDCVolCOF(fdir)
    print xc,yc,zc,vol.shape
    if idir == 0:
        theIM = im0
    else:
        theIM = im1

    #fig1 = figure()
    #ax1 = fig1.add_subplot(111)
    #ax1.imshow(theIM,cmap='hot',interpolation='bicubic')
    plt.imshow(theIM,cmap='hot',interpolation='bicubic')

    #imshow(vol[:,yc,:],cmap='hot',interpolation='bicubic')
    #show()
    
    a = ant.Annotate()
    plt.show()
    
    x0,y0,x1,y1=[int(val) for val in [a.x0,a.y0,a.x1,a.y1]]
    print x0,y0,x1,y1
    theROI = theIM[y0:y1,x0:x1]
    plt.imshow(theROI,cmap='hot',interpolation='bicubic')
    #fig2 = figure()
    #ax2 = fig2.add_subplot(111)
    #ax2.imshow(theROI,cmap='hot')
    plt.show()
    
    xc_roi,yc_roi =FindCofMass(theROI)
    print xc_roi,yc_roi
    
    # dir = 0
    xdata,ydata=ShowLineProfile(theROI,0,xc_roi,dxyz[-1],isplot=True)
    # fit a Gaussian to the line profile
    fwhm = GaussFit(xdata,ydata,isplot=True)
    print 'FWHM = ' + str(fwhm) + ' mm'


    # dir = 1
    xdata,ydata=ShowLineProfile(theROI,1,yc_roi,dxyz[0],isplot=True)
    # fit a Gaussian to the line profile
    fwhm = GaussFit(xdata,ydata,isplot=True)
    print 'FWHM = ' + str(fwhm) + ' mm'


    # pan through the z-slices
    for ii in range(3):
        tmp = vol[y0:y1,x0:x1,xc+ii]
        print tmp.shape
        xc2,yc2=FindCofMass(tmp)
        print xc2,yc2

        # dir = 0
        xdata,ydata=ShowLineProfile(tmp,0,xc2,dxyz[-1],isplot=True)
        # fit a Gaussian to the line profile
        fwhm = GaussFit(xdata,ydata,isplot=True)
        print 'FWHM = ' + str(fwhm) + ' mm'
        
        
        # dir = 1
        xdata,ydata=ShowLineProfile(tmp,1,yc2,dxyz[0],isplot=True)
        # fit a Gaussian to the line profile
        fwhm = GaussFit(xdata,ydata,isplot=True)
        print 'FWHM = ' + str(fwhm) + ' mm'

        

if __name__ == '__main__':
    main()

"""
    #from matplotlib.patches import Rectangle
    #import dicom  # dicom reader package
    #import glob   # get all the files in a directory
    
    
    thedir = fdir + '/*.DCM'
    allfiles = glob.glob(thedir)
    
    # Get file name info
    teststr = allfiles[0]
    t1 = teststr.split('.')[0]
    t2 = t1.split('/')[-1]
    t3 = t2.split('I')[0]
    testdc = dicom.read_file(teststr)
    imcol = testdc.Columns
    imrow = testdc.Rows
    pixdim = testdc.PixelSpacing
    sthick = testdc.SliceThickness
    #nslice = testdc[0x0009,0x10df].value
    nslice = len(allfiles)
    vol = np.zeros([nslice,imrow,imcol])
    thelist = range(1,nslice+1)
    for i in range(len(thelist)):
        thefname = '%s/%sI%s.DCM' % (fdir,t3,thelist[i])
        thedc = dicom.read_file(thefname)
        vol[i] = thedc.pixel_array
    print vol.shape
    """

