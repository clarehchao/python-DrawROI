from __future__ import division
import numpy as np
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import math
from scipy.optimize import leastsq,fmin

def FindCofMass3D(vol,fmax):
    """
        determine the center of mass for a given 3D volume
        i.e.X_center = sum(ix[...]*vol[...])/sum(vol)
        same for Y_center and Z_center
    """
    vmax = np.max(vol)
    iz,iy,ix = np.where(vol > fmax*vmax)
    wvol = vol[vol > fmax*vmax]
    
    sumwvol = sum(wvol)
    X_center = round(sum(ix*wvol)/sumwvol)
    Y_center = round(sum(iy*wvol)/sumwvol)
    Z_center = round(sum(iz*wvol)/sumwvol)
    
    return X_center,Y_center,Z_center

def FindCofMass2D(im):
    """
        determine the center of mass for a given 2D image
        i.e.X_center = sum(ix[...]*vol[...])/sum(vol)
        same for Y_center
    """
    iy,ix = np.where(im > 0)
    wim = im[im>0]

    sumwim = sum(wim)
    X_center = round(sum(ix*wim)/sumwim)
    Y_center = round(sum(iy*wim)/sumwim)
    
    return X_center,Y_center


def Get2DPlane(vol,idir,indx,isplot=False):
    """
        vol: 3D array
        idir: the direction of the image plane
        index: the index of the image plane
        im: return the 2D image in a desired plane
    """
    if idir == 0:
        im = vol[indx,:,:]
    elif idir == 1:
        im = vol[:,indx,:]
    else:
        im = vol[:,:,indx]

    if isplot:
        # object oriented interface of matplotlib
        fig = plt.figure()
        ax = fig.add_subplot(111)
        # specificy the min and max of image display with vmin, vmax
        ax.imshow(im,cmap=plt.cm.gray,vmin=0,vmax=10,interpolation='nearest')
        #fig.savefig(fname)

    return im

def ShowLineProfile(im,idir,isplot=False):
    """
        Plot a line profile through a given image dimension at indx
    """
    xdata = np.arange(im.shape[idir])
    ym,xm = np.unravel_index(im.argmax(),im.shape)
    
    demoim = np.copy(im)  # have to make a copy or else it will change the array that it's trying to copy
    if idir == 0:
        ydata = im[:,xm]
        demoim[:,xm] = 0.
    else:
        ydata = im[ym,:]
        demoim[ym,:] = 0.

    if isplot:
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
        ax.plot(xdata,ydata,'ro')
        plt.show()
    

    return xdata, ydata,demoim


def GetHalfLineProfile(im,idir,indx,isplot=False):
    """
        Get the half-line profile through a given image dimension at indx
    """
    xdata = np.arange(im.shape[idir])
    if idir == 0:
        ydata = im[:,indx]
    else:
        ydata = im[indx,:]

    # get the profile that is to the right of the max value

    xthresh = np.argmax(ydata)
    ydata_new = ydata[xthresh:]
    #tmp = ydata[xthresh:]
    #ydata_new = np.hstack((tmp[-2:0:-1],tmp))
    xdata_new = np.arange(len(ydata_new))
    if isplot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xdata_new,ydata_new,'ro')
        plt.show()
    return xdata_new, ydata_new

def GetR2(data,fitdata):
    fres = np.sum((data-fitdata)**2)
    ftot = np.sum((data-data.mean())**2)
    R2 = 1 - fres/ftot
    return R2

def GaussFit(xdata,ydata,pixdim,isplot=False):
    # clean up the data first
    A = ydata.max()
    #ydata[ydata < 0.] = 0.0
    nydata = ydata/A
    # for boolean logica operation (element wise) do: eh = np.all([(cond1),(cond2)],axis=0)
    
    # Fit the data to a Gaussian function
    bigx = xdata
    x0 = np.sum(bigx*nydata)/np.sum(nydata)
    sigma = math.sqrt(abs(np.sum((bigx-x0)**2*nydata)/np.sum(nydata)))
    gfunc = lambda x: A*np.exp(-(x-x0)**2/(2*sigma**2))
    fitydata = gfunc(xdata)
    
    # get the r2 of the fit and fwhm
    r2 = GetR2(ydata,fitydata)
    fwhm = 2*math.sqrt(2*math.log(2.))*sigma*pixdim
    
    # plot the data
    if isplot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xdata,ydata,'ro',label='Original data')
        xfit = np.linspace(xdata[0],xdata[-1],500)
        ax.plot(xfit,gfunc(xfit),label='Fitted data')
        plt.show()

    return A,x0,sigma,r2,fwhm

def GaussFit_lsq(xdata,ydata,pixdim,isplot=False):
    """
        leastsq doesn't like matrices, be sure to convert all data into an array before running with leastsq
        or else: error message [ValueError: object too deep for desired array]
    """
    # normalize the input
    ymax = np.amax(ydata)
    nydata = ydata/ymax
    fitfunc = lambda p,x: p[0]*np.exp(-(x-p[1])**2/(2*p[2]**2))  # the fit function
    errfunc = lambda p,x,y: fitfunc(p,x)-y  # difference to the fit function
    
    A_0 = np.amax(nydata)
    x0_0 = np.sum(xdata*nydata)/np.sum(nydata)
    sigma_0 = math.sqrt(abs(np.sum((xdata-x0_0)**2*nydata)/np.sum(nydata)))
    p0 = [A_0,x0_0,sigma_0]
    p1,success = leastsq(errfunc,p0[:],args=(xdata,nydata))
    fitydata = ymax*fitfunc(p1,xdata)  # regularize the fitted result
    A,x0,sigma = p1
    
    # get the r2 of the fit and fwhm
    r2 = GetR2(ydata,fitydata)
    fwhm = 2*math.sqrt(2*math.log(2.))*abs(p1[2])*pixdim
    
    xfit = np.linspace(xdata[0],xdata[-1],500)
    yfit = ymax*fitfunc(p1,xfit)
    if isplot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(xdata,ydata,'ro',label='Original data')
        ax.plot(xfit,yfit,label='Fitted data')
        plt.show()
    
    return xfit,yfit,A,x0,sigma,r2,fwhm

def FitBiExpo(xdata,ydata,pixdim,isplot=False):
    # normalize the input
    ymax = np.amax(ydata)
    nydata = ydata/ymax
    fitfunc = lambda p,x: p[0]*np.exp(-p[1]*x) + p[2]*np.exp(-p[3]*x)  # the fit function
    errfunc = lambda p,x,y: fitfunc(p,x)-y  # distance to the fit function
    p0 = [0.,0.,1.0,0.]
    p1,success = leastsq(errfunc,p0[:],args=(xdata,nydata))
    fitydata = ymax*fitfunc(p1,xdata)  # regularize the fitted result
    r2 = GetR2(ydata,fitydata)
    fwhm = GetBiExpoFWM(p1,0.5)*pixdim
    fwtm = GetBiExpoFWM(p1,0.1)*pixdim
    
    if isplot:
        fig = plt.figure()
        ax = fig.add_subplot(111)
        xfit = np.linspace(xdata[0],xdata[-1],500)
        yfit = ymax*fitfunc(p1,xfit)
        ax.plot(xdata,ydata,'ro',label='Original data')
        ax.plot(xfit,yfit,label='Fitted data')
        plt.show()
    
    return p1,r2,fwhm,fwtm

def Objfunc_BiExpo(xinput,coeff,pscale):
    a0,k0,a1,k1= coeff
    fitfunc = lambda p,x: p[0]*np.exp(-p[1]*x) + p[2]*np.exp(-p[3]*x)  # the fit function
    fmax = fitfunc(coeff,0.000001)
    f = fitfunc(coeff,xinput)
    err = abs(f - pscale*fmax)
    return err

def GetBiExpoFWM(coeff,pscale):
    """
        Determine the full-width [] maximum of a Bi-Gaussian function with coeff
        full-width HALF maximum: pscale = 0.5
        full-width TENTH maximum: pscale = 0.1
        """
    x0 = 0.0
    x_sol = fmin(Objfunc_BiExpo,x0,args=(coeff,pscale),xtol=0.00000001)
    return x_sol








