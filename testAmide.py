#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import math
import sys
from ImVolTool import xyz2vol as xv
from ImVolTool import ImVolFit as ivfit

fname = '/Users/shuang/Desktop/PMMA AX F18_profile_{Box_AX_F18_PETMR_FFBP}_yz.tsv'
data = np.loadtxt(fname,skiprows=1)
xdata = data[:,0]
ydata = data[:,1]
#xdata,ydata = np.hsplit(data,data.shape[1])
xdata_new = np.arange(len(ydata))
#xdata_new = xdata_new[:,np.newaxis]
print xdata_new.shape,ydata.shape
print xdata_new,ydata

pixdim = 2.78
A,x0,sigma,r2,fwhm = ivfit.GaussFit_lsq(xdata,ydata,pixdim,isplot=True)
print A,x0,sigma,'Gaussian fit r2 = {}, FWHM = {} mm'.format(r2,fwhm)
A,x0,sigma,r2,fwhm = ivfit.GaussFit_lsq(xdata_new,ydata,pixdim,isplot=True)
print A,x0,sigma,'Gaussian fit r2 = {}, FWHM = {} mm'.format(r2,fwhm)
