from __future__ import division
import numpy as np
import dicom  # dicom reader package
import glob   # get all the files in a directory
import sys
import os
import re

def IsInBound(val,low,high):
    """
    check to make sure VAL in within the range (low,high]
    """
    if len(val) == 1:   # for a single value
        if val < low:
            #print 'less than low'
            return low
        elif val >= high:
            #print 'more than high'
            return high-1
        else:
            return val
    elif len(val) > 1:  # for the case of an array
        val[val < low] = low
        val[val >= high] = high-1
        return val
    else:
        print 'empty input!'

def properRound(x):
    """
    Numpy rounds x.5 to th nearest even number (not sure why...)
    this function does the proper round e.g. for x.5, round up to the nearest integer
    Input: x is a single scalar or a numpy array
    Output: return a scalar or an array of properly rounded numbers
    """
    y = x - np.floor(x)
    if len(x) == 1:  # a scalar
        if (0 < y < 0.5):
            return np.floor(x)
        else:
            return np.ceil(x)
    else:  # an array
        output = np.empty(x.shape,dtype='int')  # make sure the output is in 'int' not 'float'
        indx = np.all([y > 0,y < 0.5],axis=0)
        notindx = np.logical_not(indx)
        output[indx] = np.floor(x[indx],out=np.empty_like(x[indx],dtype=np.int_),casting='unsafe')
        output[notindx] = np.ceil(x[notindx],out=np.empty_like(x[notindx],dtype=np.int_),casting='unsafe')
        return output
        
def GetUniqueCount(x):
    # input: x is a numpy array
    thecount = {}
    for v in x:
        if v not in thecount:
            thecount[v] = 1
        else:
            thecount[v] += 1
    return thecount
    
def SaveFlattenVol(vol,fname):
    outputType = 'float32'
    outputExtension = '.f32le'
    lefile = fname + outputExtension
    #flatArray = vol.flatten().astype(outputType)
    flatArray = np.ravel(vol).astype(outputType)
    flatArray.tofile(lefile)


def Xyz2Vol(fname,dxyz,xyz0,nxyz):
    # get the data into workable form
    data = np.loadtxt(fname,skiprows=1)
    val,wt,xmm,ymm,zmm = np.hsplit(data,data.shape[1])
    
    # get dxyz and xyz dimension
    dx,dy,dz = dxyz
    x0,y0,z0 = xyz0
    nx,ny,nz = nxyz
    ixx = IsInBound(properRound((xmm-x0)/dx),0,nx)
    iyy = IsInBound(properRound((ymm-y0)/dy),0,ny)
    izz = IsInBound(properRound((zmm-z0)/dz),0,nz)
    
    # convert coordinates into indices for 1D array operation
    xyzcoord = np.vstack((izz.T,iyy.T,ixx.T))
    indx = np.ravel_multi_index(xyzcoord,(nz,ny,nx))
    voxct = GetUniqueCount(indx)
    vol_flat = np.zeros(nx*ny*nz,dtype='uint8')
    vol_flat[voxct.keys()] = voxct.values()
    #vol_3d = vol_flat.reshape((nz,ny,x))
    return vol_flat
    
def Coord2Vol(fname,nxyz):
    data = np.loadtxt(fname)
    print 'Read the file: ',fname
    nx,ny,nz = nxyz
    
    #get the x,y,z position, first three cols of 'data'
    tmp = data[:,:-1]
    #xyz = tmp.conjugate().transpose()
    xyz = tmp.T
    xyzint = xyz.astype(int)
    val = data[:,3]
    
    #vol = np.zeros((nx,ny,nz))
    flatvol = np.zeros(nx*ny*nz)
    #convert (x,y,z) indx to flat indices
    indx = np.ravel_multi_index(xyzint,(nx,ny,nz),order='F')
    #vol.ravel()[indx] = val
    flatvol[indx] = val
    return flatvol
    
    
def SaveFlattenVol(vol,fname,ftype):
    #flatArray = vol.flatten().astype(outputType)
    flatArray = np.ravel(vol).astype(ftype)
    flatArray.tofile(fname)
    print 'saved the volume to %s as %s!' % (fname,ftype)


def VolMask_Threshold(vol,thresh=0):
    # create a vol mask based on a given threshold
    vsize = np.prod(np.array(vol.shape))
    mask = np.zeros(vsize,dtype='uint8')
    if len(thresh) == 2:
        t1,t2 = thresh
        indx = np.all([(vol >= t1),(vol < t2)],axis=0)
        mask[indx] = 1
    else:
        mask[vol.flatten() >= thresh] = 1
    return mask

def SegmentVol_ImMask(vol,maskfname,ftype,segval0=0):
    # create a segmented volume given a list of masks
    vsize = np.prod(np.array(vol.shape))
    #segvol = np.zeros(vsize,dtype=ftype)
    if len(vol.shape) > 1:
        segvol = np.ravel(vol)
    else:
        segvol = vol
    for i in range(len(maskfname)):
        ma = np.fromfile(maskfname[i],dtype=ftype)
        if re.search('Tumor',maskfname[i]):
            segvol[ma >= 1] = segval0+i
            print 'It\'s a tumor! can overwrite bone voxel!'
        elif re.search('Brain',maskfname[i]): # do not want the image mask to overwrite any 'bone' or 'air' voxel
            indx = np.all([(ma >= 1),(vol != 2),(vol != 0)],axis=0)
            segvol[indx] = segval0+i
            print 'It\'s a Brain, avoid overwriting bone and air voxels!'
        else:
            indx = np.all([(ma >= 1),(vol != 2)],axis=0)
            segvol[indx] = segval0+i
            print 'It\'s Not a brain or tumor, avoid overwriting bone voxels!'
            
        print 'anything above 1? ', np.sum(indx)
        print 'set a mask value: %s,%d' % (maskfname[i],segval0+i)
    return segvol
    
def SegmentVol_ThreshMask(vol,thresh):
    # create a segmented volume given a list of masks
    #vsize = np.prod(np.array(vol.shape))
    segvol = np.zeros(vol.shape,dtype='uint8')
    for i in range(len(thresh)-1):  # go throughe each threshold bin and assign an index of material
        t1 = thresh[i]
        t2 = thresh[i+1]
        indx = np.all([(vol >= t1),(vol < t2)],axis=0)
        #print indx.shape
        segvol[indx] = i
    #print segvol.shape
    return np.ravel(segvol),np.amax(segvol)
    
def FlattenFile2Vol(fname,ftype,nx,ny,nz):
    vol = np.fromfile(fname,dtype=ftype)
    #tmp = np.fromfile(fname,dtype=ftype)
    #vol = tmp.reshape((nz,ny,nx),order='C')
    #vol = tmp.reshape((nz,ny,nx),order='F')
    return vol
    
def Dicom2Vol(dcdir,isscrecon=True):    #instance member function
    if isscrecon:
        alldcfiles = glob.glob('%s/*.DCM' % dcdir)
        #print len(alldcfiles),alldcfiles[0]
        # find the dcfile tag: e.g. ____I10.DCM
        match = re.search(r'%s/([\w.-]+)I([\w.-]+).DCM' % dcdir,alldcfiles[0])
        print match.group()
        if match:
            dcftag = match.group(1)
            #print dcftag
        else:
            raise RuntimeError('Error: cannot get dicom file tag!')
        fnameall = ['{}/{}I{}.DCM'.format(dcdir,dcftag,i) for i in range(1,len(alldcfiles)+1)]
    else:
        alldcfiles = glob.glob('%s/*.dcm' % dcdir)
        match = re.search(r'%s/([\w-]+)sl(\d+).dcm' % dcdir,alldcfiles[0])
        print match.group()
        if match:
            dcftag = match.group(1)
            #print dcftag
        else:
            raise RuntimeError('Error: cannot get dicom file tag!')
        fnameall = ['{}/{}sl{}.dcm'.format(dcdir,dcftag,i) for i in range(1,len(alldcfiles)+1)]
    for i in range(len(fnameall)):
        dc = dicom.read_file(fnameall[i])
        if i == 0:  # initialize
            dxyz = np.array([dc.PixelSpacing[0],dc.PixelSpacing[1],dc.SliceThickness])
            nx = dc.Columns
            ny = dc.Rows
            nz = len(alldcfiles)
            nxyz = np.array([nx,ny,nz])
            vol = np.empty((nz,ny,nx))
            print dc.SeriesDescription
        im = np.flipud(dc.pixel_array)*dc.RescaleSlope + dc.RescaleIntercept
        vol[i,:,:] = im
    return dxyz,nxyz,vol

def MakeDir(fdir):
# source: http://stackoverflow.com/questions/273192/check-if-a-directory-exists-and-create-it-if-necessary
    try:
        os.mkdir(fdir)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise
        else:
            print '\nBe careful! directory %s already exists!' % fdir



    

        





