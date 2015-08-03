#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import nibabel as nib
import cv2
import skimage as sk
import skimage.transform
import numpy as np

def calcNormImageCT(img):
    timg=img.astype(np.float)
    vMin=-1000.
    vMax=+200.
    ret=255.*(timg-vMin)/(vMax-vMin)
    ret[ret<0]=0
    ret[ret>255]=255.
    return ret.astype(np.uint8)

if __name__=='__main__':
    fnii='/home/ar/data/data.crdf/пример данных/test_data2 директория/dcm/1.2.840.113619.2.55.3.2831206596.897.1412230330.947-2.nii.gz'
    fniiOut='%s-out.nii.gz' % fnii
    nii=nib.load(fnii)
    data=nii.get_data()
    newSize=(192,192,60)
    dataNew=sk.transform.resize(data,newSize,order=4, preserve_range=True)
    oldSize=data.shape
    affineOld=nii.affine.copy()
    affineNew=nii.affine.copy()
    k20_Old=float(oldSize[2])/float(oldSize[0])
    k20_New=float(newSize[2])/float(newSize[0])
    for ii in xrange(3):
        tCoeff=float(newSize[ii])/float(oldSize[ii])
        if ii==2:
            tCoeff=(affineNew[0,0]/affineOld[0,0])*(k20_Old/k20_New)
        affineNew[ii,ii]*=tCoeff
        affineNew[ii,3 ]*=tCoeff
    niiHdr=nii.header
    niiHdr.set_data_dtype(np.uint8)
    dataNew=calcNormImageCT(dataNew)
    imgNiftiResiz=nib.Nifti1Image(dataNew, affineNew, header=nii.header)
    nib.save(imgNiftiResiz, fniiOut)
