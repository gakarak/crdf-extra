#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import locale
import dicom
import nibabel as nib
import zipfile
import tempfile

import time
import glob
import argparse
import os
import sys
import shutil
import numpy as np
import skimage as sk
import skimage.transform
import skimage.color
import skimage.draw
import skimage.io
import matplotlib.pyplot as plt
import cv2
import inspect

#########################################
DEF_MIN_NUMBER_OF_SLICES    = 0
DEF_MIN_NUM_FILES_IN_SERIES = 25
DEF_NII_PREVIEW_SIZE_CT     = (192,192,60)
DEF_NII_PREVIEW_SIZE_CR     = 1024
DEF_DCM2NII_EXE             = "dcm2nii"
DEF_DCMANON_EXE             = "gdcmanon"
DEF_DCMDIRTMP_PREF          = "-tmp_%s" % time.time()
DEF_FOUT_FORMAT             = "%s-%s" # StudyID, SeriesNumber

RET_SUCCESS                 = 0
RET_FILE_NOTFOUND           = 1
RET_DIR_NOTFOUND            = 2
RET_READ_ERROR_DICOM        = 3
RET_DICOM_NOTFOUND          = 4
RET_SMALL_NUM_DICOM         = 5
RET_SMALL_NUM_SLICES        = 6
RET_SERIES_NOTFOUND         = 7
RET_BAD_SLICE_ID            = 8
RET_BAD_RESOLUTION          = 9
RET_ERR_WRITE_IMAGE         = 10
RET_ERR_RESIZ_IMAGE         = 11
RET_BAD_CMD_ARGS            = 12
RET_CONV_NII_NOTFOUND       = 13
RET_NII_RESIZ_AND_SAVE      = 14
RET_ERR_CREATE_ZIP          = 15
RET_ERR_REMOVE_DIR          = 16
RET_ERR_MOVE_DIR            = 17
RET_ERR_ANONYMIZAION        = 18
RET_ERR_CREATE_DIR          = 19
errorCodes = {
    RET_SUCCESS:            'Success',
    RET_FILE_NOTFOUND:      'File not found',
    RET_DIR_NOTFOUND:       'Directory not found',
    RET_READ_ERROR_DICOM:   'Error read DICOM file',
    RET_DICOM_NOTFOUND:     'DICOM not found in directory',
    RET_SMALL_NUM_DICOM:    'Too small number of dicom files',
    RET_SMALL_NUM_SLICES:   'Too small number of slices in DICOM Study',
    RET_SERIES_NOTFOUND:    'DICOM Series not found',
    RET_BAD_SLICE_ID:       'Bad Slice Number on DICOM file',
    RET_BAD_RESOLUTION:     'Incorrect Slice resolution',
    RET_ERR_WRITE_IMAGE:    'Cant write image to file',
    RET_ERR_RESIZ_IMAGE:    'Cant resize image',
    RET_BAD_CMD_ARGS:       'Incorrect command line arguments',
    RET_CONV_NII_NOTFOUND:  'Incorrect conversion DICOM->NII.GZ: output file not found',
    RET_NII_RESIZ_AND_SAVE: 'Cant convert and save Nifti Image',
    RET_ERR_CREATE_ZIP:     'Cant create Zip archive',
    RET_ERR_REMOVE_DIR:     'Cant remove directory',
    RET_ERR_ANONYMIZAION:   'Incorrect data in anonymization (maybe data already anonymized)',
    RET_ERR_CREATE_DIR:     'Cant create directory'
}
lstExt=['.dcm', '.dicom', '.Dicom', '.Dcm']

#########################################
"""
Helper function: return Exit code end print Error message in STDERR
"""
def exitError(errCode, isPrintError=True, metaInfo=None):
    if isPrintError:
        isGoodKey=errorCodes.has_key(errCode)
        strErrCode='Unknown error code'
        if isGoodKey:
            strErrCode=errorCodes[errCode]
        if metaInfo is None:
            print >> sys.stderr, '**ERROR (%s) [line=%d] : %s' % (errCode, inspect.currentframe().f_back.f_lineno, strErrCode)
        else:
            print >> sys.stderr, '**ERROR (%s) [line=%d] : %s, info = [%s]' % (errCode, inspect.currentframe().f_back.f_lineno, strErrCode, metaInfo)
    sys.exit(errCode)

"""
Find recursively all files in directory with predefined file extension
"""
def getListFiles(wdir, parLstExt=lstExt, maxNumFiles=2000, isRelPath=False):
    ret=[]
    cnt=0
    for root,_,files in os.walk(wdir):
        for ff in files:
            if ff.endswith(tuple(parLstExt)):
                totPath=os.path.join(root, ff)
                if not isRelPath:
                    ret.append(totPath)
                else:
                    ret.append(os.path.relpath(totPath, wdir))
                cnt+=1
                if cnt>maxNumFiles:
                    break
    return ret

def dicom_read_helper(fdcm, stop_before_pixels=False):
    _,tcodec=locale.getdefaultlocale()
    return dicom.read_file(fdcm.decode(tcodec), stop_before_pixels=stop_before_pixels)

class DCMHelper:
    def __init__(self, parPath, parDCM=None):
        self.dcm=parDCM
        self.path=parPath
        if not parDCM:
            try:
                self.dcm=dicom_read_helper(parPath, stop_before_pixels=True)
                self.path=parPath
            except:
                self.path=None
        try:
            self.keyPos=int(self.dcm.InstanceNumber)
        except:
            self.keyPos=10e6
    def getStudyKey(self):
        return (self.dcm.StudyInstanceUID, self.dcm.StudyID)
    def getSeriesKey(self):
        return self.dcm.SeriesInstanceUID
    def toString(self):
        return "%s (%s) : %s-%s : %s" % (self.path, self.dcm.Modality, self.dcm.SeriesInstanceUID, self.dcm.SeriesNumber, self.keyPos)
    def __repr__(self):
        return self.toString()
    def __str__(self):
        return self.toString()
    def __lt__(self, other):
        return self.keyPos<other.keyPos

"""
Version #2
Algorithm:
    (1) scan directory recursively and find all DICOM-headers
    (2) find all study
    (3) find all series for study, if:
        (a) Modality=XRay -> generate preview
        (b) Modality=CT   -> search largest CT-series and generate 3D-preview for it
    -------
    :return: dict-structure
        {
            StudyInstanceUID_1: {

            },
            ...
            StudyInstanceUID_N: {
            }
        }
"""
def readDICOMSeries2(wdir):
    lstFDCM=getListFiles(wdir,isRelPath=False)
    # 1. Read DICOM Headers
    lstDCM=[]
    lstDCMPath=[]
    for ii in lstFDCM:
        try:
            tdcm=dicom_read_helper(ii, stop_before_pixels=True)
            lstDCM.append(tdcm)
            lstDCMPath.append(ii)
        except:
            exitError(RET_READ_ERROR_DICOM, metaInfo=ii)
    # 2. find all StudiesID
    dictStudies={}
    for ii in xrange(len(lstDCM)):
        tdcm=lstDCM[ii]
        tpath=lstDCMPath[ii]
        tkeyStudy=(tdcm.StudyInstanceUID, tdcm.StudyID, tdcm.Modality)
        tKeySeries=(tdcm.StudyInstanceUID, tdcm.SeriesInstanceUID, tdcm.SeriesNumber, tdcm.Modality)
        tval=DCMHelper(tpath, tdcm)
        if not dictStudies.has_key(tkeyStudy):
            dictStudies[tkeyStudy]={}
        if not dictStudies[tkeyStudy].has_key(tKeySeries):
            dictStudies[tkeyStudy][tKeySeries]=[]
        dictStudies[tkeyStudy][tKeySeries].append(tval)
    # 3. find all SeriesID for Studies, and reorganize data
    tmpStudyKeys=dictStudies.keys()
    for kkStudy in tmpStudyKeys:
        tmpSeriesKeys=dictStudies[kkStudy].keys()
        for kkSeries in tmpSeriesKeys:
            tmpValues=dictStudies[kkStudy][kkSeries]
            tmpValues.sort()
        # print type(tmpValues)
    return dictStudies

def printDICOMSeries2(dictDICOMSeries, isQuickPrint=False):
    cnt=0
    for iiStudy in dictDICOMSeries.keys():
        print "Study {",cnt,"}: [", iiStudy, "] -->"
        tmpDictSeries=dictDICOMSeries[iiStudy]
        cntSeries=0
        for iiSeries in tmpDictSeries.keys():
            if not isQuickPrint:
                print "\t\t(",cntSeries,"): [", iiSeries, "] -->"
                for jj in tmpDictSeries[iiSeries]:
                    print "\t\t\t\t", jj
            else:
                print "\t\t(",cntSeries,"): [", iiSeries, "] --> #%d DICOM-files" % len(tmpDictSeries[iiSeries])
            cntSeries+=1
        cnt+=1

def resizeImageToSize(img, sizNew, parBorder=0, parInterpolation=2L): # parInterpolation=cv2.INTER_CUBIC
    if len(img.shape)<3:
        img=cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    sizImg=img.shape
    if (sizNew[0]<2) or (sizNew[1]<2) or (sizImg[0]<2) or (sizImg[1]<2):
        return None
    sizImgf=np.array(sizImg, np.float)
    sizNewf=np.array(sizNew, np.float)
    k1=sizNewf[0]/sizNewf[1]
    k2=sizImgf[0]/sizImgf[1]
    eps=np.abs(k1-k2)/np.abs(k1+k2)
    if eps<0.002:
        if parBorder<1:
            return cv2.resize(img, sizNew, interpolation=parInterpolation)
        else:
            ret=cv2.resize(img, sizNew, interpolation=parInterpolation)
            ret=cv2.copyMakeBorder(ret, parBorder,parBorder,parBorder,parBorder, borderType=cv2.BORDER_CONSTANT, value=0)
            return cv2.resize(ret, sizNew, interpolation=cv2.INTER_CUBIC)
    #
    parScl=sizNewf[0]/sizImgf[0]
    sizImgNewf=np.array( (sizNewf[0], sizImgf[1]*sizNewf[0]/sizImgf[0]),  np.float)
    if (k2<k1):
        sizImgNewf=np.array( (sizImgf[0]*sizNewf[1]/sizImgf[1], sizNewf[1]),  np.float)
        parScl=sizNewf[1]/sizImgf[1]
    dx=(sizNewf[1]-sizImgNewf[1])/2.
    dy=(sizNewf[0]-sizImgNewf[0])/2.
    warpMat=np.zeros((2,3), np.float)
    warpMat[0,0]=parScl
    warpMat[1,1]=parScl
    warpMat[0,2]=+dx
    warpMat[1,2]=+dy
    if parBorder<1:
        return cv2.warpAffine(img, warpMat, sizNew[::-1])
    else:
        ret=cv2.warpAffine(img, warpMat, sizNew[::-1])
        ret=cv2.copyMakeBorder(ret, parBorder,parBorder,parBorder,parBorder, borderType=cv2.BORDER_CONSTANT, value=0)
        return cv2.resize(ret, sizNew[::-1], interpolation=cv2.INTER_CUBIC)

"""
Normalize CT image by Lung-preset
"""
def calcNormImageCT(img):
    timg=img.astype(np.float)
    vMin=-1000.
    vMax=+200.
    ret=255.*(timg-vMin)/(vMax-vMin)
    ret[ret<0]=0
    ret[ret>255]=255.
    return ret.astype(np.uint8)

"""
Prepare preview for CT-Image on original size
"""
def makePreviewImageForCT(lstDCMHelper, isDebug=False):
    numFn=len(lstDCMHelper)
    data=None
    cnt=0
    sizPad=8
    for ii in reversed(xrange(numFn)):
        tfn=lstDCMHelper[ii].path
        try:
            tdcm=dicom_read_helper(tfn, stop_before_pixels=False)
            nr=tdcm.Rows
            nc=tdcm.Columns
            if data is None:
                data=np.zeros((nr,nc,numFn), dtype=np.float)
            else:
                if not ( (data.shape[0]==nr) and (data.shape[1]==nc) ):
                    exitError(RET_BAD_RESOLUTION, metaInfo=tfn)
            rescaleSlope=tdcm.RescaleSlope
            rescaleIntercept=tdcm.RescaleIntercept
            data[:,:,cnt]=rescaleIntercept + rescaleSlope*(tdcm.pixel_array.astype(np.float))
            cnt+=1
        except:
            exitError(RET_READ_ERROR_DICOM, metaInfo=tfn)
    lstZp=np.linspace(0.8,0.3,3)
    lstImg=[]
    for pp in lstZp:
        tidx=round(pp*numFn)
        lstImg.append(np.pad(calcNormImageCT(data[:,:,tidx]), sizPad, 'constant', constant_values=(0)))
    imgX=np.rot90(data[data.shape[0]/2,:,:])
    imgX=sk.transform.resize(imgX.copy(), data.shape[:2], order=4)
    lstImg.append(np.pad(calcNormImageCT(imgX), sizPad, 'constant', constant_values=(0)))
    #
    lstImgRGB=[]
    for ii in lstImg:
        lstImgRGB.append(sk.color.gray2rgb(ii))
    lstColors=((0,255,0),(255,255,0),(255,0,0))
    for ii in xrange(len(lstZp)):
        tsiz=imgX.shape
        tdw=42
        tr=int(tsiz[0] - round(tsiz[0]*lstZp[ii]))
        zzRange=range(-1,2,1)
        if imgX.shape[0]>400:
            zzRange=range(-2,3,1)
        for zz in zzRange:
            trr,tcc=sk.draw.line(tr+zz,tdw,tr+zz,tsiz[1]-tdw)
            sk.draw.set_color(lstImgRGB[3],(trr,tcc), lstColors[ii])
    for ii in range(len(lstImgRGB)-1):
        tsiz=lstImgRGB[ii].shape
        trad=9
        if tsiz[0]>400:
            trad=12
        trr,tcc=sk.draw.circle(64,tsiz[1]-64,trad)
        sk.draw.set_color(lstImgRGB[ii],(trr,tcc), lstColors[ii])
    imgPH0=np.concatenate((lstImgRGB[0],lstImgRGB[1]), axis=1)
    imgPH1=np.concatenate((lstImgRGB[2],lstImgRGB[3]), axis=1)
    imgPano=np.concatenate((imgPH0,imgPH1))
    #
    if isDebug:
        plt.imshow(imgPano)
        plt.show()
    ret=cv2.cvtColor(imgPano,cv2.COLOR_RGB2BGR)
    return ret

def makePreviewImageForCR(lstDCMHelper):
    fdcm=lstDCMHelper[0].path
    ret=None
    try:
        tdcm=dicom_read_helper(fdcm, stop_before_pixels=False)
        data=tdcm.pixel_array
        ret=cv2.normalize(data,None,0,255,cv2.NORM_MINMAX,cv2.CV_8U)
    except:
        pass
    return ret

def preparePreviewForDICOM(lstDICOMDict, outDir, sizPreview):
    sizW=sizPreview[0]
    sizH=sizPreview[1]
    for kkStudy in lstDICOMDict.keys():
        tmpDictSeries=lstDICOMDict[kkStudy]
        for kkSeries in tmpDictSeries.keys():
            tmpModality=kkSeries[3]
            numFilesInSeries=len(tmpDictSeries[kkSeries])
            foutPreview=DEF_FOUT_FORMAT % (kkStudy[0], kkSeries[2])
            foutPreview="%s.%d.%d.jpg" % (foutPreview, sizW, sizH)
            foutPreview=os.path.join(outDir,foutPreview)
            if tmpModality=='CT':
                # print 'CT: ', kkSeries
                if numFilesInSeries>DEF_MIN_NUM_FILES_IN_SERIES:
                    imgCT=makePreviewImageForCT(tmpDictSeries[kkSeries], isDebug=False)
                    if imgCT is not None:
                        imgCTr=resizeImageToSize(imgCT, tuple(sizPreview))
                        cv2.imwrite(foutPreview, imgCTr)
            else:
                # print 'CR: ', kkSeries
                imgCR=makePreviewImageForCR(tmpDictSeries[kkSeries])
                if imgCR is not None:
                    imgCRr=resizeImageToSize(imgCR, tuple(sizPreview), parBorder=8)
                    cv2.imwrite(foutPreview, imgCRr)

def makePreviewNifti(fniiInp, fniiOut, newSize=DEF_NII_PREVIEW_SIZE_CT):
    try:
        imgNifti=nib.load(fniiInp)
        data=imgNifti.get_data()
        if data.shape[2]>1 : # 3D-Image
            dataNew=sk.transform.resize(data,newSize,order=4, preserve_range=True)
            oldSize=data.shape
            affineOld=imgNifti.affine.copy()
            affineNew=imgNifti.affine.copy()
            k20_Old=float(oldSize[2])/float(oldSize[0])
            k20_New=float(newSize[2])/float(newSize[0])
            for ii in xrange(3):
                tCoeff=float(newSize[ii])/float(oldSize[ii])
                if ii==2:
                    tCoeff=(affineNew[0,0]/affineOld[0,0])*(k20_Old/k20_New)
                affineNew[ii,ii]*=tCoeff
                affineNew[ii,3 ]*=tCoeff
            dataNew=calcNormImageCT(dataNew)
            niiHdr=imgNifti.header
            niiHdr.set_data_dtype(np.uint8)
            imgNiftiResiz=nib.Nifti1Image(dataNew, affineNew, header=niiHdr)
            nib.save(imgNiftiResiz, fniiOut)
        else: # 2D-Image
            maxSize=np.max(data.shape[:2])
            affineNew=imgNifti.affine.copy()
            if maxSize>DEF_NII_PREVIEW_SIZE_CR:
                tCoeff=float(DEF_NII_PREVIEW_SIZE_CR)/float(maxSize)
                maxSize=DEF_NII_PREVIEW_SIZE_CR
                newSize=(maxSize, (data.shape[1]*maxSize)/data.shape[0], 1)
                if data.shape[1]>data.shape[0]:
                    newSize=((data.shape[0]*maxSize)/data.shape[1], maxSize, 1)
                dataNew=sk.transform.resize(data,newSize,order=4, preserve_range=True)
                affineNew[0,0]*=tCoeff
                affineNew[1,1]*=tCoeff
            else:
                dataNew=data
            dataNew=cv2.normalize(dataNew[:,:,0],None,0,255, cv2.NORM_MINMAX,cv2.CV_8U)
            niiHdr=imgNifti.header
            niiHdr.set_data_dtype(np.uint8)
            imgNiftiResiz=nib.Nifti1Image(dataNew, affineNew, header=niiHdr)
            nib.save(imgNiftiResiz, fniiOut)
    except:
        exitError(RET_NII_RESIZ_AND_SAVE, metaInfo=fniiInp)

def prepareZipForDICOM(lstDICOMDict,outDir, dirRelative, pref=None):
    for kkStudy in lstDICOMDict.keys():
        tmpDictSeries=lstDICOMDict[kkStudy]
        foutZip="%s.zip" % kkStudy[0]
        foutZip=os.path.join(outDir,foutZip)
        zObj=zipfile.ZipFile(foutZip, 'w')
        zipDir='/'
        if pref is not None:
            zipDir=pref
        for kkSeries in tmpDictSeries.keys():
            tdcmHelperList=tmpDictSeries[kkSeries]
            for tdcmHelper in tdcmHelperList:
                tpathRel=os.path.relpath(tdcmHelper.path, dirRelative)
                if os.path.isfile(tdcmHelper.path):
                    try:
                        zObj.write(tdcmHelper.path, tpathRel)
                    except:
                        pass

"""
convert DICOM to Nifti with dcm2nii utility
"""
def convertDICOM2Nifti2(lstDICOMDict, outDir, isNiiPreview=False):
    for kkStudy in lstDICOMDict.keys():
        tmpDictSeries=lstDICOMDict[kkStudy]
        # print kkStudy
        for kkSeries in tmpDictSeries.keys():
            # print "\t\t",kkSeries
            tmpModality=kkSeries[3]
            isNeedConvert=True
            if tmpModality=='CT':
                tmpNumFilesInSeries=len(tmpDictSeries[kkSeries])
                if tmpNumFilesInSeries<DEF_MIN_NUM_FILES_IN_SERIES:
                    isNeedConvert=False
            if isNeedConvert:
                tmpDir=tempfile.mkdtemp('dcm2nii')
                tmpListDCM=tmpDictSeries[kkSeries]
                foutNii="%s.nii.gz" % (DEF_FOUT_FORMAT % (kkStudy[0], kkSeries[2]))
                foutNiiPath=os.path.join(outDir,foutNii)
                try:
                    for tdcm in tmpListDCM:
                        shutil.copy2(tdcm.path, tmpDir)
                    tmpListDCMinTmpDIR=getListFiles(tmpDir)
                    cmdLine='%s -a y -e n -r n %s' % (DEF_DCM2NII_EXE, tmpListDCMinTmpDIR[0])
                    retCode=os.system(cmdLine)
                    lstNii=glob.glob('%s/*.nii*' % tmpDir)
                    if len(lstNii)>0:
                        fnii=lstNii[0]
                        shutil.move(fnii, foutNiiPath)
                    else:
                        exitError(RET_CONV_NII_NOTFOUND, metaInfo=cmdLine)
                except:
                    print 'Error processing series ', kkSeries
                shutil.rmtree(tmpDir, ignore_errors=True)
                #
                if isNiiPreview:
                    foutNiiPrv="%s-preview.nii.gz" % (DEF_FOUT_FORMAT % (kkStudy[0], kkSeries[2]))
                    foutNiiPrvPath=os.path.join(outDir,foutNiiPrv)
                    makePreviewNifti(foutNiiPath, foutNiiPrvPath)

"""
Anonymization function #1: one call of gdcmanon but
is unstable in the case of a partial anonymization
of data
"""
def makeAnonymization(dirDICOM):
    isNoError=True
    if not os.path.isdir(dirDICOM):
        exitError(RET_DIR_NOTFOUND, metaInfo=dirDICOM)
    dirDICOMtmp="%s%s" % (dirDICOM,DEF_DCMDIRTMP_PREF)
    try:
        shutil.move(dirDICOM,dirDICOMtmp)
    except:
        exitError(RET_ERR_MOVE_DIR, metaInfo=dirDICOMtmp)
    tmpCmd="%s -r --continue -i \"%s\" -o \"%s\"" % (DEF_DCMANON_EXE, dirDICOMtmp, dirDICOM)
    retCode=-1
    try:
        retCode=os.system(tmpCmd)
    except Exception as e:
        print e.message
    if not os.path.isdir(dirDICOM):
        isNoError=False
        shutil.move(dirDICOMtmp,dirDICOM)
    else:
        if retCode!=0:
            shutil.rmtree(dirDICOM)
            shutil.move(dirDICOMtmp,dirDICOM)
    return isNoError

"""
Anonymization function #2: one call per DICOM file, but
work stable - if gdcmanon return error - just copy file
"""
def makeAnonymization2(dirDICOM, isCheckGDCMError=True, isRemoveTmpDir=False):
    isNoError=True
    if not os.path.isdir(dirDICOM):
        exitError(RET_DIR_NOTFOUND, metaInfo=dirDICOM)
    dirDICOMtmp="%s%s" % (dirDICOM,DEF_DCMDIRTMP_PREF)
    try:
        shutil.move(dirDICOM,dirDICOMtmp)
    except:
        exitError(RET_ERR_MOVE_DIR, metaInfo=dirDICOMtmp)
    lstFDCMRel=getListFiles(dirDICOMtmp, parLstExt=lstExt, isRelPath=True)
    cntTot=len(lstFDCMRel)
    cntGood=0
    for ll in lstFDCMRel:
        fdcmInp=os.path.join(dirDICOMtmp, ll)
        fdcmOut=os.path.join(dirDICOM, ll)
        dirOut=os.path.dirname(fdcmOut)
        if not os.path.isdir(dirOut):
            os.makedirs(dirOut)
        tmpCmd="%s --dumb --empty 10,10 --empty 10,20 --remove 10,40 --remove 10,1010 -r --continue -i \"%s\" -o \"%s\"" % (DEF_DCMANON_EXE, fdcmInp, fdcmOut)
        #tmpCmd="%s -r --continue -i \"%s\" -o \"%s\"" % (DEF_DCMANON_EXE, fdcmInp, fdcmOut)
        retCode=os.system(tmpCmd)
        if retCode==0:
            cntGood+=1
        else:
            shutil.copy2(fdcmInp, fdcmOut)
    if isCheckGDCMError:
        if cntGood!=cntTot:
            isNoError=False
            shutil.rmtree(dirDICOM)
            shutil.move(dirDICOMtmp,dirDICOM)
    if isRemoveTmpDir:
        shutil.rmtree(dirDICOMtmp)
    return isNoError

"""
Anonymization function #3: one call per DICOM file, but
work stable - if gdcmanon return error - just copy file
append shadow creation tmp-file for solve non-latin characters in path (Windows bug)
"""
def makeAnonymization3(dirDICOM, isCheckGDCMError=True, isRemoveTmpDir=False):
    isNoError=True
    if not os.path.isdir(dirDICOM):
        exitError(RET_DIR_NOTFOUND, metaInfo=dirDICOM)
    dirDICOMtmp="%s%s" % (dirDICOM,DEF_DCMDIRTMP_PREF)
    try:
        shutil.move(dirDICOM,dirDICOMtmp)
    except:
        exitError(RET_ERR_MOVE_DIR, metaInfo=dirDICOMtmp)
    lstFDCMRel=getListFiles(dirDICOMtmp, parLstExt=lstExt, isRelPath=True)
    cntTot=len(lstFDCMRel)
    cntGood=0
    tmpDir=tempfile.mkdtemp(suffix='anon3')
    tmpFileInp=os.path.join(tmpDir, 'tmp_dcm_inp.dcm')
    tmpFileOut=os.path.join(tmpDir, 'tmp_dcm_out.dcm')
    for ll in lstFDCMRel:
        fdcmInp=os.path.join(dirDICOMtmp, ll)
        fdcmOut=os.path.join(dirDICOM, ll)
        # copy input to tmp-input
        try:
            shutil.copy2(fdcmInp,tmpFileInp)
        except:
            print 'Cant copy INPUT tmp-file [%s] -> [%s]' % (fdcmInp,tmpFileInp)
        #
        dirOut=os.path.dirname(fdcmOut)
        if not os.path.isdir(dirOut):
            os.makedirs(dirOut)
        # tmpCmd="%s --dumb --empty 10,10 --empty 10,20 --remove 10,40 --remove 10,1010 -r --continue -i \"%s\" -o \"%s\"" % (DEF_DCMANON_EXE, fdcmInp, fdcmOut)
        tmpCmd="%s --dumb --empty 10,10 --empty 10,20 --remove 10,40 --remove 10,1010 -r --continue -i \"%s\" -o \"%s\"" % (DEF_DCMANON_EXE, tmpFileInp, tmpFileOut)
        retCode=os.system(tmpCmd)
        try:
            shutil.copy2(tmpFileOut,fdcmOut)
        except:
            print 'Cant copy OUTPUT tmp-file [%s] -> [%s]' % (tmpFileOut,fdcmOut)
        if retCode==0:
            cntGood+=1
        else:
            shutil.copy2(fdcmInp, fdcmOut)
    try:
        shutil.rmtree(tmpDir)
    except:
        print '!!!WARNING!!! Cant remove tmp-anon directory [%s]' % tmpDir
    if isCheckGDCMError:
        if cntGood!=cntTot:
            isNoError=False
            shutil.rmtree(dirDICOM)
            shutil.move(dirDICOMtmp,dirDICOM)
    if isRemoveTmpDir:
        shutil.rmtree(dirDICOMtmp)
    return isNoError

"""
Parse Command Line Arguments
"""
def parseCMD(argv):
    sizDef=[512,512]
    pProg=os.path.basename(argv[0])
    parser=argparse.ArgumentParser(
        description='Prepare preview image for DICOM directory. Example: %s -siz 512x512 -nii -zip -rm /path/to/directory-with-DICOM' % pProg)
    parser.add_argument('-siz', help='size output preview image, like 512x512')
    parser.add_argument('-preview', action="store_true",  help='flag, if present - generate preview image')
    parser.add_argument('-out', help='output directory (if empty - use input directory)')
    parser.add_argument('-rm',  action="store_true", help='flag, if present - remove input directory with DICOMs')
    parser.add_argument('-nii', action="store_true", help='flag, if present - create Nifti image')
    parser.add_argument('-niiprv', action="store_true", help='flag, if present - create preview in Nifti format')
    parser.add_argument('-zip', action="store_true", help='flag, if present - create Zip archive with selected DICOMs files')
    parser.add_argument('-anon', action="store_true", help='flag, if present - anonymize DICOM files in directory')
    parser.add_argument('-info', action="store_true", help='flag, print info about directory with DICOMs')
    parser.add_argument('-infoq', action="store_true", help='flag, print quick info about directory with DICOMs')
    parser.add_argument('wdir')
    args=parser.parse_args()
    retSiz=sizDef
    retWdir='./'
    if args.siz is not None:
        strSiz=args.siz.split('x')
        if len(strSiz)>1:
            try:
                sizW=int(strSiz[0])
                sizH=int(strSiz[1])
                retSiz=[sizH, sizW]
            except:
                pass
    if args.wdir is not None:
        if os.path.isdir(args.wdir):
            retWdir=args.wdir
        else:
            exitError(RET_DIR_NOTFOUND, metaInfo=args.wdir)
    else:
        parser.print_help()
        exitError(RET_BAD_CMD_ARGS, metaInfo=argv)
    dirOut=retWdir
    if args.out is not None:
        dirOut=args.out
    ret=args
    if ret.niiprv:
        ret.nii=True
    ret.wdir=retWdir
    ret.siz=retSiz
    ret.out=dirOut
    # ret=(retWdir, retSiz, retOfile)
    return ret

#########################################
if __name__=='__main__':
    retCMD=parseCMD(sys.argv)
    wdir=retCMD.wdir
    wdirRoot=os.path.dirname(wdir)
    # (1) Anonymize DICOM files
    if retCMD.anon:
        print ":::makeAnonymization3()"
        if not makeAnonymization3(wdir, isCheckGDCMError=False, isRemoveTmpDir=False):
            exitError(RET_ERR_ANONYMIZAION, metaInfo=wdir)

    # (2) check output directory
    if not os.path.isdir(retCMD.out):
        try:
            os.makedirs(retCMD.out)
        except:
            exitError(RET_ERR_CREATE_DIR, metaInfo=retCMD.out)

    # (3) Prepare dict with DICOM series
    lstFDCM=getListFiles(wdir)
    lstFDCMRel=getListFiles(wdir, isRelPath=True)
    lstDICOM=readDICOMSeries2(wdir)

    # (4) Check Info parameter
    if retCMD.info:
        printDICOMSeries2(lstDICOM, isQuickPrint=False)
        sys.exit(RET_SUCCESS)
    if retCMD.infoq:
        printDICOMSeries2(lstDICOM, isQuickPrint=True)
        sys.exit(RET_SUCCESS)

    # (5) Check -nii parameter: convert DICOM to Nifti
    if retCMD.nii:
        print ":::convertDICOM2Nifti2()"
        convertDICOM2Nifti2(lstDICOM, retCMD.out, isNiiPreview=retCMD.niiprv)

    # (6) Check -preview parameter: make preview for CT and CR
    if retCMD.preview:
        print ":::preparePreviewForDICOM()"
        preparePreviewForDICOM(lstDICOM, retCMD.out, sizPreview=retCMD.siz)

    # (7) Check -zip parameter: make zip archive for every StudyUID
    if retCMD.zip:
        print ":::prepareZipForDICOM()"
        prepareZipForDICOM(lstDICOM, retCMD.out, wdir)

    # (8) Check -rm parameter: clean input directory
    if retCMD.rm:
        try:
            shutil.rmtree(wdir)
        except:
            exitError(RET_ERR_REMOVE_DIR, metaInfo=wdir)
        # shutil.move(wdir, '%s-moved' % wdir)
    exitError(RET_SUCCESS, isPrintError=False)
