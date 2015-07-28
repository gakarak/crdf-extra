#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'ar'


import locale
import dicom
import nibabel as nib
import zipfile

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
import inspect
# from nipype.interfaces.dcmstack import DcmStack

#########################################
DEF_MIN_NUMBER_OF_SLICES    = 0
DEF_NII_PREVIEW_SIZE        = (128,128,60)
DEF_DCM2NII_EXE             = "dcm2nii"
DEF_DCMANON_EXE             = "gdcmanon"
DEF_DCMDIRTMP_PREF          = "-tmp_%s" % time.time()

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
    RET_ERR_ANONYMIZAION:   'Incorrect data in anonymization (maybe data already anonymized)'
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

"""
Scan recursively directory, find DICOM files
and prepare Dictionary with a key=(PatientID, StudyID, SeriesNumber, StudyInstanceUID, Modality)
and values like (#SliceNumber, pathToDICOMSLice)
"""
def readDICOMSeries(wdir):
    lstDICOM=getListFiles(wdir,isRelPath=False)
    numDICOM=len(lstDICOM)
    dictDICOM={}
    if numDICOM>DEF_MIN_NUMBER_OF_SLICES:
        for ii in lstDICOM:
            dcm=None
            try:
                dcm=dicom_read_helper(ii, stop_before_pixels=True) #dicom.read_file(ii, stop_before_pixels=True)
            except:
                exitError(RET_READ_ERROR_DICOM, metaInfo=ii)
            if dcm is not None:
                tkey=(dcm.PatientID, dcm.StudyID, dcm.SeriesNumber, dcm.StudyInstanceUID, dcm.ImageNum, dcm.Modality)
                tval=(dcm.InstanceNumber, ii)
                if not dictDICOM.has_key(tkey):
                    dictDICOM[tkey]=[]
                dictDICOM[tkey].append(tval)
            else:
                exitError(RET_READ_ERROR_DICOM, metaInfo=ii)
        if len(dictDICOM)<1:
            exitError(RET_SERIES_NOTFOUND, metaInfo=wdir)
    else:
        exitError(RET_SMALL_NUM_DICOM)
    return dictDICOM


class DCMHelper:
    def __init__(self, parPath, parDCM=None):
        self.dcm=parDCM
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
        return "%s : %s : %s" % (self.path, self.dcm.SeriesInstanceUID, self.keyPos)
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
        tkey=(tdcm.StudyID, tdcm.StudyInstanceUID)
        tval=DCMHelper(tdcm,tpath)
        if not dictStudies.has_key(tkey):
            dictStudies[tkey]=[]
        dictStudies[tkey].append(tval)
    # 3. find all SeriesID for Studies, and reorganize data
    return dictStudies

def printDICOMSeries2(dictDICOMSeries):
    cnt=0
    for ii in dictDICOMSeries.keys():
        print "(",cnt,"): [", ii, "] -->"
        for jj in dictDICOMSeries[ii]:
            print "\t\t", jj
        cnt+=1

def mergeDICOMSeries(dictDICOM):
    retMerged={}
    return retMerged

"""
Select DICOM series with maximum number of slice
"""
def findBestDICOMSeries(dictDICOM):
    ret=None
    maxSlices=-1
    bestKey=None
    for kk in dictDICOM.keys():
        tnum=len(dictDICOM[kk])
        if tnum>maxSlices:
            maxSlices=tnum
            bestKey=kk
    # if maxSlices<DEF_MIN_NUMBER_OF_SLICES:
    #     exitError(RET_SMALL_NUM_SLICES)
    if bestKey is not None:
        arrSiD=[]
        lstDCM=dictDICOM[bestKey]
        for ii in lstDCM:
            try:
                arrSiD.append(int(ii[0]))
            except:
                exitError(RET_BAD_SLICE_ID, metaInfo=ii[1])
        sortIdx=np.argsort(arrSiD)
        arrID=[]
        arrFN=[]
        for ii in sortIdx:
            tval=lstDCM[ii]
            arrID.append(arrSiD[ii])
            arrFN.append(tval[1])
        ret=(arrID, arrFN)
    return (ret,bestKey)

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
Prepare Preview image on original size
"""
def generatePreviewOutput(lstIdFn, isDebug=False):
    lstId=lstIdFn[0]
    lstFn=lstIdFn[1]
    numFn=len(lstId)
    data=None
    cnt=0
    dcmStudyId='unknown-study-id'
    sizPad=8
    for ii in reversed(xrange(numFn)):
        tfn=lstFn[ii]
        try:
            tdcm=dicom_read_helper(tfn, stop_before_pixels=False) #dicom.read_file(tfn, stop_before_pixels=False)
            nr=tdcm.Rows
            nc=tdcm.Columns
            dcmStudyId=tdcm.StudyInstanceUID
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
    return (imgPano.copy(), dcmStudyId)

"""
convert DICOM to Nifti with dcm2nii utility
"""
def convertDICOM2Nifti(fdcm,fniiOut):
    if os.path.isfile(fdcm):
        ddir=os.path.dirname(fdcm)
        cmdLine='%s -a y -e n -r n %s' % (DEF_DCM2NII_EXE, fdcm)
        # cmdLine='%s -e n -r n %s >/dev/null' % (DEF_DCM2NII_EXE, fdcm)
        retCode=os.system(cmdLine)
        lstNii=glob.glob('%s/*.nii*' % ddir)
        if len(lstNii)>0:
            fnii=lstNii[0]
            shutil.move(fnii, fniiOut)
            if not os.path.isfile(fniiOut):
                exitError(RET_FILE_NOTFOUND, metaInfo=fniiOut)
        else:
            exitError(RET_CONV_NII_NOTFOUND, metaInfo=fdcm)
    else:
        exitError(RET_FILE_NOTFOUND, metaInfo=fdcm)


"""
Read, resize and save Nifti image
"""
def resizeNifti(fnii, fniiOut, newSize=DEF_NII_PREVIEW_SIZE):
    if not os.path.isfile(fnii):
        exitError(RET_FILE_NOTFOUND, metaInfo=fnii)
    odir=os.path.dirname(fniiOut)
    if not os.path.isdir(odir):
        exitError(RET_DIR_NOTFOUND, metaInfo=odir)
    try:
        imgNifti=nib.load(fnii)
        data=imgNifti.get_data()
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
        imgNiftiResiz=nib.Nifti1Image(dataNew, affineNew, header=imgNifti.header)
        nib.save(imgNiftiResiz, fniiOut)
    except:
        exitError(RET_NII_RESIZ_AND_SAVE, metaInfo=fnii)

"""
Export list-of-path to files in Zip archive
"""
def saveListFilesToZip(lstFiles, foutZip, pref=None):
    doutZip=os.path.dirname(foutZip)
    if not os.path.isdir(doutZip):
        exitError(RET_DIR_NOTFOUND, metaInfo=doutZip)
    zObj=zipfile.ZipFile(foutZip, 'w')
    zipDir='/'
    if pref is not None:
        zipDir=pref
    for ff in lstFiles:
        if not os.path.isfile(ff):
            exitError(RET_FILE_NOTFOUND, metaInfo=ff)
        try:
            fbaseName=os.path.basename(ff)
            zObj.write(ff, '/%s/%s' % (zipDir, fbaseName))
        except:
            exitError(RET_ERR_CREATE_ZIP, metaInfo=foutZip)

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
        tmpCmd="%s -r --continue -i \"%s\" -o \"%s\"" % (DEF_DCMANON_EXE, fdcmInp, fdcmOut)
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
Parse Command Line Arguments
"""
def parseCMD(argv):
    sizDef=[512,512]
    pProg=os.path.basename(argv[0])
    parser=argparse.ArgumentParser(
        description='Prepare preview image for DICOM directory. Example: %s -siz 512x512 -nii -zip -rm /path/to/directory-with-DICOM' % pProg)
    parser.add_argument('-siz', help='size output preview image, like 512x512')
    parser.add_argument('-preview', action="store_true",  help='flag, if present - generate preview image')
    parser.add_argument('-out', help='output prefix (without extension)')
    parser.add_argument('-rm',  action="store_true", help='flag, if present - remove input directory with DICOMs')
    parser.add_argument('-nii', action="store_true", help='flag, if present - create Nifti image')
    parser.add_argument('-zip', action="store_true", help='flag, if present - create Zip archive with selected DICOMs files')
    parser.add_argument('-anon', action="store_true", help='flag, if present - anonymize DICOM files in directory')
    parser.add_argument('wdir')
    args=parser.parse_args()
    retSiz=sizDef
    retWdir='./'
    retOfile=None
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
    if args.out is not None:
        retOfile=args.out
    ret=args
    ret.wdir=retWdir
    ret.siz=retSiz
    ret.out=retOfile
    # ret=(retWdir, retSiz, retOfile)
    return ret

#########################################
if __name__=='__main__':
    retCMD=parseCMD(sys.argv)
    wdir=retCMD.wdir
    wdirRoot=os.path.dirname(wdir)
    # (1) Anonymize DICOM files
    if retCMD.anon:
        if not makeAnonymization2(wdir, isCheckGDCMError=False, isRemoveTmpDir=True):
            exitError(RET_ERR_ANONYMIZAION, metaInfo=wdir)
    # (2) Preprocess and generate preview
    # lstDICOM=readDICOMSeries(wdir)
    # for ii in lstDICOM:
    #     print ii, " : ", len(lstDICOM[ii])
    # lstFDCM=getListFiles(wdir, isRelPath=False)

    # (2) Test DICOM series v.2
    # lstDICOM=readDICOMSeries2(wdir)
    # printDICOMSeries2(lstDICOM)

    lstFDCM=getListFiles(wdir)
    lstDCM=[]
    for ii in lstFDCM:
        lstDCM.append(DCMHelper(ii))
    lstDCM.sort()
    for ii in lstDCM:
        print ii

    # lstId=[]
    # for ii in lstFDCM:
    #     dcm=dicom.read_file(ii, stop_before_pixels=True)
    #     tmpId=(dcm.SeriesInstanceUID)#,dcm.InstanceNumber)
    #     lstId.append(tmpId)
    #     print tmpId
    # print "-----------"
    # print list(set(lstId))
    sys.exit(2)
    lstData,bestKey=findBestDICOMSeries(lstDICOM)
    imgPreviewRet=generatePreviewOutput(lstData)
    imgPreviewResiz=imgPreviewRet[0]
    newSize=retCMD.siz
    try:
        imgPreviewResiz=sk.transform.resize(imgPreviewRet[0],newSize,order=4,preserve_range=True).astype(np.uint8)
    except:
        exitError(RET_ERR_RESIZ_IMAGE, metaInfo="size=%s" % newSize)
    #
    studyId=imgPreviewRet[1] #os.path.basename(wdir)
    foutPrefix=os.path.join(wdirRoot, studyId)
    if retCMD.out is not None:
        tNewDir=retCMD.out
        if not os.path.isdir(tNewDir):
            os.makedirs(tNewDir)
        if not os.path.isdir(tNewDir):
            exitError(RET_DIR_NOTFOUND, metaInfo=tNewDir)
        foutPrefix=os.path.join(tNewDir, studyId)
    foutImg='%s_%d_%d.jpg' % (foutPrefix, newSize[1],newSize[0])
    foutNiiOrig='%s_orig.nii.gz' % foutPrefix
    foutNiiResiz='%s.nii.gz' % foutPrefix
    foutZip='%s.zip' % foutPrefix
    # (3) Save preview image
    try:
        sk.io.imsave(foutImg,imgPreviewResiz)
    except:
        exitError(RET_ERR_WRITE_IMAGE, metaInfo=foutImg)
    # (4) Save Nifti data
    if retCMD.nii:
        convertDICOM2Nifti(lstData[1][0], foutNiiOrig)
        resizeNifti(foutNiiOrig, foutNiiResiz)
    # (5) Prepare DICOM Zip archive
    if retCMD.zip:
        saveListFilesToZip(lstData[1], foutZip, pref=imgPreviewRet[1])
    # (6) Clean input data
    if retCMD.rm:
        try:
            shutil.rmtree(wdir)
        except:
            exitError(RET_ERR_REMOVE_DIR, metaInfo=wdir)
        # shutil.move(wdir, '%s-moved' % wdir)
    exitError(RET_SUCCESS, isPrintError=False)
