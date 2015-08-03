#!/usr/bin/python
__author__ = 'ar'

import skimage as sk
import numpy as np
import os
import sys
import cv2
import argparse
import dicom
import locale

################################
def dicom_read_helper(fdcm, stop_before_pixels=False):
    _,tcodec=locale.getdefaultlocale()
    return dicom.read_file(fdcm.decode(tcodec), stop_before_pixels=stop_before_pixels)

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


def parseCMD(argv):
    pProg=os.path.basename(argv[0])
    parser=argparse.ArgumentParser(
        description='show/save prepare image for IMAGE/DICOM file. Example: %s -siz 512x512 /path/to/inpit-image [-out /path/to/out-image]' % pProg)
    parser.add_argument('-siz', help='size output preview image, like 512x512')
    parser.add_argument('-border', help='size output preview image, like 16')
    parser.add_argument('-out', help='output file')
    parser.add_argument('finp')
    args=parser.parse_args()
    #
    tmpSiz=[512,512]
    if args.siz is not None:
        strSiz=args.siz.split('x')
        if len(strSiz)>1:
            try:
                sizW=int(strSiz[0])
                sizH=int(strSiz[1])
                tmpSiz=[sizH, sizW]
            except:
                pass
    args.siz=tuple(tmpSiz)
    tBorder=0
    if args.border is not None:
        try:
            tBorder=int(args.border)
            if tBorder<0:
                tBorder=0
        except:
            pass
    args.border=tBorder
    if args.finp is not None:
        if not os.path.isfile(args.finp):
            print 'Incorrent input file-path [%s]' % args.finp
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)
    return args

################################
if __name__=='__main__':
    retCMD=parseCMD(sys.argv)
    img=None
    isDICOM=False
    try:
        img=dicom_read_helper(retCMD.finp, stop_before_pixels=False).pixel_array
        img=cv2.normalize(img,None,0,255, cv2.NORM_MINMAX,cv2.CV_8U)
        isDICOM=True
    except:
        pass
    if not isDICOM:
        try:
            img=cv2.imread(retCMD.finp, cv2.CV_LOAD_IMAGE_UNCHANGED)
        except:
            print "Incorrect IMAGE [%s]" % retCMD.finp
            sys.exit(1)
    if img is None:
        print "Cant load input image [%s]" % retCMD.finp
    imgr=resizeImageToSize(img, retCMD.siz, parBorder=retCMD.border)
    # print imgr.shape
    if retCMD.out is not None:
        cv2.imwrite(retCMD.out, imgr)
    else:
        cv2.imshow("image-output", imgr)
        cv2.waitKey(0)
