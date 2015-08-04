#!/usr/bin/python
__author__ = 'ar'

import sys
import cv2
import numpy as np
import boto3

def getImageFromS3Object(s3Object):
    img_array = np.asarray(bytearray(s3Object['Body'].read()), dtype=np.uint8)
    ret=cv2.imdecode(img_array, cv2.CV_LOAD_IMAGE_UNCHANGED)
    return ret

########################
if __name__=='__main__':
    s3=boto3.resource('s3')
    s3b=s3.Bucket('imlab')
    for oo in s3b.objects.all():
        print(oo)
        o1=s3b.Object(oo.key).get()
        winName="win"+oo.key
        try:
            cv2.imshow(winName, getImageFromS3Object(o1))
        except:
            pass
    cv2.waitKey()
