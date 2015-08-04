#!/usr/bin/python
__author__ = 'ar'

import sys
import boto3

if __name__=='__main__':
    s3 = boto3.resource('s3')
    for bb in  s3.buckets.all():
        print (bb.name)
