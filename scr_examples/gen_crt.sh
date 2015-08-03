#!/bin/bash


##cmd1="openssl genrsa -des3 -out CA_DICOM_key.pem 2048"
cmd1="openssl genrsa -out CA_DICOM_key.pem"
echo "[START] : { $cmd1 }"
echo "--->"
$cmd1


cmd2="openssl req -new -key CA_DICOM_key.pem -x509 -days 9999 -out CA_DICOM_cert.cer"
echo "[START] : { $cmd1 }"
echo "--->"
$cmd2