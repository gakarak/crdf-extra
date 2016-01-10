#!/usr/bin/python
from duplicity.tarfile import _data

__author__ = 'ar'

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fcsv='/home/ar/tmp/CRDF/clinical_records_20160107_140558.unzip/cases.csv'
fcsvOUT_CT='%s_info_ct_v3.csv' % fcsv
isUseDT=True

par_keys=[
        'CT',
        'X-RAY',
        'lab_findings',
        'demographics',
        'followup',
        'drug_resistance',
        'drug_resistance_bactec_test',
        'drug_resistance_hain_test',
        'drug_resistance_gene_test',
        'treatment',
        'radiology_protocol',
        'comments',
        'sequences']

################################
paramTot_lab_findings=['isolate_source', 'smear_result', 'smear_score', 'culture_result',
 'culture_score', 'was_cultured_from_sample', 'method_of_identification',
 'isolate_id', 'provider_name', 'provider_abbreviation',
 'provider_street_address_line', 'provider_city', 'provider_postal_code',
 'provider_country', 'provider_telecom']

paramTot_demographics=['date_of_birth', 'gender', 'birth_address_country', 'clinical_record_id',
 'provider_name', 'provider_abbreviation', 'provider_street_address_line',
 'provider_city', 'provider_postal_code', 'provider_country',
 'provider_telecom']

paramTot_followup=['co_morbidity', 'diagnosis', 'height_cm', 'weight_cm', 'disability_level',
 'symptoms_of_tb', 'result_of_previous_treatment', 'education', 'habitation',
 'low_paid_work_abroad', 'ex_prisoner', 'lifestyle_alcoholic',
 'lifestyle_addict', 'lifestyle_smoking', 'job', 'number_of_residents', 'hiv',
 'art', 'type_of_drug_resistance', 'provider_name', 'provider_abbreviation',
 'provider_street_address_line', 'provider_city', 'provider_postal_code',
 'provider_country', 'provider_telecom']

paramTot_drug_resistance=['drug_susceptibility_testing_am', 'drug_susceptibility_testing_amx_clv',
 'drug_susceptibility_testing_cm', 'drug_susceptibility_testing_cs',
 'drug_susceptibility_testing_e', 'drug_susceptibility_testing_h',
 'drug_susceptibility_testing_km', 'drug_susceptibility_testing_lfx',
 'drug_susceptibility_testing_mb', 'drug_susceptibility_testing_mfx',
 'drug_susceptibility_testing_ofx', 'drug_susceptibility_testing_pas',
 'drug_susceptibility_testing_pto', 'drug_susceptibility_testing_r',
 'drug_susceptibility_testing_s', 'drug_susceptibility_testing_z',
 'type_of_drug_resistance', 'provider_name', 'provider_abbreviation',
 'provider_street_address_line', 'provider_city', 'provider_postal_code',
 'provider_country', 'provider_telecom']

paramTot_drug_resistance_bactec_test=['bactec_test_is_performed', 'bactec_test_culture', 'bactec_test_h',
 'bactec_test_ofl', 'bactec_test_z', 'bactec_test_pto', 'bactec_test_r',
 'bactec_test_cm', 'bactec_test_lfx', 'bactec_test_cs', 'bactec_test_s',
 'bactec_test_am', 'bactec_test_mfx', 'bactec_test_amx_clv', 'bactec_test_e',
 'bactec_test_km', 'bactec_test_pas', 'bactec_test_h2', 'provider_name',
 'provider_abbreviation', 'provider_street_address_line', 'provider_city',
 'provider_postal_code', 'provider_country', 'provider_telecom']

paramTot_drug_resistance_hain_test=['hain_test_is_performed', 'hain_test_culture', 'hain_test_h', 'hain_test_ft',
 'hain_test_r', 'hain_test_agcp', 'hain_test_e', 'provider_name',
 'provider_abbreviation', 'provider_street_address_line', 'provider_city',
 'provider_postal_code', 'provider_country', 'provider_telecom']

paramTot_drug_resistance_gene_test=['gene_test_is_performed', 'gene_test_culture', 'gene_test_r', 'provider_name',
 'provider_abbreviation', 'provider_street_address_line', 'provider_city',
 'provider_postal_code', 'provider_country', 'provider_telecom']

paramTot_treatment=['treatment_regimen_am', 'treatment_regimen_amx_clv', 'treatment_regimen_cm',
 'treatment_regimen_cs', 'treatment_regimen_e', 'treatment_regimen_h',
 'treatment_regimen_lfx', 'treatment_regimen_lzd', 'treatment_regimen_mfx',
 'treatment_regimen_ofx', 'treatment_regimen_pas', 'treatment_regimen_pto',
 'treatment_regimen_r', 'treatment_regimen_s', 'treatment_regimen_z',
 'treatment_regimen_km', 'treatment_end_date', 'treatment_result',
 'result_of_previous_treatment', 'surgery_treatment', 'msc_treatment',
 'treatment_regimen_cotrimoxazol', 'provider_name', 'provider_abbreviation',
 'provider_street_address_line', 'provider_city', 'provider_postal_code',
 'provider_country', 'provider_telecom']

paramTot_radiology_protocol=['accumulation_of_contrast', 'affect_level', 'affect_pleura',
 'affected_segments', 'anomaly_development_of_mediastinum_vessels',
 'anomaly_of_lung_develop', 'bronchial_obstruction', 'dissemination',
 'lung_capacity_decrease', 'lung_cavity_size', 'lymphoadenopatia',
 'no_of_cavities', 'nodi_calcinatum', 'plevritis', 'pneumothorax',
 'post_tb_residuals', 'process_prevalence', 'shadow_pattern',
 'side_of_process', 'thromboembolism_of_the_pulmonary_artery',
 'provider_name', 'provider_abbreviation', 'provider_street_address_line',
 'provider_city', 'provider_postal_code', 'provider_country',
 'provider_telecom']

paramTot_comments=['comments']

paramTot_sequences=['sequence_name','Patric','GenBank']


################################
param_demographics=[
    'date_of_birth',
    'gender',
    'clinical_record_id'
]
param_followup=[
    'height_cm',
    'weight_cm',
    'disability_level',
    'symptoms_of_tb',
    'diagnosis',
    'co_morbidity',
    'hiv',
    'art',
    'education',
    'habitation',
    'number_of_residents',
    'job',
    'ex_prisoner',
    'low_paid_work_abroad',
    'lifestyle_alcoholic',
    'lifestyle_smoking',
    'lifestyle_addict',
]
param_treatment=[
    'treatment_result', #???
    'result_of_previous_treatment',
]
param_lab_findings=[
    'smear_result', 'smear_score',
    'culture_result','culture_score',
]
param_drug_resistance=[
    'type_of_drug_resistance',
    'drug_susceptibility_testing_h',
    'drug_susceptibility_testing_r',
    'drug_susceptibility_testing_s',
    'drug_susceptibility_testing_e',
    'drug_susceptibility_testing_ofx',
    'drug_susceptibility_testing_cm',
    'drug_susceptibility_testing_am',
    'drug_susceptibility_testing_km',
    'drug_susceptibility_testing_z',
    'drug_susceptibility_testing_lfx',
    'drug_susceptibility_testing_mfx',
    'drug_susceptibility_testing_pas',
    'drug_susceptibility_testing_pto',
    'drug_susceptibility_testing_cs',
    'drug_susceptibility_testing_amx_clv',
    'drug_susceptibility_testing_mb'
]

param_radiology_protocol=[
    'side_of_process',
    'lung_capacity_decrease',
    'process_prevalence',
    'affect_level',
    'affected_segments',
    'shadow_pattern',
    'lung_cavity_size',
    'no_of_cavities',
    'bronchial_obstruction',
    'dissemination',
    'lymphoadenopatia',
    'nodi_calcinatum',
    'post_tb_residuals',
    'plevritis',
    'affect_pleura',
    'pneumothorax',
    'anomaly_of_lung_develop',
    'anomaly_development_of_mediastinum_vessels'
]

lst_params={
    'demographics':         param_demographics,
    'followup':             param_followup,
    'treatment':            param_treatment,
    'lab_findings':         param_lab_findings,
    'drug_resistance':      param_drug_resistance,
    'radiology_protocol':   param_radiology_protocol
}
################################
class KeyVal:
    def __init__(self, strDate, strParam, strVal):
        self.param=strParam
        self.val=strVal
        self.date=pd.Timestamp(strDate)
        self.dateStr=strDate
    def toString(self):
        return 'KeyVal[%s]={%s: %s}' % (self.date,self.param,self.val)
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()
    def calcDt(self, parKeyVal):
        return (self.date-parKeyVal.date).day

class Record:
    def __init__(self,pidx=-1, pkeys=None):
        self.clean()
        self.id=pidx
        if pkeys is None:
            pkeys=par_keys
        self.data={}
        self.dataf={}
        for kk in pkeys:
            self.data[kk]=[]
            self.dataf[kk]=None
    def getCT(self):
        return self.data['CT']
    def getXR(self):
        return self.data['X-RAY']
    def findXRByDate(self, pdate):
        lstXR=self.getXR()
        if len(lstXR)>0:
            tmp=np.array( [(pdate-ii.date).days for ii in lstXR] )
            tpos=np.argmin(np.abs(tmp))
            return (lstXR[tpos].val, tmp[tpos])
        else:
            return ('', 0)
    def getCTNum(self):
        return len(self.getCT())
    def clean(self):
        self.id=-1
        self.data=None
        self.dataf=None
    def getInfoForCT(self, pctKV, pkeys, isUseDT=True):
        ret=[]
        ret.extend( (self.id, pctKV.dateStr, pctKV.val) )
        # Age
        dataDemogr=self.dataf['demographics']
        tmpDateBirth=dataDemogr[dataDemogr['Param']=='date_of_birth']['Value']
        retAge="-1"
        if len(tmpDateBirth)>0:
            dateBirth=pd.Timestamp(tmpDateBirth.values[0])
            retAge="%0.3f" % np.abs((pctKV.date-dateBirth).days/365.24)
        ret.append(retAge)
        #
        tmp_ret=[]
        tmp_ret_dt=[]
        ret_XR=self.findXRByDate(pctKV.date)
        tmp_ret.append(ret_XR[0])
        if isUseDT:
            tmp_ret_dt.append(ret_XR[1])
        for kk,vv in pkeys.iteritems():
            tdata=self.dataf[kk]
            for pp in vv:
                tpdata=tdata[tdata['Param']==pp]
                arrDT=np.array([ (pctKV.date - pd.Timestamp(ii)).days for ii in tpdata['Date'] ])
                #arrDT[arrDT<0]=10000
                tpos=np.argmin(np.abs(arrDT))
                tval=tpdata['Value'].values[tpos]
                tdt=arrDT[tpos]
                if isUseDT:
                    tmp_ret.append(tval)
                    tmp_ret_dt.append(tdt)
                    # ret.extend((tval, tdt))
                else:
                    # ret.append(tval)
                    tmp_ret.append(tval)
        ret.extend(tmp_ret)
        ret.extend(tmp_ret_dt)
        return ret

def getHeadersCT(pkeys, isUseDT=True):
    ret=['id','date_ct_data', 'path_ct', 'Age', 'path_xr']
    tmp=[]
    tmp_dt=[]
    if isUseDT:
        tmp_dt.append( 'dt_path_xr' )
    for kk,vv in pkeys.iteritems():
        if isUseDT:
            for ii in vv:
                tmp.append(ii)
                tmp_dt.append('dt_%s' % ii)
        else:
            tmp.extend(vv)
    ret.extend(tmp)
    if isUseDT:
        ret.extend(tmp_dt)
    return ret

###########################
if __name__=='__main__':
    if not os.path.isfile(fcsv):
        print >> sys.stderr, 'ERROR: cant find file [%s]' % fcsv
        sys.exit(1)
    #
    dataraw=pd.read_csv(fcsv, header=None, names=["Id","Date","Key","Param","Value"])
    dataIdx=pd.unique(dataraw['Id'])
    dataKey=pd.unique(dataraw['Key'])
    #
    dbData=[]
    for tidx in dataIdx:
        tdatarw=dataraw[dataraw['Id']==tidx]
        trecord=Record(pidx=tidx, pkeys=dataKey)
        for vv in tdatarw.values:
            tkey=vv[2]
            tdata=pd.Timestamp(vv[1])
            tdataStr=tdata.strftime('%Y%m%d')
            if (tkey=='CT') and (vv[3]=='preview'):
                tpath='%d/CT/%s' % (tidx, tdataStr)
                tmpKV=KeyVal(vv[1],'path',tpath)
            elif (tkey=='X-RAY') and (vv[3]=='preview'):
                tpath='%d/X-RAY/%s' % (tidx, tdataStr)
                tmpKV=KeyVal(vv[1],'path',tpath)
            else:
                tmpKV=None #KeyVal(vv[1],vv[3],vv[4])
            if tmpKV is not None:
                trecord.data[tkey].append(tmpKV)
        for kk in dataKey:
            trecord.dataf[kk]=tdatarw[tdatarw['Key']==kk]
        dbData.append(trecord)
        if (int(tidx)%20)==0:
            print '%s/%d' % (tidx, len(dataIdx))
    #
    dataColumns=getHeadersCT(lst_params,isUseDT=isUseDT)
    print dataColumns
    #
    dataArr=[]
    for ii,rr in enumerate(dbData):
        for cc in rr.getCT():
            tmp=rr.getInfoForCT(cc,lst_params,isUseDT=isUseDT)
            dataArr.append(tmp)
        if (ii%20)==0:
            print '%d/%d' % (ii,len(dbData))
    dataOut=pd.DataFrame(dataArr, columns=dataColumns)
    dataOut.to_csv(fcsvOUT_CT, sep="|", index=False)
