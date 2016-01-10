#!/usr/bin/python
__author__ = 'ar'

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

fcsv='/home/ar/tmp/CRDF/clinical_records_20160107_140558.unzip/cases.csv'

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
    def toString(self):
        return 'KeyVal[%s]={%s: %s}' % (self.date,self.param,self.val)
    def __str__(self):
        return self.toString()
    def __repr__(self):
        return self.toString()

class Record:
    def __init__(self,idx=-1, pkeys=None):
        self.clean()
        if pkeys is None:
            pkeys=par_keys
        self.data={}
        for kk in pkeys:
            self.data[kk]=[]
    def getCT(self):
        return self.data['CT']
    def getCTNum(self):
        return len(self.getCT())
    def clean(self):
        self.id=-1
        self.data=None

class DBData:
    def __init__(self):
        self.clean()
    def clean(self):
        self.arr=None

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
    print '-----'
    foutCSV='%s_dump_tot.txt' % fcsv
    with open(foutCSV,'w') as f:
        for tkk in dataKey:
        # for tkk in lst_params.keys():
            tlstParams=dataraw[dataraw['Key']==tkk]
            tlstParamsUniq=pd.unique(tlstParams['Param'])
            # tlstParamsUniq=lst_params[tkk]
            print '----- [%s] -----' % tkk
            for tpp in tlstParamsUniq:
                tlstValues=pd.unique( tlstParams[ tlstParams['Param']==tpp]['Value'] )
                if len(tlstValues)<9:
                    print '\t %s : %s' % (tpp, tlstValues)
                else:
                    print '\t (!!!) %s : %s ...' % (tpp, tlstValues[:9])
                strOut='%s.%s : %s\n' % (tkk, tpp, tlstValues.tolist())
                f.write(strOut)
            print '*'
