#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 09:09:01 2018

@author: neuro
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
K=4000

dataobject_x=nib.load('./alllagtimes_1200.nii.gz')
datamemory_x=dataobject_x.get_data()
dim_0=datamemory_x.shape[0]
dim_1=datamemory_x.shape[1]
dim_2=datamemory_x.shape[2]
dim_3=datamemory_x.shape[3]
header_x=dataobject_x.get_header()
affine_x=dataobject_x.affine
subject_num=dim_3

s=datamemory_x.reshape([-1,subject_num])

dataobject_mask=nib.load('./globalmask_1200_thr.nii.gz')
datamemory_mask=dataobject_mask.get_data()
dim_0=datamemory_mask.shape[0]
dim_1=datamemory_mask.shape[1]
dim_2=datamemory_mask.shape[2]
header_x=dataobject_mask.get_header()
affine_x=dataobject_mask.affine

s3=datamemory_mask.reshape([-1,1])

index3=np.where(s3!=0)

s_m=s[index3[0],:]
s3_m=s3[index3[0],:]


thepca = PCA(n_components=int(K))
thefit=thepca.fit(np.transpose(s_m))
data_projected=thepca.transform(np.transpose(s_m))
data_backprojected=thepca.inverse_transform(data_projected)

np.save('./output/Stage1_Delay4000.npy',data_backprojected)
