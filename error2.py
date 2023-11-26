#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 25 00:48:00 2019

@author: neuro
"""

import numpy as np
import nibabel as nib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
K=150

dataobject_x=nib.load('./alllagtimes_1200.nii.gz')
datamemory_x=dataobject_x.get_data()
dim_0=datamemory_x.shape[0]
dim_1=datamemory_x.shape[1]
dim_2=datamemory_x.shape[2]
dim_3=datamemory_x.shape[3]
header_x=dataobject_x.get_header()
affine_x=dataobject_x.affine
subject_num=dim_3

dataobject_x=nib.load('./s1000.nii.gz')
datamemory_x4000=dataobject_x.get_data()



#error=(np.mean(np.square(datamemory_x))-np.mean(np.square(datamemory_x4000)))/np.mean(np.square(datamemory_x))
error2=datamemory_x-datamemory_x4000
s=error2.reshape([-1,subject_num])

dataobject_x=nib.load('./ASPECTS_HCP.nii.gz')
dataobject_x=nib.load('./ASPECTS_HCP.nii.gz')
datamemory_x=dataobject_x.get_data()
dim_0=datamemory_x.shape[0]
dim_1=datamemory_x.shape[1]
dim_2=datamemory_x.shape[2]
header_x=dataobject_x.get_header()
affine_x=dataobject_x.affine

#s1=s1[:,0:5]
#s2=s2[:,0:5]

s3=datamemory_x.reshape([-1,1])
index3=np.where(s3!=0)

s_m=s[index3[0],:]

s_a=np.mean(s_m,axis=0)


#plt.hist(s_a,60, density=True, facecolor='g')
plt.hist(s_a,60, facecolor='g')
#plt.axvline(x=1,linewidth=4, color='r')
plt.xlabel('Error',fontsize=16)
plt.ylabel('Number of Voxels',fontsize=17)
plt.legend([ 'Error'],fontsize=13)
#font = {'family': 'serif',
#        'color':  'darkred',
#        'weight': 'normal',
#        'size': 16,
#        }
#plt.xlim(-0.5,3)
#plt.text(-0.49, 8.2, r'Runs Where Deep Learning Prediction is Better', fontdict=font,fontsize=18)
#plt.text(1.2, 8.2, r'Runs Where Raw Prediction is Better', fontdict=font,fontsize=18)
#plt.show()
plt.savefig('./ErrorHist1000.png',dpi=1000)
#plt.savefig('./output/ErrorHist.png',dpi=1000)
