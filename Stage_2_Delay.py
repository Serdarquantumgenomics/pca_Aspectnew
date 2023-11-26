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

s_m = np.load('Stage1_Delay3000.npy')
K=3000
thepca = PCA(n_components=int(K))
thefit=thepca.fit(np.transpose(s_m))
data_projected=thepca.transform(np.transpose(s_m))

np.save('Stage2_Delay3000.npy',data_projected)
