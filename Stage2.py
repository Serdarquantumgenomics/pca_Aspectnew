import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

def Stage2(K):
    s_m = np.load('./output/Stage1_Delay4000.npy')
    #K=100
    print(K)
    thepca = PCA(n_components=int(K))
    thefit=thepca.fit(np.transpose(s_m))
    data_projected=thepca.transform(np.transpose(s_m))
    
    np.save('./output/Stage2_Delay'+str(K)+'.npy',data_projected)
