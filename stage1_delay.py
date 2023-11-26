import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def s_gen(K):
    from sklearn.decomposition import PCA
    
    s_m = np.load('./output/Stage1_Delay4000.npy')
    
    K=4000
    
    thepca = PCA(n_components=int(39))
    thefit=thepca.fit(np.transpose(s_m))
    data_projected=thepca.transform(np.transpose(s_m))
    
    np.save('./output/Stage1_Delay'+str(K)+'.npy',data_projected)
