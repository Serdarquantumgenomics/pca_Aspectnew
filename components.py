import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def comp_extractor(K):
    s_m = np.load('./output/Stage1_Delay4000.npy')
    
    thepca = PCA(n_components=int(K))
    thefit=thepca.fit((s_m))
    comp=thefit.components_[:K]
    np.save('./output/components300.npy',comp)

K=300
comp_extractor(K)
