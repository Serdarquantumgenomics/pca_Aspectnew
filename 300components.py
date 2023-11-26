import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def comp_extractor(K):
    s_m = np.load('./output/Stage1_Delay4000.npy')
    
    thepca = PCA(n_components=int(K))
    thefit=thepca.fit((s_m))
    comp=thefit.components_[:5]
    np.save('./output/components.npy',comp)

comp_extractor(5)
