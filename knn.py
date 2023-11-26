import numpy as np
#import matplotlib.pyplot as plt
import nibabel as nib
import matplotlib.pyplot as plt
from keras.utils import to_categorical

from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score

def nn(K):
    #s1_m=np.load('Stage2_Strength.npy')
    s2_m=np.load('./output/Stage2_Delay'+str(K)+'.npy')
    #s4_m=np.load('Stage2_Sigma.npy')
    
    dataobject_x=nib.load('./ASPECTS_HCP.nii.gz')
    datamemory_x=dataobject_x.get_data()
    dim_0=datamemory_x.shape[0]
    dim_1=datamemory_x.shape[1]
    dim_2=datamemory_x.shape[2]
    header_x=dataobject_x.get_header()
    affine_x=dataobject_x.affine
    
    s3=datamemory_x.reshape([-1,1])
    
    
    dataobject_xg=nib.load('./globalmask_1200_thr.nii.gz')
    datamemory_xg=dataobject_xg.get_data()
    dim_0g=datamemory_xg.shape[0]
    dim_1g=datamemory_xg.shape[1]
    dim_2g=datamemory_xg.shape[2]
    header_xg=dataobject_xg.get_header()
    affine_xg=dataobject_xg.affine

    s3g=datamemory_xg.reshape([-1,1])

    index3=np.where((s3g!=0 ))
    s3_Ag=s3[index3[0],:]

    index3=np.where(s3_Ag!=0)
    s3_m=s3_Ag[index3[0],:]
    s2_m=s2_m[index3[0],:]


    
    train_yt = to_categorical(s3_m)
    
    #s1_t=np.concatenate((s1_m,s2_m,s4_m),axis=1)
    #s1_t=np.concatenate((s1_m,s4_m),axis=1)
    s1_t=s2_m
    #s1_t=s1_m
    #s1_t=s4_m
    
    
    
    train_xt = s1_t
    print('x: ',train_xt.shape)                    
    print('y: ',train_yt.shape)                    
    
    train_x, test_x, train_y, test_y = train_test_split( train_xt, train_yt, test_size=0.2, random_state=42)
    train_x2, test_x2, train_y2, test_y2 = train_test_split( train_xt, s3_m, test_size=0.2, random_state=42)
    
    
    from sklearn.neighbors import KNeighborsClassifier
    
    kNN=KNeighborsClassifier()
    kNN.fit(train_x, train_y)
    preds_kNN = kNN.predict(test_x)
    
    
    print('kNN',accuracy_score(test_y, preds_kNN))
    np.save('./output/knn_small'+str(K)+'.npy',accuracy_score(test_y, preds_kNN))
    return accuracy_score(test_y, preds_kNN)
