import numpy as np
import nibabel as nib

dataobject_mask=nib.load('./globalmask_1200_thr.nii.gz')
datamemory_mask=dataobject_mask.get_data()
dim_0=datamemory_mask.shape[0]
dim_1=datamemory_mask.shape[1]
dim_2=datamemory_mask.shape[2]
header_x=dataobject_mask.get_header()
affine_x=dataobject_mask.affine

s3=datamemory_mask.reshape([-1,1])

index3=np.where(s3!=0)


c=np.load('./output/components300.npy')
K=300
eig=np.zeros([902629,K])
eig[index3[0],:]=c.transpose()

data_matrix=eig.reshape([dim_0,dim_1,dim_2,K])
new=nib.Nifti1Image(data_matrix,affine_x)

nib.save(new,'./output/eig_comp'+str(K)+'.nii.gz')

