import knn as knn
import numpy as np

components=[5,10,20,50,100,150,200,500,1000]
components=[5,10,20,50,100,150,200]
#components=[300,400,500]
components=[500,1000]
components=[1,5,10,20,40,60,80,100,125,150,175,200,250,300]
components=[1,5,10,20,40,60,80,100,125,150,175,200,250,300,500,1000]


#components=[1,5]
#components=[1,5,10,20,40,60,80,100,125,150]
#components=[5,10,20]

val=np.zeros(len(components))

for c1,c in list(enumerate(components)):
    print(c1)
    val[c1]=knn.nn(c)
    
np.save('./output/knn_val3.npy',val)
