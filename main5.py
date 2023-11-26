import knn_confusion  as knn
import numpy as np


components=[300]


val=np.zeros(len(components))

for c1,c in list(enumerate(components)):
    print(c1)
    val[c1]=knn.nn(c)
    
