import nn as nn
import numpy as np

components=[5,10,20,50,100,150,200,500,1000]
components=[1,5,10,20,40,60,80,100,125,150,175,200,250,300,500,1000]

val=np.zeros(len(components))

for c1,c in list(enumerate(components)):
    val[c1]=nn.nn(c)
    
np.save('./output/nn_valA.npy',val)
