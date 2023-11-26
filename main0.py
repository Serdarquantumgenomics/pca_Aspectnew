import delaynpy as d


#components=[5,10,20,50,100,150,200,500,1000]
components=[4000]

for c1,c in list(enumerate(components)):
    d.delaynpy(c)

