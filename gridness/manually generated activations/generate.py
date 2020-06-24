import numpy as np
import matplotlib.pyplot as plt
from sac import calculate_sac, masked_sac

def generate_vertex(v0,v1,v2,ran):
    '''
    Generate vertex of grid, starting from v0 and spanning in 
    v1 and v2 direction according to range ran=(min,max)'''
    vertex = []
    for i in range(ran[0],ran[1]):
        for j in range(ran[0],ran[1]):
            nv = v0+i*v1+j*v2
            vertex.append(nv)
    vertex = np.array(vertex)
    return vertex

def generate_activation(centers, res=40, sigmax=0.12, sigmay=0.12):
    grid = np.linspace(-1.1, 1.1, res)
    act = np.zeros((res,res))
    for i in range(len(centers)):
        gridx = grid - centers[i,0]
        gridy = grid + centers[i,1]
        x = np.exp(-gridx**2/(sigmax**2))
        y = np.exp(-gridy**2/(sigmay**2))
        g = x[None,:]*y[:,None]
        act +=g
    return act

def generate_masked_sac(v0,v1,v2,ran, res=40, sigmax=0.12, sigmay=0.12, r=19):
    vertex = generate_vertex(v0,v1,v2,ran)
    act = generate_activation(vertex, res, sigmax, sigmay)
    sac = calculate_sac(act)
    return masked_sac(sac,r)
    
    