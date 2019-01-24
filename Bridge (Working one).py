import numpy as np
from scipy.stats import norm
import sobol_seq as sob

################################################
def sobol(dim,seed): # p = dimension , seed = rank
    return list(sob.i4_sobol(dim,seed)[0])
def phi(x,y): # midpoint index
    if int((x-y)/2) == 0:
        return 0
    else:
        return y + int((x-y)/2) 
def alternate(l_max,l_min):
    z = [l_max[0]]
    for i in range(len(l_min)):
        z = z + [l_min[i]] + [l_max[i+1]]
    return z
def discretize(lst): #next discretization of the index list
    h=[]
    nxt = [phi(lst[i+1],lst[i]) for i in range(len(lst)-1)]
    for i in range(len(alternate(lst,nxt))):
        if alternate(lst,nxt)[i] != 0 or i==0 :
            h+=[alternate(lst,nxt)[i]]
    return h
def ninv(x):
    return norm.ppf(x)
################################################
def generate_sobol_bridge(T,seed):
    sample = sobol(len(T),seed)
    b_start = ninv(sample[1])*np.sqrt(T[0])
    b_end = b_start + ninv(sample[0])*np.sqrt(T[1]-T[0])
    del sample[0:2]

    G = [[0,len(T)-1],[b_start,b_end]]
    while len(G[0]) != len(T):
        new_brownian = []
        print(sample)
        nz = 0
        for k in range(len(G[0])-1):
            if phi(G[0][k+1],G[0][k]) == 0:
                new_brownian = new_brownian + [0]
            else:
                nz+=1
                v = T[phi(G[0][k+1], G[0][k])]
                a = G[1][k]
                b = G[1][k+1] 
                w = T[G[0][k+1]] 
                u = T[G[0][k]]
                dt = w-u
                mu = a*(w-v)/dt + b*(v-u)/dt
                var = (v-u)*(w-v)/dt
                x = mu + np.sqrt(var)*ninv(sample[k])
                new_brownian = new_brownian + [x]
        G[0] = discretize(G[0])
        G[1] = [x for x in alternate(G[1],new_brownian) if x != 0]
        del sample[0:nz-1]
    return G[0]
#################################################
def generate_regular_bridge(T , steps):
    b_start = np.random.normal(0,1,1)*np.sqrt(T[0])
    b_end = b_start + np.random.normal(0,1,1)*np.sqrt(T[1]-T[0])
    G = [[0,len(T)-1],[b_start,b_end]]
    while len(G[0]) != len(T):
        new_brownian = []
        for k in range(len(G[0])-1):
            if phi(G[0][k+1],G[0][k]) == 0:
                new_brownian = new_brownian + [0]
            else:
                v = T[phi(G[0][k+1], G[0][k])]
                a = G[1][k]
                b = G[1][k+1] 
                w = T[G[0][k+1]] 
                u = T[G[0][k]]
                dt = w-u
                mu = a*(w-v)/dt + b*(v-u)/dt
                var = (v-u)*(w-v)/dt
                x = np.random.normal(mu,var,1)[0]
                new_brownian = new_brownian + [x]
        G[0] = discretize(G[0])
        G[1] = [x for x in alternate(G[1],new_brownian) if x != 0]
    return G[1]
##################################################



