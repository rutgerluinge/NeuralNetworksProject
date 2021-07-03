from learn_esn import train_esn
from ESN import ESN
import matplotlib.pyplot as plt
import gc
import numpy as np
import pandas as pd
import os

BASEDIR = os.path.abspath(os.path.dirname(__file__))

INPUT_LENGTH = 500

lr_min = 0.2
lr_max = 1
lr_step = 0.2

W_min = 0.1
W_max = 2.1
W_step = 0.5

Win_min = 0.1
Win_max = 2.1
Win_step = 0.5

rs_min = 2000
rs_max = 4000
rs_step = 500

alph_min = 0.1
alph_max = 2.1
alph_step = 0.5


def optimize():
    df = pd.read_csv(BASEDIR + '/../datasets/processed/processed.csv')
    data = df['usage'][:4768].copy()
    data2 = df['usage'][:5200].copy()
    del df
    gc.collect()
    MSE = 0
    output_MSE = []
    output_lr =[]
    output_W =[]
    output_Win =[]
    output_rs =[]
    output_alph =[]
    for i in np.arange(lr_min, lr_max, lr_step):
        for j in np.arange(W_min, W_max, W_step):
            for k in np.arange(Win_min, Win_max, Win_step):
                for l in range(rs_min, rs_max, rs_step):
                    for m in np.arange(alph_min, alph_max, alph_step):
                        for n in range(3):
                            esn = ESN(1, l, 1, leaking_rate = i, Wscalar = j, WinScalar = k)
                            train_esn(esn, data, INPUT_LENGTH, alpha = m)
                            for p in range(len(data), len(data2)-1):
                                MSE += (esn.get_output(data2[p]) - data2[p+1])**2
                        MSE = MSE/2160 #average over 5 runs
                        if len(output_MSE) == 0 or MSE > output_MSE[len(output_MSE)-1]:
                            output_MSE.append(MSE)
                            output_lr.append(i)
                            output_W.append(j)
                            output_Win.append(k)
                            output_rs.append(l)
                            output_alph.append(m)
                        else:
                            for q in range(len(output_MSE)):
                                if MSE<output_MSE[q]:
                                    output_MSE.insert(q, MSE)
                                    output_lr.insert(q, i)
                                    output_W.insert(q, j)
                                    output_Win.insert(q, k)
                                    output_rs.insert(q, l)
                                    output_alph.insert(q, m)
                                    break
                        MSE = 0
                    print("check")


    for r in range(10):
        print(str(output_MSE[r]) + ", " + str(output_lr[r]) + ", " + str(output_W[r]) + ", " + str(output_Win[r]) + ", " + str(output_rs[r]) + ", " + str(output_alph[r]))
                                


                                
                                                      
                                
                    








if __name__ == '__main__':
    optimize()
