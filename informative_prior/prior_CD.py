#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 14:58:12 2021

@author: zitongzhou
"""

from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
import pandas as pd
from smt.sampling_methods import LHS
from sklearn.neural_network import MLPRegressor
from matplotlib.patches import Rectangle
import shutil
from scipy.interpolate import Rbf
from scipy import interpolate
import time


##set ploting properties
SMALL_SIZE = 12
MEDIUM_SIZE = 14
BIGGER_SIZE = 16
plt.rc('font',**{'family':'serif','serif':['Times']})
plt.rcParams['text.usetex'] = True
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SMALL_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)

'''
Two methods for generation of informative prior distribution for C and D from 
the data are provided here, corresponding to those in the paper.
1. Connectivity informed prior;
2. Field data informed prior.
'''

'''1. Prior with the information of the generated DFN conncectivity, 20 cases 
considered for each pair of (C, D).'''
with open(
        '/Users/zitongzhou/Desktop/FRACTURE/correlation/uncorr_100_20pdf_conn.pkl',
        'rb'
        ) as file:
    [C_conn, D_conn, num_conn] = pkl.load(file)


## interpolate with the connectivity information takes a long time since the 
## data is 2D. The results are not smooth as there are many noisy datapoints.
# start = time.time()
# # tck = interpolate.bisplrep(C_conn, D_conn, num_conn, s=0)
# # rbf = Rbf(C_conn, D_conn, num_conn, epsilon=2)
# interp = interpolate.interp2d(C_conn, D_conn, num_conn,kind='linear')
# print('interpolation took', time.time()-start,'seconds')

# Cnew = np.linspace(2.5, 6.5, 10000)
# # D = k*C +b
# Dnew = np.linspace(1.0, 1.3, 1000)
# znew = interp(Cnew, Dnew)
# Cnew, Dnew = np.meshgrid(Cnew, Dnew)
# znew[znew>20] = 20
# znew[znew<num_conn.min()] = num_conn.min()


## train a simple MLP to do the interpolation task.
cd_train = np.array([C_conn, D_conn]).T
y_train = np.reshape(num_conn, (-1))
regr = MLPRegressor(hidden_layer_sizes=(100,100,100,), random_state=1, max_iter=5000).fit(cd_train, y_train)
print(regr.loss_)
os.chdir("/Users/zitongzhou/Desktop/FRACTURE/correlation/")
with open('conn_regr.pkl', 'wb') as file:
    pkl.dump(regr, file)

C = np.linspace(2.5, 6.5, 100)
D = np.linspace(1.0, 1.3, 100)
C, D = np.meshgrid(C, D)
mesh_shape = C.shape
C, D = np.reshape(C,(-1,)), np.reshape(D, (-1))

znew = regr.predict(np.array([C, D]).T)
znew = np.reshape(znew, mesh_shape)
znew = np.round(znew)
znew = znew/(np.sum(znew*4/99*0.3/99))


## plot the connectivity scatter plot, and the interpolation.
fig, axs = plt.subplots(1,2, figsize=(8, 4))
axs = axs.flat
ax = axs[0]
c01map = ax.scatter(C_conn, D_conn, c=num_conn, cmap='jet', marker=",", s = 0.5,
                    # interpolation='nearest',
          # extent=[C_d[:,0].min(), C_d[:,0].max(), C_d[:,1].min(), C_d[:,1].max()],
          vmin=num_conn.min(), vmax = num_conn.max(),
          # origin='lower',aspect='auto'
          )
ax.set_xlim(2.5, 6.5)
ax.set_xticks(np.linspace(2.5, 6.5, 5, endpoint=True))
ax.set_ylim(1.0, 1.3)
ax.set_xlabel('C [-]')
ax.set_ylabel('D [-]')
v1 = np.arange(np.min(num_conn), np.max(num_conn)+1)
fig.colorbar(c01map, ax=ax, fraction=0.08, pad=0.03,ticks=v1,)

ax = axs[1]
c01map = ax.imshow(znew, cmap='jet', interpolation='nearest',
          extent=[C.min(), C.max(), D.min(), D.max()],
          vmin=znew.min(), vmax = znew.max(),
          origin='lower',aspect='auto')
ax.set_xlim(2.5, 6.5)
ax.set_xticks(np.linspace(2.5, 6.5, 5, endpoint=True))
ax.set_ylim(1.0, 1.3)
ax.set_xlabel('C [-]')
# ax.set_ylabel('D [-]')
v2 = np.linspace(znew.min(), znew.max(), num=len(v1), endpoint=True)
fig.colorbar(c01map, ax=ax, fraction=0.08, pad=0.03,ticks=v2,)
plt.tight_layout()
fig.savefig('conn_prior_density.pdf', format='pdf',bbox_inches='tight')
plt.show()


'''
2. Prior with the field data (C,D) pairs.
'''
CD_data =pd.read_csv("Classeur1.csv")


'''KDE estimation to obtain the prior, (C,D) in [0,10]x[0.25, 2]'''
C_D = CD_data[['C', 'D']].to_numpy()

C_D = np.asarray([
        [float(C_D[i][0]), float(C_D[i][1])] 
        for i in range(len(C_D)) 
        if is_number_repl_isdigit(C_D[i][0])
      ])

included_C_D = np.asarray([C_D[i] for i in range(len(C_D))
                if C_D[i,0] <= 15 and C_D[i,0] > 0.0
                and C_D[i,1] >= 0.
                ])
ymin, ymax=0.25, 2.1
xmin, xmax = 0, 15

X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:1000j]
positions = np.vstack([X.ravel(), Y.ravel()])
values =included_C_D
kernel = stats.gaussian_kde(values.T)
Z = np.reshape(kernel(positions).T, X.shape)

ymin1, ymax1 =1.0, 1.3
xmin1, xmax1 = 2.5, 6.5
X1, Y1 = np.mgrid[xmin1:xmax1:100j, ymin1:ymax1:1000j]
positions1 = np.vstack([X1.ravel(), Y1.ravel()])
Z1 = np.reshape(kernel(positions1).T, X1.shape)

fig, axs = plt.subplots(1,2, figsize=(7, 3),constrained_layout=True)

cmap = axs[0].imshow(np.rot90(Z), cmap=plt.cm.rainbow,
           extent=[xmin, xmax, ymin, ymax],
           aspect='auto', )
axs[0].add_patch(Rectangle((2.5, 1.0), 4.0, 0.3, fc='none', ec='w'))
axs[0].text(3.5, 1.1, 'prior region')
axs[0].plot(included_C_D[:,0], included_C_D[:, 1], 'k*', markersize=4, label='data')
axs[0].set_xlim([xmin, xmax])
axs[0].set_ylim([ymin, ymax])
fig.colorbar(cmap, ax=axs[0],aspect=30)
axs[0].set_aspect('auto')
cmap1 = axs[1].imshow(np.rot90(Z1), cmap=plt.cm.rainbow,
           extent=[xmin1, xmax1, ymin1, ymax1],
           aspect='auto',)
axs[1].set_xlim([xmin1, xmax1])
axs[1].set_ylim([ymin1, ymax1])

fig.colorbar(cmap1, ax = axs[1], aspect=30, pad=0.005)
axs[1].set_aspect('auto')
axs[0].legend(loc='lower right')
axs[0].set_ylabel('D [-]')
axs[0].set_xlabel('C [-]')
axs[1].set_ylabel('D [-]')
axs[1].set_xlabel('C [-]')
axs[1].set_xticks(np.linspace(2.5, 6.5, 5, endpoint=True))
plt.show()

# fig.savefig('prior.pdf', bbox_inches='tight')
filename = '/Users/zitongzhou/Desktop/FRACTURE/correlation/kde.pkl'
with open(filename,'wb') as file:
    pkl.dump(kernel, file)

