import numpy as np 
import quinoa as qu
import GPy as gpy

xx, yy = np.mgrid[-3:3:30j, -3:3:30j]

X = np.vstack([xx.flatten(), yy.flatten()]).T
Y = X

ker1 = qu.RBF(2, 2, 1)

C1 = ker1.d_cov_d_l(X)
print C1.shape
C2 = ker1.d_cov_d_l(X, Y)
print C2.shape

#print C1[0,:,:]-C2[0,:,:]

import matplotlib.pyplot as plt 

fig = plt.figure(figsize = (8,4))
ax1 = fig.add_subplot(121)
ax1.contourf(C1[:,:], 40)
ax2 = fig.add_subplot(122)
ax2.contourf(C2[0,:,:], 40)
plt.show()