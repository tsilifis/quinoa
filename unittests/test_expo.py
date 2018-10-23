import numpy as np 
import quinoa as qu
import GPy as gpy

xx, yy = np.mgrid[-3:3:30j, -3:3:30j]

X = np.vstack([xx.flatten(), yy.flatten()]).T

ker1 = qu.Exponential(2,1,1)
ker2 = gpy.kern.Exponential(2,1, [1, 1], ARD = True)

C1 = ker1.cov(X)
C2 = ker2.K(X)
print np.linalg.norm(C1 - C2)

import matplotlib.pyplot as plt 

fig = plt.figure(figsize = (8,4))
ax1 = fig.add_subplot(121)
ax1.contourf(C1, 40)
ax2 = fig.add_subplot(122)
ax2.contourf(C2, 40)
plt.show()