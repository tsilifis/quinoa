import numpy as np 
import kernel_py as kp
import GPy as gpy

xx, yy = np.mgrid[-3:3:30j, -3:3:30j]

X = np.vstack([xx.flatten(), yy.flatten()]).T
Y = X

ker1 = kp.RBF(2, 1, [1,2])
ker2 = gpy.kern.RBF(2, 1, [1,2], ARD = True)

C1 = ker1.cov(X)
#C2 = ker2.K(X)
C2 = ker1.cov(X, Y)

print np.linalg.norm(C1 - C2)
import matplotlib.pyplot as plt 

fig = plt.figure(figsize = (8,4))
ax1 = fig.add_subplot(121)
ax1.contourf(C1, 40)
ax2 = fig.add_subplot(122)
ax2.contourf(C2, 40)
plt.show()