import numpy as np 
import scipy.stats as st 
import sys
sys.path.insert(0, '../')
from GP_integral import *
import kernel_py as kp
import chaos_toolbox as ct
import matplotlib.pyplot as plt 


X = np.load('X.npy')
Y = np.load('Y.npy')
Q = 5

var = np.load('sig_smooth.npy')[-1]
ell = np.load('ell_smooth.npy')[-2]

rbf = kp.RBF(X.shape[1], var, ell)


I = IntegralPosterior(X, Y, Q, rbf)
[m, C] = I.predict('L')
#v_all = I.v_all('L')
#print v_all
#np.save('coeffs/v_all.npy', v_all)
print m

plt.plot(m)
plt.show()

pol = ct.chaos.PolyBasis(2, Q, 'L')
xi = st.norm.rvs(size = (1000,2))
P = pol(xi)
y = np.dot(P, m)
plt.hist(y, bins = 100, normed = True)
plt.show()