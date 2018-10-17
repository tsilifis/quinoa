import numpy as np 
import kernel_py as kp
import matplotlib.pyplot as plt 
import chaos_toolbox as ct
import sys
sys.path.insert(0, '../../')
from GP_integral import *

X = np.load('../UU_X.npy')[:290,:]
Y = np.load('../UU_Y.npy')[:290,:]

var = np.load('../UU_sig.npy')[:290]
ell = np.load('../UU_ell.npy')[:290]

[x_quad, w] = ct.util.QuadratureRule('CC').get_rule(5, 6)
y_quad = np.load('../data_pressure_CC_quadrature_lev6.npy')[:,-1]
#coeffs = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, 9)


l2_err = np.zeros((6, 8))

for i in range(l2_err.shape[0]):
	for j in range(l2_err.shape[1]):
		Q = 2+j
		X_i = X[:50*i+40,:].copy()
		Y_i = Y[:50*i+40,:].copy()
	
		rbf = kp.RBF(X.shape[1], var[50*i+20], ell[50*i+20])

		I = IntegralPosterior(X_i, Y_i, Q, rbf)
		[m, C] = I.predict('L')
		
		pol = ct.chaos.PolyBasis(5, Q, 'L')
		P = pol(x_quad)
		y_pred = np.dot(P, m)
		diff = (y_quad - y_pred)**2
		l2_err[i,j] = np.abs(np.sum(diff * w)) / np.sum(y_quad**2 * w)
		print 'order :' + str(j) 
	print i

y_axis = [50*i for i in range(6)]
x_axis = [2,3,4,5,6,7,8,9]
np.save('./images/l2_err.npy', l2_err)
#l2_err = np.load('./images/l2_err_1.npy')
#print l2_err

CS = plt.contour(x_axis, y_axis, np.log(l2_err), 12)
plt.clabel(CS, inline = 1, fmt = '%1.1f',fontsize = 10)
plt.xlabel('PC order', fontsize = 12)
plt.ylabel('# abscissae', fontsize = 12)
plt.title('Log l2 error', fontsize = 12)
plt.savefig('images/l2_error_bayes_U.ps')
plt.savefig('images/l2_error_bayes_U.png')
plt.show()