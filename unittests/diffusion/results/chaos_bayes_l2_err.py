import numpy as np 
import kernel_py as kp
import matplotlib.pyplot as plt 
import chaos_toolbox as ct
import sys
sys.path.insert(0, '../../')
from GP_integral import *

X = np.load('../U_X.npy')
Y = np.load('../U_Y.npy')

var = np.load('../U_sig_smooth_300.npy')
ell = np.load('../U_ell_smooth_300.npy')

[x_quad, w] = ct.util.QuadratureRule('CC').get_rule(2, 8)
y_quad = np.load('../data_concentrations_CC_quadrature_lev8.npy')[:,-1]
#coeffs = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, 9)


#l2_err = np.zeros((30, 17))

#for i in range(23,l2_err.shape[0]):
#	for j in range(l2_err.shape[1]):
#		Q = 4+j
#		X_i = X[:10*(i+1)+4,:].copy()
#		Y_i = Y[:10*(i+1)+4,:].copy()
	
#		rbf = kp.RBF(X.shape[1], var[10*(i+1)], ell[10*(i+1)])

#		I = IntegralPosterior(X_i, Y_i, Q, rbf)
#		[m, C] = I.predict('L')
		
#		pol = ct.chaos.PolyBasis(2, Q, 'L')
#		P = pol(x_quad)
#		y_pred = np.dot(P, m)
#		diff = (y_quad - y_pred)**2
#		l2_err[i,j] = np.abs(np.sum(diff * w)) / np.sum(y_quad**2 * w)
#		print 'order :' + str(j) 
#	print i

y_axis = [10*(i+1) for i in range(30)]
x_axis = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
#np.save('./images/l2_err_2.npy', l2_err)
l2_err = np.load('./images/l2_err_1.npy')
#print l2_err

CS = plt.contour(x_axis, y_axis, np.log(l2_err), 12)
plt.clabel(CS, inline = 1, fmt = '%1.1f',fontsize = 10)
plt.xlabel('PC order', fontsize = 12)
plt.ylabel('# abscissae', fontsize = 12)
plt.title('Log l2 error', fontsize = 12)
plt.savefig('images/l2_error_bayes_U.ps')
plt.savefig('images/l2_error_bayes_U.png')
plt.show()