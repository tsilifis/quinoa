import numpy as np
import chaos_toolbox as ct
import matplotlib.pyplot as plt 


[x_quad3, w3] = ct.util.QuadratureRule('CC').get_rule(2, 4)
[x_quad4, w4] = ct.util.QuadratureRule('CC').get_rule(2, 5)
[x_quad5, w5] = ct.util.QuadratureRule('CC').get_rule(2, 6)
[x_quad6, w6] = ct.util.QuadratureRule('CC').get_rule(2, 7)
[x_quad7, w7] = ct.util.QuadratureRule('CC').get_rule(2, 8)
quad_points = [x_quad3] + [x_quad4] + [x_quad5] + [x_quad6] + [x_quad7]
quad_wghts = [w3] + [w4] + [w5] + [w6] + [w7]
y_quad3 = np.load('../data_concentrations_CC_quadrature_lev4.npy')[:,-1]
y_quad4 = np.load('../data_concentrations_CC_quadrature_lev5.npy')[:,-1]
y_quad5 = np.load('../data_concentrations_CC_quadrature_lev6.npy')[:,-1]
y_quad6 = np.load('../data_concentrations_CC_quadrature_lev7.npy')[:,-1]
y_quad7 = np.load('../data_concentrations_CC_quadrature_lev8.npy')[:,-1]
y_quads = [y_quad3] + [y_quad4] + [y_quad5] + [y_quad6] + [y_quad7]

l2_err = np.zeros((5, 17))
for i in range(l2_err.shape[0]):
	for j in range(l2_err.shape[1]):
		Q = 4+j
		[x, w] = quad_points[i], quad_wghts[i]
		y = y_quads[i]
		c = ct.chaos.PolyChaos().comp_coeffs(x, y, w, Q, 'L')
		pol = ct.chaos.PolyBasis(2, Q, 'L')
		P = pol(x_quad7)
		y_pred = np.dot(P, c)
		diff = (y_quad7 - y_pred)**2
		l2_err[i,j] = np.abs(np.sum(diff * w7)) / np.sum(y_quad7**2 * w7)

y_axis = [quad_points[i].shape[0] for i in range(5)]
x_axis = [4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

print l2_err.shape

CS = plt.contour(x_axis, y_axis, np.log(l2_err), 12)
plt.clabel(CS, inline = 1, fmt = '%1.1f', fontsize = 10)
plt.xlabel('PC order', fontsize = 12)
plt.ylabel('# abscissae', fontsize = 12)
plt.title('Log l2 error', fontsize = 12)
plt.savefig('images/l2_error_quad_U.ps')
plt.savefig('images/l2_error_quad_U.png')
plt.show()