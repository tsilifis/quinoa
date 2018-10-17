import numpy as np
import chaos_toolbox as ct
import matplotlib.pyplot as plt 


[x_quad3, w3] = ct.util.QuadratureRule('CC').get_rule(5, 3)
[x_quad4, w4] = ct.util.QuadratureRule('CC').get_rule(5, 4)
[x_quad5, w5] = ct.util.QuadratureRule('CC').get_rule(5, 5)
[x_quad6, w6] = ct.util.QuadratureRule('CC').get_rule(5, 6)

quad_points = [x_quad3] + [x_quad4] + [x_quad5] + [x_quad6]
quad_wghts = [w3] + [w4] + [w5] + [w6]
y_quad3 = np.load('../data_pressure_CC_quadrature_lev3.npy')[:,-1]
y_quad4 = np.load('../data_pressure_CC_quadrature_lev4.npy')[:,-1]
y_quad5 = np.load('../data_pressure_CC_quadrature_lev5.npy')[:,-1]
y_quad6 = np.load('../data_pressure_CC_quadrature_lev6.npy')[:,-1]

y_quads = [y_quad3] + [y_quad4] + [y_quad5] + [y_quad6]

l2_err = np.zeros((4, 8))
for i in range(l2_err.shape[0]):
	for j in range(l2_err.shape[1]):
		Q = 2+j
		[x, w] = quad_points[i], quad_wghts[i]
		y = y_quads[i]
		c = ct.chaos.PolyChaos().comp_coeffs(x, y, w, Q, 'L')
		pol = ct.chaos.PolyBasis(5, Q, 'L')
		P = pol(x_quad6)
		y_pred = np.dot(P, c)
		diff = (y_quad6 - y_pred)**2
		l2_err[i,j] = np.abs(np.sum(diff * w6)) / np.sum(y_quad6**2 * w6)

y_axis = [quad_points[i].shape[0] for i in range(4)]
x_axis = [2,3,4,5,6,7,8,9]

print l2_err.shape

CS = plt.contour(x_axis, y_axis, np.log(l2_err), 12)
plt.clabel(CS, inline = 1, fmt = '%1.1f', fontsize = 10)
plt.xlabel('PC order', fontsize = 12)
plt.ylabel('# abscissae', fontsize = 12)
plt.title('Log l2 error', fontsize = 12)
plt.savefig('images/l2_error_quad_U.ps')
plt.savefig('images/l2_error_quad_U.png')
plt.show()