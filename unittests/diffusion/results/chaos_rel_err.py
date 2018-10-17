import numpy as np 
import chaos_toolbox as ct
import matplotlib.pyplot as plt 


[x_quad, w] = ct.util.QuadratureRule().get_rule(2, 6)
y_quad = np.load('../data_concentrations_GH_quadrature_lev6.npy')[:,-1]

coeffs4 = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, 4)
coeffs5 = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, 5)
coeffs6 = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, 6)
coeffs7 = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, 7)
coeffs8 = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, 8)
coeffs9 = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, 9)
coeffs10 = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, 10)
coeffs11 = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, 11)
coeffs12 = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, 12)
coeffs13 = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, 13)

rel_err = np.zeros(9)
c1 = coeffs5[:coeffs4.shape[0]] - coeffs4
c2 = coeffs5[coeffs4.shape[0]:]
rel_err[0] = np.linalg.norm(c1) + np.linalg.norm(c2)
c1 = coeffs6[:coeffs5.shape[0]] - coeffs5
c2 = coeffs6[coeffs5.shape[0]:]
rel_err[1] = np.linalg.norm(c1) + np.linalg.norm(c2)
c1 = coeffs7[:coeffs6.shape[0]] - coeffs6
c2 = coeffs7[coeffs6.shape[0]:]
rel_err[2] = np.linalg.norm(c1) + np.linalg.norm(c2)
c1 = coeffs8[:coeffs7.shape[0]] - coeffs7
c2 = coeffs8[coeffs7.shape[0]:]
rel_err[3] = np.linalg.norm(c1) + np.linalg.norm(c2)
c1 = coeffs9[:coeffs8.shape[0]] - coeffs8
c2 = coeffs9[coeffs8.shape[0]:]
rel_err[4] = np.linalg.norm(c1) + np.linalg.norm(c2)
c1 = coeffs10[:coeffs9.shape[0]] - coeffs9
c2 = coeffs10[coeffs9.shape[0]:]
rel_err[5] = np.linalg.norm(c1) + np.linalg.norm(c2)
c1 = coeffs11[:coeffs10.shape[0]] - coeffs10
c2 = coeffs11[coeffs10.shape[0]:]
rel_err[6] = np.linalg.norm(c1) + np.linalg.norm(c2)
c1 = coeffs12[:coeffs11.shape[0]] - coeffs11
c2 = coeffs12[coeffs11.shape[0]:]
rel_err[7] = np.linalg.norm(c1) + np.linalg.norm(c2)
c1 = coeffs13[:coeffs12.shape[0]] - coeffs12
c2 = coeffs13[coeffs12.shape[0]:]
rel_err[8] = np.linalg.norm(c1) + np.linalg.norm(c2)

plt.plot(rel_err, '-x')
plt.show()

