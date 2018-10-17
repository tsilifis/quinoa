import numpy as np 
import chaos_toolbox as ct
import matplotlib.pyplot as plt 

[x, w] = ct.util.QuadratureRule().get_rule(2, 6)
print x.shape

y = np.load('data_concentrations_GH_quadrature.npy')[:,-1]
print y.shape

coeffs = ct.chaos.PolyChaos().comp_coeffs(x, y, w, 9)

plt.plot(x[:,0], x[:,1], '*')
plt.show()

plt.plot(coeffs, '-x')
plt.show()