import numpy as np
import kernel_py as kp

rbf = kp.RBF(1,1,1)

L = kp.DifferentialKernel(1, 2, base_kernel = rbf, active_inp_1 = False)

L_1d = kp.DifferentialKernel_1D(1, 4, base_kernel = rbf, active_inp_1 = False)

print L.mi_terms()

def const(X, Y = None):
	if Y is None:
		return 0.25 * np.ones(X.shape)
	else:
		return 0.25 * np.ones((X.shape[0], Y.shape[0]))

def a1(X = None, Y = None):
	assert X is not None or Y is not None, 'at least one input must be not null'
	if Y is None:
		return 0.5 * ( (0.5 * np.arctan(20.*(X-1.)) + 1.).flatten() * np.ones((X.shape[0], X.shape[0])).T).T
	elif X is None:
		return 0.5 * ( (0.5 * np.arctan(20.*(Y-1.)) + 1.).flatten() * np.ones((Y.shape[0], Y.shape[0])).T)
	else:
		return 0.5 * ( (0.5 * np.arctan(20.*(X-1.)) + 1.).flatten() * np.ones((X.shape[0], Y.shape[0])).T).T

def a2(X = None, Y = None):
	assert X is not None or Y is not None, 'at least one input must be not null'
	if Y is None:
		return 0.5 * ( (0.5 * np.arctan(20.*(X-1.)) + 1.).flatten() * np.ones((X.shape[0], X.shape[0])).T)
	else:
		return 0.5 * ( (0.5 * np.arctan(20.*(Y-1.)) + 1.).flatten() * np.ones((Y.shape[0], X.shape[0])).T)

def a_prime(X = None, Y = None):
	assert X is not None or Y is not None, 'at least one input must be not null'
	if Y is None:
		return 0.5 * ( (10. / (1. + 400. * (X - 1.) ** 2)).flatten() * np.ones((X.shape[0], X.shape[0])).T ).T
	else:
		return 0.5 * ( (10. / (1. + 400. * (X - 1.) ** 2)).flatten() * np.ones((X.shape[0], Y.shape[0])).T ).T

def a_prime2(X, Y = None):
	if Y is None:
		return 0.5 * ( (10. / (1. + 400. * (X - 1.) ** 2)).flatten() * np.ones((X.shape[0], X.shape[0])).T )
	else:
		return 0.5 * ( (10. / (1. + 400. * (X - 1.) ** 2)).flatten() * np.ones((Y.shape[0], X.shape[0])).T )

def a_sq(X, Y = None):
	if Y is None:
		return 4. * a1(X) * a2(X)
	else:
		return 4. * a1(X, Y) * a2(X, Y)

def a_ap(X, Y = None):
	if Y is None:
		return 4. * a1(X) * a_prime(X).T
	else:
		return 4. * a1(X,Y) * a_prime(Y,X).T

def ap_a(X, Y = None):
	if Y is None:
		return 4. * a_prime(X) * a2(X)
	else:
		return 4. * a_prime(X, Y) * a2(X, Y)

def ap_sq(X, Y = None):
	if Y is None:
		return 4. * a_prime(X) * a_prime(X).T
	else:
		return 4. * a_prime(X, Y) * a_prime(Y, X).T

def zero(X, Y = None):
	if Y is None:
		return np.zeros(X.shape)
	else:
		return np.zeros((X.shape[0], Y.shape[1]))


L_1d._diff_factors[0] = const
L_1d._diff_factors[1] = a_prime
L_1d._diff_factors[2] = a_prime2
L_1d._diff_factors[3] = a1
L_1d._diff_factors[4] = a2
L_1d._diff_factors[5] = ap_sq
L_1d._diff_factors[6] = a_ap
L_1d._diff_factors[7] = ap_a
L_1d._diff_factors[8] = a_sq


X = np.linspace(0., 3, 100).reshape(100,1)


print a1(X).shape
C = L_1d.cov(X)
print C.shape
import matplotlib.pyplot as plt 

plt.contourf(C, 40)
plt.colorbar()
plt.show()