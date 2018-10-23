"""
Class for Gaussian Process regression

Author : Panos Tsilifis
Date : 09/12/2017

"""

__all__ = ['GP']

from _kern import *
import numpy as np
from scipy.optimize import minimize, basinhopping
from scipy.optimize import fmin_l_bfgs_b


class GP(object):
	"""
	A class for Gaussian Process regression

	"""

	_X = None

	_Y = None

	_kern = None

	_log_marginal_likelihood = None

	_noise_var = None

	_scaled_data = None

	_chol = None

	def __init__(self, X, Y, kernel, noise_var = 1.):
		"""
		Initializes the object

		"""

		assert X.ndim == 2
		self._X = X.copy()

		assert Y.ndim == 2
		self._Y = Y.copy()

		assert isinstance(kernel, Kernel)
		self._kern = kernel

		assert noise_var > 0.
		self._noise_var = noise_var

		cov = self._kern.cov(X)
		n = X.shape[0]
		cov += np.diag((self._noise_var + 1e-8) * np.ones(cov.shape[0]))
		L = np.linalg.cholesky(cov)
		self._chol = L
		a = np.linalg.solve(L.T, np.linalg.solve(L, Y[:,0]))

		self._scaled_data = a
		self._log_marginal_likelihood = - n * np.log(2*np.pi) / 2. - np.log(np.diag(L)).sum() - (Y[:,0] * a).sum() / 2.

	def predict(self, X_test):
		assert X_test.shape[1] == self._X.shape[1]

		K_test = self._kern.cov(self._X, X_test)
		f_mean = np.dot(K_test.T, self._scaled_data.flatten())
		v = np.linalg.solve(self._chol, K_test)
		f_var = self._kern.cov(X_test) + np.diag( (self._noise_var + 1e-8) * np.ones(X_test.shape[0])) - np.dot(v.T, v)
		#f_var = self._kern.cov(X_test) - np.dot(v.T, v)
		return f_mean, f_var

	def log_marginal_likelihood(self, params):
		assert len(params) == self._kern._n_params + 1
		n = self._kern._n_params
		if self._kern._iso:
			self._kern._lengthscale = [params[0]] * self._kern._input_dim
		else:
			self._kern._lengthscale = list(params[:(n-1)])

		self._kern._var = params[n-1]
		self._noise_var = params[n]

		cov = self._kern.cov(self._X)
		cov_noise = cov.copy() + np.diag( (self._noise_var + 1e-8) * np.ones(cov.shape[0]) )
		#print self._kern._var
		#print self._noise_var
		#print self._kern._lengthscale
		#print cov_noise
		L = np.linalg.cholesky(cov_noise)
		self._chol = L
		a = np.linalg.solve(L.T, np.linalg.solve(L, self._Y[:,0])).reshape(L.shape[1], 1)
		self._scaled_data = a
		self._log_marginal_likelihood = - 0.5 * self._X.shape[0] * np.log(2*np.pi) - np.log(np.diag(L)).sum() - 0.5 * (self._Y[:,0] * a.flatten()).sum()
		
		likelihood_grad = np.zeros(n + 1)
		
		if self._kern._iso:
			likelihood_grad[0] = 0.5 * np.diag( np.dot( np.dot(a, a.T), self._kern.d_cov_d_l(self._X) ) - np.linalg.solve(L.T, np.linalg.solve(L, self._kern.d_cov_d_l(self._X))) ).sum()
		else:
			likelihood_grad[:(n-1)] = [ 0.5 * np.diag( np.dot( np.dot(a, a.T), self._kern.d_cov_d_l(self._X)[i,:,:] ) - np.linalg.solve(L.T, np.linalg.solve(L, self._kern.d_cov_d_l(self._X)[i,:,:]) ) ).sum() for i in range(n-1)]

		likelihood_grad[n-1] = 0.5 * np.diag( np.dot(np.dot(a, a.T), self._kern.d_cov_d_var(self._X) ) - np.linalg.solve(L.T, np.linalg.solve(L, self._kern.d_cov_d_var(self._X)) ) ).sum()
		likelihood_grad[n]   = 0.5 * np.diag( np.dot(a, a.T) - np.linalg.solve(L.T, np.linalg.solve(L, np.eye(self._Y.shape[0])))  ).sum()

		return self._log_marginal_likelihood, likelihood_grad

	def optimize(self):

		def neg_log_like(params):
			f, df = self.log_marginal_likelihood(params)
			return -f, -df
		
		if self._kern._iso:
			params = np.hstack([self._kern._lengthscale[0], self._kern._var, self._noise_var])
			bnds = ((1e-16, None),) * 3 #+ ((1e-16, None), (1e-16, None),)
		else:
			params = np.hstack([np.array(self._kern._lengthscale), self._kern._var, self._noise_var])
			bnds = ((1e-16, None),) * self._kern._input_dim + ((1e-16, None), (1e-16, None),)

		#res = minimize(neg_log_like, params, method = 'L-BFGS-B', jac = True, bounds = bnds, options = {'disp': False, 'maxcor' : 50, 'ftol': 1e-32, 'gtol': 1e-16})
		#res = fmin_l_bfgs_b(neg_log_like, params, fprime = None, bounds = bnds, factr = 1e2, pgtol = 1e-16)
		
		# --- Using optimize.minimize
		#res = minimize(neg_log_like, params, method = 'TNC', jac = True, bounds = bnds, options = {'ftol': 1e-16, 'gtol': 1e-16, 'maxiter' : 1000})
		
		# --- Using optimize.basinhopping
		minimizer_kwargs = {'method':'TNC', 'jac':True, 'bounds': bnds, 'options': {'ftol': 1e-16, 'gtol': 1e-16, 'maxiter' : 1000}}
		res = basinhopping(neg_log_like, params, minimizer_kwargs = minimizer_kwargs)
		print res

	def argmaxvar(self, bounds = (-4., 4.)):
		assert isinstance(bounds, tuple)

		def neg_var_x(x):
			assert x.shape[0] == self._kern._input_dim
			X_test = x.reshape(1, x.shape[0])
			K_test = self._kern.cov(self._X, X_test)
			K_test_grad = self._kern.d_cov_d_X(X_test, self._X)[:,0,:].T
			v = np.linalg.solve(self._chol, K_test)
			v_grad = np.linalg.solve(self._chol, K_test_grad)

			return - self.predict(X_test)[1][0,0], 2. * np.dot(v_grad.T, v)

		bnds = (bounds,) * self._kern._input_dim
		#res = minimize(neg_var_x, np.random.normal(size = (self._kern._input_dim,)), method = 'L-BFGS-B', jac = True, bounds = bnds, options = {'ftol': 1e-16, 'gtol': 1e-16, 'maxiter' : 1000})
		#res = minimize(neg_var_x, x0, method = 'TNC', jac = True)
		
		# --- Using optimize.basinhopping
		minimizer_kwargs = {'method': 'L-BFGS-B', 'jac': True, 'bounds': bnds, 'options' : {'ftol': 1e-16, 'gtol': 1e-16, 'maxiter' : 1000}}
		res = basinhopping(neg_var_x, np.random.uniform(size = (self._kern._input_dim,))- 0.5, minimizer_kwargs = minimizer_kwargs) 
		return res.x
		
		#f_mean = np.dot(K_test.T, self._scaled_data.flatten())
		


