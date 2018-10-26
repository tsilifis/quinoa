"""
A class for covariance kernel functions.


Author: Panagiotis Tsilifis
Date: 09/10/2017
"""

__all__ = ['Kernel', 'RBF', 'Exponential', 'DifferentialKernel', 'DifferentialKernel_1D']

import numpy as np
from scipy import misc

class Kernel(object):
	"""
	A class representing covariance kernels.
	"""

	_input_dim = None

	_name = None

	@property
	def input_dim(self):
		return self._input_dim

	@input_dim.setter
	def input_dim(self, value):
		assert isinstance(dim , int)
		assert dim > 0
		self._input_dim = value

	def __init__(self, input_dim, name = 'Kernel'):
		"""
		Initializes the object
		"""
		assert isinstance(input_dim, int)
		assert input_dim > 0
		self._input_dim = input_dim
		self._name = name


	def cov(self, X, X2 = None):
		raise NotImplementedError

	def cov_diag(self, X):
		raise NotImplementedError



class RBF(Kernel):
	"""
	A class of Kernel type representing the squared exponential covariance kernel.
	"""

	_var = None

	_lengthscale = None

	_iso = None

	_n_params = None

	@property
	def var(self):
		return self._var

	@var.setter
	def var(self, value):
		assert value > 0.
		self._var = value

	@property
	def lengthscale(self):
		return self._lengthscale

	@lengthscale.setter
	def lengthscale(self, value):
		assert value > 0.
		self._lengthscale = value


	def __init__(self, input_dim, variance = 1., corr_length = 1., name = 'rbf', iso = True):
		"""
		Initializing the object.
		"""
		super(RBF, self).__init__(input_dim, name)
		assert variance > 0
		if isinstance(corr_length, list):
			assert len(corr_length) == input_dim
			self._iso = False
			for i in range(input_dim):
				assert corr_length[i] > 0
			self._lengthscale = corr_length
		else:
			assert corr_length > 0
			self._lengthscale = [corr_length] * input_dim
			self._iso = True
		self._var = variance

		if self._iso:
			self._n_params = 2
		else:
			self._n_params = self._input_dim + 1

	def eval(self, x, y):
		if self._input_dim == 1:
			diff = (x - y) / self._lengthscale[0]
			return self._var * np.exp( - np.square(diff)/ 2.)
		else:
			diff = np.array([(x[i] - y[i]) / self._lengthscale[i] for i in range(self._input_dim)])
			return self._var * np.exp( - np.sum(np.square(diff))/ 2. )

	def cov(self, X, Y = None):
		assert X.shape[1] == self._input_dim
		if Y is None:
			diff = np.vstack([(X[:,i][:,None] - X[:,i][None,:]).reshape(1, X.shape[0], X.shape[0]) / self._lengthscale[i] for i in range(X.shape[1])])
			diff_sq = np.sum(np.square(diff), 0)
			return self._var * np.exp( - diff_sq / 2.)
		else:
			assert Y.shape[1] == self._input_dim
			diff = np.vstack([ (X[:,i][:,None] - Y[:,i][None,:]).reshape(1, X.shape[0], Y.shape[0]) / self._lengthscale[i] for i in range(self._input_dim)])
			diff_sq = np.sum(np.square(diff), 0)
			return self._var * np.exp( - diff_sq / 2.)


	def d_cov_d_var(self, X, Y = None):
		if Y is None:
			return self.cov(X) / self._var
		else:
			return self.cov(X, Y) / self._var

	def d_cov_d_logvar(self, X, Y = None):
		if Y is None:
			return 2 * self.cov(X)
		else:
			return 2 * self.cov(X, Y)


	def d_cov_d_l(self, X, Y = None):
		assert X.shape[1] == self._input_dim
		if Y is None:
			diff = np.vstack([ (X[:,i][:,None] - X[:,i][None,:]).reshape(1, X.shape[0], X.shape[0]) / self._lengthscale[i] for i in range(X.shape[1])])
			diff_sq = np.sum(np.square(diff), 0)
			if self._iso:
				return self._var * np.exp( - diff_sq / 2.) * np.einsum(1 / np.array(self._lengthscale), [0], np.square(diff), [0,1,2])
			else:
				return np.vstack([ ( self._var * np.exp( - diff_sq / 2.) * np.square(diff)[i,:,:] / self._lengthscale[i] ** 3).reshape(1, X.shape[0], X.shape[0]) for i in range(self._input_dim)])
		else:
			diff = np.vstack([ (X[:,i][:,None] - Y[:,i][None,:]).reshape(1, X.shape[0], Y.shape[0]) / self._lengthscale[i] for i in range(X.shape[1])])
			diff_sq = np.sum(np.square(diff), 0)
			if len(self._lengthscale) == 1:
				return self._var * np.exp( - diff_sq / 2.) * np.einsum(1 / np.array(self._lengthscale), [0], np.square(diff), [0,1,2])
			else:
				return np.vstack([ ( self._var * np.exp( - diff_sq / 2.) * np.square(diff)[i,:,:] / self._lengthscale[i] ** 3).reshape(1, X.shape[0], Y.shape[0]) for i in range(self._input_dim)])


	def d_cov_d_X(self, X, Y = None):
		assert X.shape[1] == self._input_dim
		if Y is None:
			der = np.vstack([ (self.cov(X) * (X[:,i][:,None] - X[:,i][None,:])).reshape(1, X.shape[0], X.shape[0]) / self._lengthscale[i] ** 2 for i in range(X.shape[1]) ])
			return - der
		else:
			assert Y.shape[1] == self._input_dim
			der = np.vstack([ (self.cov(X,Y) * (X[:,i][:,None] - Y[:,i][None,:])).reshape(1, X.shape[0], Y.shape[0]) / self._lengthscale[i] ** 2 for i in range(X.shape[1]) ])
			return - der

	def d_cov_d_Y(self, X, Y = None):
		assert X.shape[1] == self._input_dim
		if Y is None:
			return np.vstack([ (self.cov(X) * (X[:,i][:,None] - X[:,i][None,:])).reshape(1, X.shape[0], X.shape[0]) / self._lengthscale[i] ** 2 for i in range(X.shape[1]) ])
		else:
			assert Y.shape[1] == self._input_dim
			return np.vstack([ (self.cov(X,Y) * (X[:,i][:,None] - Y[:,i][None,:])).reshape(1, X.shape[0], Y.shape[0]) / self._lengthscale[i] ** 2 for i in range(X.shape[1]) ])
	

	def d2_cov_d_XY(self, X, Y = None):
		assert X.shape[1] == self._input_dim
		if Y is None:
			return np.vstack([ (self.cov(X) * (1 * (i==j) - (X[:,i][:,None] - X[:,i][None,:]) * (X[:,j][:,None] - X[:,j][None,:]) / self._lengthscale[i] ** 2) / self._lengthscale[j] ** 2).reshape(1, X.shape[0], X.shape[0]) for i in range(X.shape[1]) for j in range(X.shape[1])]).reshape((X.shape[1], X.shape[1], X.shape[0], X.shape[0]))
		else:
			assert Y.shape[1] == self._input_dim
			return np.vstack([ (self.cov(X, Y) * (1 * (i==j) - (X[:,i][:,None] - Y[:,i][None,:]) * (X[:,j][:,None] - Y[:,j][None,:]) / self._lengthscale[i] ** 2) / self._lengthscale[j] ** 2).reshape(1, X.shape[0], Y.shape[0]) for i in range(X.shape[1]) for j in range(X.shape[1])]).reshape((X.shape[1], X.shape[1], X.shape[0], Y.shape[0]))

	def d2_cov_d_XX(self, X, Y = None):
		assert X.shape[1] == self._input_dim
		if Y is None:
			return - np.vstack([ (self.cov(X) * (1 * (i==j) - (X[:,i][:,None] - X[:,i][None,:]) * (X[:,j][:,None] - X[:,j][None,:]) / self._lengthscale[i] ** 2) / self._lengthscale[j] ** 2).reshape(1, X.shape[0], X.shape[0]) for i in range(X.shape[1]) for j in range(X.shape[1])]).reshape((X.shape[1], X.shape[1], X.shape[0], X.shape[0]))
		else:
			assert Y.shape[1] == self._input_dim
			return - np.vstack([ (self.cov(X, Y) * (1 * (i==j) - (X[:,i][:,None] - Y[:,i][None,:]) * (X[:,j][:,None] - Y[:,j][None,:]) / self._lengthscale[i] ** 2) / self._lengthscale[j] ** 2).reshape(1, X.shape[0], Y.shape[0]) for i in range(X.shape[1]) for j in range(X.shape[1])]).reshape((X.shape[1], X.shape[1], X.shape[0], Y.shape[0]))

	def d2_cov_d_YY(self, X, Y = None):
		return - self.d2_cov_d_XX(X, Y)

	def d3_cov_d_XXY(self, X, Y = None):
		assert X.shape[1] == self._input_dim
		if Y is None:
			return - np.vstack([ (self.cov(X) * ((i==l) * (X[:,j][:,None] - X[:,j][None,:]) / (self._lengthscale[j]*self._lengthscale[l])**2 + (j==l) * (X[:,i][:,None] - X[:,i][None,:]) / (self._lengthscale[i]*self._lengthscale[l])**2 + (i==j) * (X[:,l][:,None] - X[:,l][None,:]) / (self._lengthscale[l]*self._lengthscale[i])**2 ) - (X[:,i][:,None] - X[:,i][None,:]) * (X[:,j][:,None] - X[:,j][None,:]) * (X[:,l][:,None] - X[:,l][None,:]) / (self._lengthscale[i]*self._lengthscale[j]*self._lengthscale[l])**2 ).reshape(1, X.shape[0], X.shape[0]) for i in range(X.shape[1]) for j in range(X.shape[1]) for l in range(X.shape[1]) ]).reshape((X.shape[1], X.shape[1], X.shape[1], X.shape[0], X.shape[0]))
		else:
			assert Y.shape[1] == self._input_dim
			return - np.vstack([ (self.cov(X, Y) * ((i==l) * (X[:,j][:,None] - Y[:,j][None,:]) / (self._lengthscale[j]*self._lengthscale[l])**2 + (j==l) * (X[:,i][:,None] - Y[:,i][None,:]) / (self._lengthscale[i]*self._lengthscale[l])**2 + (i==j) * (X[:,l][:,None] - Y[:,l][None,:]) / (self._lengthscale[l]*self._lengthscale[i])**2 ) - (X[:,i][:,None] - Y[:,i][None,:]) * (X[:,j][:,None] - Y[:,j][None,:]) * (X[:,l][:,None] - Y[:,l][None,:]) / (self._lengthscale[i]*self._lengthscale[j]*self._lengthscale[l])**2 ).reshape(1, X.shape[0], Y.shape[0]) for i in range(X.shape[1]) for j in range(X.shape[1]) for l in range(X.shape[1]) ]).reshape((X.shape[1], X.shape[1], Y.shape[1], X.shape[0], Y.shape[0]))

	def d3_cov_d_YYX(self, X, Y = None):
		return self.d3_cov_d_XXY(X, Y)

	def d4_cov_d_XXYY(self, X, Y = None):
		assert X.shape[1] == self._input_dim
		if Y is None:
			return np.vstack([ (self.cov(X) * ( (i==l)*(j==k)/(self._lengthscale[i]*self._lengthscale[k])**2 + (l==j)*(k==i) / (self._lengthscale[i]*self._lengthscale[j])**2 + (i==j)*(k==l) / (self._lengthscale[i]*self._lengthscale[l])**2 - (j==i) * (X[:,l][:,None] - X[:,l][None,:])*(X[:,k][:,None] - X[:,k][None,:]) / (self._lengthscale[i]*self._lengthscale[l]*self._lengthscale[k])**2 - (l==j) * (X[:,i][:,None] - X[:,i][None,:]) * (X[:,k][:,None] - X[:,k][None,:]) / (self._lengthscale[l]*self._lengthscale[i]*self._lengthscale[k])**2  - (l==i)*(X[:,j][:,None] - X[:,j][None,:])*(X[:,k][:,None] - X[:,k][None,:]) / (self._lengthscale[i]*self._lengthscale[j]*self._lengthscale[k])**2 - (l==k)* (X[:,i][:,None] - X[:,i][None,:]) * (X[:,j][:,None] - X[:,j][None,:]) / (self._lengthscale[l]*self._lengthscale[k]*self._lengthscale[i])**2  - (k==j)*(X[:,i][:,None] - X[:,i][None,:])*(X[:,l][:,None] - X[:,l][None,:]) / (self._lengthscale[i]*self._lengthscale[j]*self._lengthscale[l])**2  - (i==k)*(X[:,l][:,None] - X[:,l][None,:]) * (X[:,j][:,None] - X[:,j][None,:]) / (self._lengthscale[i]*self._lengthscale[l]*self._lengthscale[j])**2  + (X[:,i][:,None] - X[:,i][None,:])*(X[:,j][:,None] - X[:,j][None,:])*(X[:,l][:,None]-X[:,l][None,:] )*(X[:,k][:,None] - X[:,k][None,:]) / (self._lengthscale[i]*self._lengthscale[j]*self._lengthscale[k]*self._lengthscale[l])**2 )  ).reshape(1, X.shape[0], X.shape[0]) for i in range(X.shape[1]) for j in range(X.shape[1]) for l in range(X.shape[1]) for k in range(X.shape[1]) ]).reshape((X.shape[1], X.shape[1], X.shape[1], X.shape[1], X.shape[0], X.shape[0]))
		else:
			assert Y.shape[1] == self._input_dim
			return np.vstack([ (self.cov(X, Y) * ( (i==l)*(j==k)/(self._lengthscale[i]*self._lengthscale[k])**2 + (l==j)*(k==i) / (self._lengthscale[i]*self._lengthscale[j])**2 + (i==j)*(k==l) / (self._lengthscale[i]*self._lengthscale[l])**2 - (j==i) * (X[:,l][:,None] - Y[:,l][None,:])*(X[:,k][:,None] - Y[:,k][None,:]) / (self._lengthscale[i]*self._lengthscale[l]*self._lengthscale[k])**2 - (l==j) * (X[:,i][:,None] - Y[:,i][None,:]) * (X[:,k][:,None] - Y[:,k][None,:]) / (self._lengthscale[l]*self._lengthscale[i]*self._lengthscale[k])**2  - (l==i)*(X[:,j][:,None] - Y[:,j][None,:])*(X[:,k][:,None] - Y[:,k][None,:]) / (self._lengthscale[i]*self._lengthscale[j]*self._lengthscale[k])**2 - (l==k)* (X[:,i][:,None] - Y[:,i][None,:]) * (X[:,j][:,None] - Y[:,j][None,:]) / (self._lengthscale[l]*self._lengthscale[k]*self._lengthscale[i])**2  - (k==j)*(X[:,i][:,None] - Y[:,i][None,:])*(X[:,l][:,None] - Y[:,l][None,:]) / (self._lengthscale[i]*self._lengthscale[j]*self._lengthscale[l])**2  - (i==k)*(X[:,l][:,None] - Y[:,l][None,:]) * (X[:,j][:,None] - Y[:,j][None,:]) / (self._lengthscale[i]*self._lengthscale[l]*self._lengthscale[j])**2  + (X[:,i][:,None] - Y[:,i][None,:])*(X[:,j][:,None] - Y[:,j][None,:])*(X[:,l][:,None] - Y[:,l][None,:] )*(X[:,k][:,None] - Y[:,k][None,:]) / (self._lengthscale[i]*self._lengthscale[j]*self._lengthscale[k]*self._lengthscale[l])**2 )  ).reshape(1, X.shape[0], Y.shape[0]) for i in range(X.shape[1]) for j in range(X.shape[1]) for l in range(X.shape[1]) for k in range(X.shape[1]) ]).reshape((X.shape[1], X.shape[1], X.shape[1], X.shape[1], X.shape[0], Y.shape[0]))



class Exponential(Kernel):
	"""
	A class of Kernel type representing the squared exponential covariance kernel.
	"""

	_var = None

	_lengthscale = None

	_iso = None

	_gamma = None

	_n_params = None

	@property
	def var(self):
		return self._var

	@var.setter
	def var(self, value):
		assert value > 0.
		self._var = value

	@property
	def lengthscale(self):
		return self._lengthscale

	@lengthscale.setter
	def lengthscale(self, value):
		assert value > 0.
		self._lengthscale = value

	@property
	def gamma(self):
		return self._gamma

	@gamma.setter
	def gamma(self, value):
		assert value > 0
		assert value <= 2.
		self._gamma = value


	def __init__(self, input_dim, variance = 1., corr_length = 1., name = 'exponential', iso = True, gamma = 1):
		"""
		Initializing the object.
		"""
		super(Exponential, self).__init__(input_dim, name)
		assert variance > 0
		if isinstance(corr_length, list):
			assert len(corr_length) == input_dim
			for i in range(input_dim):
				assert corr_length[i] > 0
			self._lengthscale = corr_length
			self._iso = False
		else:
			assert corr_length > 0
			self._lengthscale = [corr_length] * input_dim
			self._iso = True
		self._var = variance
		self._gamma = gamma

		#if self._iso:
		self._n_params = 2
		#else:
		#	self._n_params = self._input_dim + 1


	def cov(self, X, Y = None):
		assert X.shape[1] == self._input_dim
		if Y is None:
			diff = np.vstack([(X[:,i][:,None] - X[:,i][None,:]).reshape(1, X.shape[0], X.shape[0]) / self._lengthscale[i] for i in range(X.shape[1])])
			diff_sq = np.sum(np.square(diff), 0)
			return self._var * np.exp(- ( np.sqrt(diff_sq) ) ** self._gamma)
		else:
			diff = np.vstack([(X[:,i][:,None] - Y[:,i][None,:]).reshape(1, X.shape[0], Y.shape[0]) / self._lengthscale[i] for i in range(X.shape[1])])
			diff_sq = np.sum(np.square(diff), 0)
			return self._var * np.exp(- ( np.sqrt(diff_sq) ) ** self._gamma)

	def d_cov_d_var(self, X, Y = None):
		if Y is None:
			return self.cov(X) / self._var
		else:
			return self.cov(X, Y) / self._var

	def d_cov_d_logvar(self, X, Y = None):
		if Y is None:
			return 2 * self.cov(X)
		else:
			return 2 * self.cov(X, Y)

	def d_cov_d_l(self, X, Y = None):
		assert X.shape[1] == self._input_dim
		if Y is None:
			diff = np.vstack([(X[:,i][:,None] - X[:,i][None,:]).reshape(1, X.shape[0], X.shape[0]) / self._lengthscale[i] for i in range(X.shape[1])])
			#diff = np.vstack([(X[:,i][:,None] - X[:,i][None,:]).reshape(1, X.shape[0], X.shape[0]) / self._lengthscale[i] for i in range(X.shape[1])])
			diff_sq = np.sum(np.square(diff), 0)
			if self._iso:
				return self._var * np.exp( - (np.sqrt(diff_sq) ) ** self._gamma ) * np.sqrt(diff_sq) ** self._gamma / self._lengthscale[0] ** (self._gamma - 1.)
		else:
			diff = np.vstack([(X[:,i][:,None] - Y[:,i][None,:]).reshape(1, X.shape[0], Y.shape[0]) / self._lengthscale[i] for i in range(X.shape[1])])
			diff_sq = np.sum(np.square(diff), 0)
			if self._iso:
				return self._var * np.exp( - (np.sqrt(diff_sq) ) ** self._gamma ) * np.sqrt(diff_sq) ** self._gamma / self._lengthscale[0] ** (self._gamma - 1.)




class DifferentialKernel(Kernel):
	"""
	A class of covariance kernels defined by differential operators.
	"""

	_order = None

	_diff_factors = None

	_base_kernel = None

	_active_inp = None

	def __init__(self, input_dim, order, diff_factors = None, base_kernel = None, active_inp_1 = True, name = 'differential kernel'):
		"""
		Initializes the object
		"""
		super(DifferentialKernel, self).__init__(input_dim, name)
		assert isinstance(order, int)
		assert order > -1 and order < 3, 'Only up to 2nd order differential operators are currently supported.'
		self._order = order
		if base_kernel is not None:
			assert isinstance(base_kernel, Kernel), 'Covariance kernel must be a Kernel object.'
			self._base_kernel = base_kernel
		if diff_factors is None:
			self._diff_factors = [self.const_factor] * self.mi_terms(order, input_dim).shape[0]
		else:
			assert isinstance(diff_factors, list)
			assert len(diff_factors) == self.mi_terms(order, input_dim).shape[0]
			self._diff_factors = diff_factors
		self._active_inp = active_inp_1


	def mi_terms(self, order = None, dim = None):
		""" matrix of basis terms
		Input
		:order: PCE order
		:dim: PCE dimension
		"""
		if order is None:
			order = self._order
		if dim is None:
			dim = self._input_dim

		if order == 0:
			return np.array(np.zeros(dim, dtype = int), dtype = int)
		else:
			q_num = [int(misc.comb(dim+i-1, i)) for i in range(order+1)]
			mul_ind = np.array(np.zeros(dim, dtype = int), dtype = int)
			mul_ind = np.vstack([mul_ind, np.eye(dim, dtype = int)])
			I = np.eye(dim, dtype = int)
			ind = [1] * dim
			for j in range(1,order):
				ind_new = []
				for i in range(dim):
					a0 = np.copy(I[int(np.sum(ind[:i])):,:])
					a0[:,i] += 1
					mul_ind = np.vstack([mul_ind, a0])
					ind_new += [a0.shape[0]]
				ind = ind_new
				I = np.copy(mul_ind[np.sum(q_num[:j+1]):,:])
			return mul_ind

	def const_factor(self, X):
		return np.ones(X.shape[0])

	def set_diff_factor(self, factor, index):
		assert isinstance(index, int)
		assert index > -1 and index < len(self._diff_factors)
		self._diff_factors[index] = factor


	def cov(self, X, Y = None):
		assert X.shape[1] == self._input_dim
		if Y is not None:
			assert Y.shape[1] == self._input_dim
		#cov = np.zeros((X.shape[0],X.shape[0]))
		if self._active_inp:
			if self._order == 0:
				C = self._base_kernel.cov(X, Y)
				return (self._diff_factors[0](X) * C.T).T
			elif self._order == 1:
				C = self._base_kernel.cov(X, Y)
				dC = self._base_kernel.d_cov_d_X(X, Y)
				return (self._diff_factors[0](X) * C.T).T + np.array([(self._diff_factors[i+1](X) * dC[i,:,:].T).T for i in range(self._input_dim)]).sum(axis = 0)
			else:
				C = self._base_kernel.cov(X, Y)
				dC = self._base_kernel.d_cov_d_X(X, Y)
				d2C = self._base_kernel.d2_cov_d_XX(X, Y)
			#K = [C, dC, d2C] # Shapes are (n x n), (d x n x n), (d x d x n x n)
		
				return (self._diff_factors[0](X) * C.T).T + np.array([(self._diff_factors[i+1](X) * dC[i,:,:].T).T for i in range(self._input_dim)]).sum(axis = 0) + np.array([(self._diff_factors[i*self._input_dim + j - np.sum(range(i+1))](X) * d2C[i,j,:,:]).T for i in range(self._input_dim) for j in range(i, self._input_dim)]).sum(axis = 0)
		else:
			if self._order == 0:
				C = self._base_kernel.cov(X, Y)
				return (self._diff_factors[0](X) * C.T).T
			elif self._order == 1:
				C = self._base_kernel.cov(X, Y)
				dC = self._base_kernel.d_cov_d_Y(X, Y)
				return (self._diff_factors[0](X) * C.T).T + np.array([(self._diff_factors[i+1](X) * dC[i,:,:].T).T for i in range(self._input_dim)]).sum(axis = 0)
			else:
				C = self._base_kernel.cov(X, Y)
				dC = self._base_kernel.d_cov_d_Y(X, Y)
				d2C = self._base_kernel.d2_cov_d_YY(X, Y)
			#K = [C, dC, d2C] # Shapes are (n x n), (d x n x n), (d x d x n x n)
		
				return (self._diff_factors[0](X) * C.T).T + np.array([(self._diff_factors[i+1](X) * dC[i,:,:].T).T for i in range(self._input_dim)]).sum(axis = 0) + np.array([(self._diff_factors[i*self._input_dim + j - np.sum(range(i+1))](X) * d2C[i,j,:,:]).T for i in range(self._input_dim) for j in range(i, self._input_dim)]).sum(axis = 0)


		
class DifferentialKernel_1D(Kernel):
	"""
	A class of covariance kernels defined by differential operators.
	"""

	_order = None

	_diff_factors = None

	_base_kernel = None

	_active_inp = None

	def __init__(self, input_dim, order, diff_factors = None, base_kernel = None, active_inp_1 = True, name = 'differential kernel'):
		"""
		Initializes the object
		"""
		super(DifferentialKernel_1D, self).__init__(input_dim, name)
		assert isinstance(order, int)
		#assert order > -1 and order < 3, 'Only up to 2nd order differential operators are currently supported.'
		self._order = order
		if base_kernel is not None:
			assert isinstance(base_kernel, Kernel), 'Covariance kernel must be a Kernel object.'
			self._base_kernel = base_kernel
		if diff_factors is None:
			self._diff_factors = [self.const_factor] * 9
		else:
			assert isinstance(diff_factors, list)
			assert len(diff_factors) == self.mi_terms(order, input_dim).shape[0]
			self._diff_factors = diff_factors
		self._active_inp = active_inp_1


	def mi_terms(self, order = None, dim = None):
		""" matrix of basis terms
		Input
		:order: PCE order
		:dim: PCE dimension
		"""
		if order is None:
			order = self._order
		if dim is None:
			dim = self._input_dim

		if order == 0:
			return np.array(np.zeros(dim, dtype = int), dtype = int)
		else:
			q_num = [int(misc.comb(dim+i-1, i)) for i in range(order+1)]
			mul_ind = np.array(np.zeros(dim, dtype = int), dtype = int)
			mul_ind = np.vstack([mul_ind, np.eye(dim, dtype = int)])
			I = np.eye(dim, dtype = int)
			ind = [1] * dim
			for j in range(1,order):
				ind_new = []
				for i in range(dim):
					a0 = np.copy(I[int(np.sum(ind[:i])):,:])
					a0[:,i] += 1
					mul_ind = np.vstack([mul_ind, a0])
					ind_new += [a0.shape[0]]
				ind = ind_new
				I = np.copy(mul_ind[np.sum(q_num[:j+1]):,:])
			return mul_ind

	def const_factor(self, X):
		return np.ones(X.shape[0])

	def set_diff_factor(self, factor, index):
		assert isinstance(index, int)
		assert index > -1 and index < len(self._diff_factors)
		self._diff_factors[index] = factor


	def cov(self, X, Y = None):
		assert X.shape[1] == self._input_dim
		if Y is not None:
			assert Y.shape[1] == self._input_dim
		#cov = np.zeros((X.shape[0],X.shape[0]))
		C = self._base_kernel.cov(X,Y)
		dC_1 = self._base_kernel.d_cov_d_X(X, Y)
		dC_2 = self._base_kernel.d_cov_d_Y(X, Y)
		d2C_1 = self._base_kernel.d2_cov_d_XX(X, Y)
		d2C_2 = self._base_kernel.d2_cov_d_YY(X, Y)
		d2C_3 = self._base_kernel.d2_cov_d_XY(X, Y)
		d3C_1 = self._base_kernel.d3_cov_d_XXY(X, Y)
		d3C_2 = self._base_kernel.d3_cov_d_YYX(X, Y)
		d4C = self._base_kernel.d4_cov_d_XXYY(X, Y)

		return self._diff_factors[0](X,Y) * C + self._diff_factors[1](X,Y) * dC_1[0,:,:] + self._diff_factors[2](X,Y) * dC_2[0,:,:] + self._diff_factors[3](X,Y) * d2C_1[0,0,:,:] + self._diff_factors[4](X,Y) * d2C_2[0,0,:,:] + self._diff_factors[5](X,Y) * d2C_3[0,0,:,:] + self._diff_factors[6](X,Y) * d3C_1[0,0,0,:,:] + self._diff_factors[7](X,Y) * d3C_2[0,0,0,:,:] + self._diff_factors[8](X,Y) * d4C[0,0,0,0,:,:]

		





