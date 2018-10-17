import numpy as np
import scipy.stats as st
import math
import kernel_py as kp
from chaos_toolbox import chaos, util


class IntegralPosterior(object):

	_input_dim = None

	_X = None

	_Y = None

	_pol_order = None

	_kern = None

	def __init__(self, X, Y, pol_order, kern):
		"""
		Initializes the object
		"""

		assert X.shape[0] == Y.shape[0]
		self._input_dim = X.shape[1]
		assert Y.shape[1] == 1
		assert isinstance(pol_order, int)
		assert isinstance(kern, kp.Kernel)
		self._X = X
		self._Y = Y
		self._pol_order = pol_order
		self._kern = kern


	def pi_H(self, n, xi, l, var = False):
		
		C = np.exp(- xi**2 / (2*(l**2+1)))
		mu = xi / (l**2 + 1)
		if not var:
			sig2 = l**2 / (l**2 + 1)
		else:
			sig2 = (l**2 + 1.) / (l**2 + 2.)
		if n == 0:
			return C * np.sqrt(2*np.pi*sig2)
		elif n == 1:
			return C * np.sqrt(2*np.pi*sig2) * mu
		else:
			pi_new = 0.
			pi_old1 = C * np.sqrt(2*np.pi*sig2)
			pi_old2 = C * np.sqrt(2*np.pi*sig2) * mu
			for i in range(2, n+1):
				pi_new = mu * pi_old2 / np.sqrt(float(i)) + (sig2 - 1.) * pi_old1 * np.sqrt(float(i-1) / float(i))
				pi_old1 = pi_old2.copy()
				pi_old2 = pi_new.copy()
			return pi_new

	def pi_L(self, n, xi, l):

		if n == 0:
			return np.sqrt(2*np.pi*l**2) * (st.norm(loc = xi, scale = np.sqrt(l)).cdf(1) - st.norm(loc = xi, scale = np.sqrt(l)).cdf(-1))
		elif n == 1:
			return np.sqrt(6*np.pi*l**2) * xi * (st.norm(loc = xi, scale = np.sqrt(l)).cdf(1) - st.norm(loc = xi, scale = np.sqrt(l)).cdf(-1)) - np.sqrt(3.) * l**2 * (np.exp(- (1-xi)**2 / (2*l**2)) - np.exp(- (1+xi)**2 / (2*l**2)))
		else:
			C = -l**2 * np.sqrt(2*n+1) * (float(2*n-1)/float(n)) * (np.exp(- (1-xi)**2/(2*l**2)) - (-1.)**(n-1.) *np.exp( - (1.+xi)**2 / (2*l**2)))
			all_pis = []
			#pi_new = 0.
			pi_old1 = np.sqrt(2*np.pi*l**2) * (st.norm(loc = xi, scale = np.sqrt(l)).cdf(1) - st.norm(loc = xi, scale = np.sqrt(l)).cdf(-1))
			pi_old2 = np.sqrt(6*np.pi*l**2) * xi * (st.norm(loc = xi, scale = np.sqrt(l)).cdf(1) - st.norm(loc = xi, scale = np.sqrt(l)).cdf(-1)) - np.sqrt(3.) * l**2 * (np.exp(- (1-xi)**2 / (2*l**2)) - np.exp(- (1+xi)**2 / (2*l**2)))
			all_pis += [pi_old1, pi_old2]
			for i in range(2, n+1):
				js = range(i-2,-1, -2)[::-1]
				pi_new = C - np.sqrt((2*i+1)*(2*i-1)) * (xi / float(i)) * all_pis[-1] - (float(i-1)/float(i))*np.sqrt(float(2*i+1)/float(2*i-3)) * all_pis[-2]
				
				pi_new += l**2 * np.sqrt(2*i+1) * (float(2*i-1)/float(i))* np.sum([all_pis[js[j]] * np.sqrt(2*js[j]+1) for j in range(len(js))]) 
				all_pis += [pi_new]
				assert len(all_pis) == i+1
			return all_pis[-1]

	def pi_L_quad(self, n, xi, l, quad_x, quad_w):
		#print l
		#[x, w] = util.QuadratureRule('CC').get_rule(1, n)
		return 2. * np.dot((quad_w * np.exp(- (quad_x.flatten() - xi)**2 / (2*l**2)) ), chaos.Legendre1d(n)(quad_x) )


	def rho_H(self, n, m, l):
		return (-1.)**m * ( 2.*np.pi* ( l**2 / (l**2 + 1.) ) * float(math.factorial(n+m)) / (math.factorial(n)*math.factorial(m)) )**(1./2.) * self.pi_H(n+m, 0., l, var = True)

	def rho_L_quad(self, n, m, l, quad_x, quad_w):
		#[x, w] = util.QuadratureRule('CC').get_rule(2, np.max([n,m]))
		if n == 0:
			psi_a = np.ones(quad_x[:,0].shape)
		else:
			psi_a = chaos.Legendre1d(n)(quad_x[:,0])[:,-1]
		if m == 0:
			psi_b = np.ones(quad_x[:,1].shape)
		else:
			psi_b = chaos.Legendre1d(m)(quad_x[:,1])[:,-1]
		return 4 * np.sum([ psi_a[i] * psi_b[i] * np.exp(- (quad_x[i,0] - quad_x[i,1])**2 / (2*l**2)) * quad_w[i] for i in range(quad_x.shape[0])] )

	def v_all(self, pol = 'H'):	
		
		assert pol in ['H', 'L']
		ell = self._kern._lengthscale
		mi = chaos.PolyBasis().mi_terms(self._input_dim, self._pol_order)
		v_all = np.ones((mi.shape[0], self._X.shape[0]))

		if pol == 'H':
			#print ell
			C = self._kern._var / np.sqrt(2.*np.pi) ** (self._input_dim)
			for i in range(mi.shape[0]):
				for j in range(self._X.shape[0]):
					for l in range(self._input_dim):
						v_all[i,j] *= self.pi_H(mi[i,l], self._X[j,l], ell[l])
					v_all[i,j] *= C
			return v_all
		else:
			C = self._kern._var / (2** self._input_dim)
			PIs = np.zeros((self._pol_order+1, self._X.shape[0], self._X.shape[1]))
			order = np.min([self._pol_order, 10])
			quad = util.QuadratureRule('CC').get_rule(1, order)

			for i in range(self._X.shape[0]):
				for j in range(self._X.shape[1]):
					PIs[:,i,j] = self.pi_L_quad(self._pol_order, self._X[i,j], ell[j], quad[0], quad[1])
			for i in range(mi.shape[0]):
				for j in range(self._X.shape[0]):
					for l in range(self._X.shape[1]):
						v_all[i,j] *= PIs[mi[i,l], j, l]
					v_all[i,j] *= C
			return v_all

	def u_all(self, pol = 'H'):

		assert pol in ['H', 'L']
		order = np.min([self._pol_order, 10])

		ell = self._kern._lengthscale
		mi = chaos.PolyBasis().mi_terms(self._input_dim, self._pol_order)
		u_all = np.ones(mi.shape[0])
		if pol == 'H':
			C = self._kern._var / (2. * np.pi) ** self._input_dim
			for i in range(mi.shape[0]):
				for j in range(mi.shape[1]):
					u_all[i] *= self.rho_H(mi[i,j], mi[i,j], ell[j])
				u_all[i] *= C
			return u_all
		else:
			quad = util.QuadratureRule('CC').get_rule(2, order)
			C = self._kern._var / (2. ** (2*self._input_dim))
			for i in range(mi.shape[0]):
				for j in range(mi.shape[1]):
					u_all[i] *= self.rho_L_quad(mi[i,j], mi[i,j], ell[j], quad[0], quad[1])
				u_all[i] *= C
			return u_all 


	def predict(self, pol = 'H'):
		assert pol in ['H', 'L']
		V = self.v_all(pol)
		cov = self._kern.cov(self._X) + np.diag( 1e-8 * np.ones(self._X.shape[0]))
		L = np.linalg.cholesky(cov)
		scaled_data = np.linalg.solve(L.T, np.linalg.solve(L, self._Y[:,0])).flatten()
		VCV = np.dot( V, np.linalg.solve(L.T, np.linalg.solve(L, V.T)) )

		return np.dot(V, scaled_data), 0.#self.u_all(pol) - np.diag(VCV)

#	def mu(alpha, sig, Xi):


#	def covar_entry(alpha, sig, ):

#def mean_comp(n)