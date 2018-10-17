import numpy as np 
import kernel_py as kp
import theano 
import theano.tensor as T 
from theano import pp

class HeatL(object):
	"""
	Class representing the differential operator kernel function.
	"""

	dim = None

	kernel = None

	def __init__(self, dim = 1, kernel = None):
		assert isinstance(dim, int)
		assert dim > 0
		self._dim = dim
		if kernel is None:
			self._kernel = kp.RBF(1)
		else:
			assert isinstance(kernel, Kernel)
			self.kernel = kernel


	def L_eval(self, x, y, arg = 1):

		def a(x0): 
			return np.arctan(20*(x0-1.)) / 2. + 1.

		assert arg in [-1,1]
		r = T.dscalar('x')
		s = a(r)
		gs = T.grad(s, r)
		d_a = theano.function([r], gs)
		if arg == 1:
			u = self.kernel.eval(r, y)
			gu = T.grad(u, r)
			d_u = theano.function([r], gu)
			ggu = T.grad(gu, r)
			d2_u = theano.function([r], ggu)
			return - d_a(x) * d_u(x) - a(x) * d2_u(x) - theano.function([r], u)(x) / 2.
		else:
			u = self.kernel.eval(x, r)
			gu = T.grad(u, r)
			d_u = theano.function([r], gu)
			ggu = T.grad(gu, r)
			d2_u = theano.function([r], ggu)
			return - d_a(y) * d_u(y) - a(y) * d2_u(y) - theano.function([r], u)(y) / 2.

	def LL_eval(self, x, y):
		def a(x0): 
			return np.arctan(20*(x0-1.)) / 2. + 1.

		r = T.dscalar('x')
		t = T.dvector('y')
		s = a(r)
		gs = T.grad(s, r)
		d_a = theano.function([r], gs)
		u = self.kernel.eval(t[0], t[1])
		#v = self.kernel.eval(t[0], t[1])
		gu = T.grad(u, t)
		#gv = T.grad(v, r)
		d_u = theano.function([t], gu)
		#d_v = theano.function([r], gv)
		ggu, updates = theano.scan(lambda i, gu, t : T.grad(gu[i], t), sequences=T.arange(gu.shape[0]), non_sequences=[gu, t])
		d2_u = theano.function([t], ggu, updates=updates)
		#ggu = T.grad(gu, t)
		#g2u_gxy = np.diag(d2_u(t))
		gg2u_g2xgy, updates2 = theano.scan(lambda i, g2u_gxy, t: T.grad(ggu[i,i], t), sequences = T.arange(ggu.shape[0]), non_sequences = [ggu, t])
		d3u_dxy = theano.function([t], gg2u_g2xgy, updates = updates2)

		#g3u_dxy = d3u_dxy(t)[0,1]
		g4u_g4xy = T.grad(gg2u_g2xgy[0,1], t)
		d4u_dx2y2 = theano.function([t], g4u_g4xy)
		#ggv = T.grad(gv, r)
		#d2_u = theano.function([t], ggu)
		#d2_v = theano.function([r], ggv)
		final = a(x) * a(y) * d4u_dx2y2([x,y])[1] + a(x) * d_a(y) * d3u_dxy([x,y])[0,1] + d_a(x) * a(y) * d3u_dxy([x,y])[1,0] + d_a(x) * d_a(y) * d2_u([x,y])[0,1] + 0.5 * d_a(x) * d_u([x,y])[0] + 0.5 * a(x) * d2_u([x,y])[0,0] + 0.5 * (d_a(y) * d_u([x,y])[1] + a(y) * d2_u([x,y])[1,1] + theano.function([t], u)([x,y]) / 2.)
		return final
		#return - d_a(x) * d_u([x,y])[0] - a(x) * d2_u([x,y])[0,0] - theano.function([t], u)([x,y]) / 2.

	def B_eval(self, x, y, arg = 1):

		assert arg in [-1, 1]
		r = T.dscalar('r')
		if arg == 1:
			u = self.kernel.eval(r,y)
			gu = T.grad(u, r)
			d_u = theano.function([r], gu)
		else:
			u = self.kernel.eval(x, r)
			gu = T.grad(u, r)
			d_u = theano.function([r], gu)

		return d_u(x) * (x == 0) + theano.function([r], u)(x) * (x == 3.)


	def BB_eval(self, x, y):

		r = T.dvector('r')
		u = self.kernel.eval(r[0], r[1])
		gu = T.grad(u, r)
		d_u = theano.function([r], gu)
		ggu, updates = theano.scan(lambda i, gu, r : T.grad(gu[i], r), sequences=T.arange(gu.shape[0]), non_sequences=[gu, r])
		d2_u = theano.function([r], ggu, updates=updates)
		return d2_u([x,y])[0,0] * (x == 0) * (y == 0) + d_u([x,y])[0] * (x == 0) * (x == 3) + d_u([x,y])[1] * (x == 3) * (x == 0) + theano.function([r], u)([x,y]) * (x == 3) * (y == 3)


	def LB_eval(self, x, y):

		def a(x0): 
			return np.arctan(20*(x0-1.)) / 2. + 1.

		r = T.dscalar('x')
		t = T.dvector('y')
		s = a(r)
		gs = T.grad(s, r)
		d_a = theano.function([r], gs)
		u = self.kernel.eval(t[0], t[1])
		gu = T.grad(u, t)
		d_u =theano.function([t], gu)
		ggu, updates = theano.scan(lambda i, gu, t : T.grad(gu[i], t), sequences=T.arange(gu.shape[0]), non_sequences=[gu, t])
		d2_u = theano.function([t], ggu, updates=updates)
		gg2u_g2xgy, updates2 = theano.scan(lambda i, g2u_gxy, t: T.grad(ggu[i,i], t), sequences = T.arange(ggu.shape[0]), non_sequences = [ggu, t])
		d3u_dxy = theano.function([t], gg2u_g2xgy, updates = updates2)
		return (-d_a(x) * d2_u([x,y])[0,1] - a(x) * d3u_dxy([x,y])[0,1] - 0.5 *  d_u([x,y])[1]) * (y == 0) + ( - d_a(x) * d_u([x,y])[0] - a(x) * d2_u([x,y])[0,0] - 0.5 * theano.function([t], u)([x,y]) ) * (y == 3)

#class HeatB(object):
	"""
	"""

#	dim = None 

#	kernel = None 

#	def __init__(self, dim = 1, kernel = None):
#		assert isinstance(dim, int)
#		assert dim > 0
#		self._dim = dim
#		if kernel is None:
#			self._kernel = kp.RBF(1)
#		else:
#			assert isinstance(kernel, Kernel)
#			self.kernel = kernel




#L = HeatL()
#L.kernel = kp.RBF(1)

#X = np.linspace(0, 3, 42)[1:41]
#Y = np.array([0., 3.])
#L_vals = np.zeros(vals.shape)
#B_vals = np.zeros(vals.shape)

#LL_vals = np.zeros((X.shape[0], X.shape[0]))
#LB_vals = np.zeros((X.shape[0], Y.shape[0]))
#BB_vals = np.zeros((Y.shape[0], Y.shape[0]))



"""
for i in range(X.shape[0]):
	for j in range(X.shape[0]):
		LL_vals[i] = L.LL_eval(X[i], X[j])
	print i
	
print 'Done with LL '
np.save('./test_Heat/LL.npy', LL_vals)
	
for i in range(X.shape[0]):
	for j in range(Y.shape[0]):
		LB_vals[i] = L.LB_eval(X[i],Y[j])
	print i

print 'Done with LB'
np.save('./test_Heat/LB.npy', LB_vals)

for i in range(Y.shape[0]):
	for j in range(Y.shape[0]):
		BB_vals[i] = L.BB_eval(Y[i], Y[j])

np.save('./test_Heat/BB.npy', BB_vals)
"""

#import matplotlib.pyplot as plt 

#plt.plot(vals, L_vals, '-o')
#plt.plot(vals, dL_vals, '-x')
#plt.show()