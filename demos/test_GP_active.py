import numpy as np 
import quinoa as qu
import matplotlib.pyplot as plt
import GPy as gpy
from scipy import linalg
#np.random.seed(31051985)

X = np.random.normal(scale = 1, size = (2,1))
#X = np.array([0, 1.]).reshape((2,1))

Y = np.sin(X)# + 0.1 * np.random.normal(size = (3,1))

kern = qu.RBF(1, 1, 1)
ker = gpy.kern.RBF(1, 1, 1)

m = gpy.models.GPRegression(X, Y, ker)


gp = qu.GP(X, Y, kern)

x = np.linspace(-4., 4., 100).reshape(100,1)


f, var = gp.predict(x)

#x0 = np.array([np.random.normal( size = (2,))]).reshape((2,1))

#fig = plt.figure(tight_layout = True)
#ax = fig.add_subplot(111)
#ax.plot(x, f, '-')
#ax.fill_between(x[:,0], f - 2*np.sqrt(np.diag(var)), f + 2*np.sqrt(np.diag(var)), alpha = 0.5)
#ax.plot(X[:,0], Y[:,0], 'x')
#ax.set_xlim([-4, 4])
#plt.show()
#m.optimize(messages = True)

#print '-' * 30
#print m.kern.lengthscale[0], m.kern.variance[0], m.likelihood.gaussian_variance()[0]
#print '-' * 30


#m.plot()
#plt.show()

#print gp._kern._iso
gp.optimize()
f, var = gp.predict(x)
fig1 = plt.figure()
ax2 = fig1.add_subplot(111)
ax2.plot(x, f, '-')
ax2.fill_between(x[:,0], f - 2*np.sqrt(np.diag(var)), f + 2*np.sqrt(np.diag(var)), alpha = 0.5)
ax2.plot(X[:,0], Y[:,0], 'x')
plt.show()
#
N_points = 15
sig = np.zeros(N_points)
sig_noise = np.zeros(N_points)
ell = np.zeros(N_points)
sig[0] = gp._kern._var
sig_noise[0] = gp._noise_var
ell[0] = gp._kern._lengthscale[0]
for i in range(N_points-1):
	x_new = gp.argmaxvar()
	print 'New design :' + str(x_new)
	print x_new.shape
	y_new = np.sin(x_new)# + 0.1 * np.random.normal(size = (1,1))
	X = np.vstack([X, x_new])
	Y = np.vstack([Y, y_new])
	
	gp_new = qu.GP(X, Y, kern)
	gp_new.optimize()
	#gp_new._kern._lengthscale
	sig[i+1] = gp_new._kern._var
	sig_noise[i+1] = gp_new._noise_var
	ell[i+1] = gp_new._kern._lengthscale[0]
	gp = gp_new


#print gp._log_marginal_likelihood
#print m._log_marginal_likelihood
	x =  np.linspace(np.min([x.min(), x_new[0]]), np.max([x.max(), x_new[0]]), 100).reshape(100,1)
	#x = np.linspace(np.min([x.min() x_new[0]]), np.max([x.max(), x_new[0]]), 100).reshape(100,1)

	f, var = gp_new.predict(x)

	fig1 = plt.figure()
	ax2 = fig1.add_subplot(111)
	ax2.plot(x, f, '-')
	ax2.fill_between(x[:,0], f - 2*np.sqrt(np.diag(var)), f + 2*np.sqrt(np.diag(var)), alpha = 0.5)
	ax2.plot(X[:,0], Y[:,0], 'x')
	plt.show()

np.save('sig.npy', sig)
np.save('sig_noise.npy', sig_noise)
np.save('ell.npy', ell)
np.save('X.npy', X)
np.save('Y.npy', Y)
#print gp.log_marginal_likelihood(np.array([m.kern.lengthscale[0], m.kern.variance[0], m.likelihood.gaussian_variance()[0]]))
