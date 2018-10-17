import numpy as np 
import kernel_py as kp
import matplotlib.pyplot as plt
import GPy as gpy
from scipy import linalg
#np.random.seed(31051985)

X = np.random.normal(scale = 1, size = (100,2))

Y = (np.sin(X[:,0]) - np.sin(X[:,1])).reshape(X.shape[0],1)

kern = kp.RBF(2, 1, [1,1])
ker = gpy.kern.RBF(2, 1, [1, 1], ARD = True)

print kern._n_params

m = gpy.models.GPRegression(X, Y, ker)


gp = kp.GP(X, Y, kern)

#x = np.linspace(-4., 4., 100).reshape(100,1)


#f, var = gp.predict(x)


#fig = plt.figure(tight_layout = True)
#ax = fig.add_subplot(111)
#ax.plot(x, f, '-')
#ax.fill_between(x[:,0], f - 2*np.sqrt(np.diag(var)), f + 2*np.sqrt(np.diag(var)), alpha = 0.5)
#ax.plot(X[:,0], Y[:,0], 'x')
#ax.set_xlim([-4, 4])
#plt.show()
m.optimize(messages = True)

print '-' * 60
print m.kern.lengthscale[0], m.kern.lengthscale[1], m.kern.variance[0], m.likelihood.gaussian_variance()[0]
print '-' * 60


m.plot()
#plt.show()

#print gp._kern._iso
gp.optimize()

#

print gp._log_marginal_likelihood
print m._log_marginal_likelihood


print gp.log_marginal_likelihood(np.array([m.kern.lengthscale[0], m.kern.lengthscale[1], m.kern.variance[0], m.likelihood.gaussian_variance()[0]]))
