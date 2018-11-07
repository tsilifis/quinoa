import numpy as np 
import quinoa as qu
import matplotlib.pyplot as plt
import GPy as gpy
from scipy import linalg
#np.random.seed(31051985)

X = np.random.normal(scale = 1, size = (10,1))

Y = np.sin(X) + 0.01 * np.random.normal(size = (10,1))

#kern = qu.RBF(1, 1, 1)
ker = gpy.kern.RBF(1, 1, 1)
kern = qu.RBF(1,1,1)

m = gpy.models.GPRegression(X, Y, ker)


gp = qu.GP(X, Y, kern)

x = np.linspace(-4., 4., 501).reshape(501,1)


f, var = gp.predict(x)

#x0 = np.array([np.random.normal( size = (2,))]).reshape((2,1))

#fig = plt.figure(tight_layout = True)
#ax = fig.add_subplot(111)
#ax.plot(x, f, '-')
#ax.fill_between(x[:,0], f - 2*np.sqrt(np.diag(var)), f + 2*np.sqrt(np.diag(var)), alpha = 0.5)
#ax.plot(X[:,0], Y[:,0], 'x')
#ax.set_xlim([-4, 4])
#plt.show()
m.optimize(messages = True)

print '-' * 30
print m.kern.lengthscale[0], m.kern.variance[0], m.likelihood.gaussian_variance()[0]
print '-' * 30


m.plot()
#plt.show()

#print gp._kern._iso
gp.optimize()

#
#print gp.argmaxvar()

print gp._log_marginal_likelihood
print m._log_marginal_likelihood

f, var = gp.predict(x)
print np.diag(var)[:10]
z = gp.sample(x, 5)#[:,0]


fig1 = plt.figure()
ax2 = fig1.add_subplot(111)
ax2.plot(x, f, '-')
ax2.plot(x, z, 'r--', linewidth = 1.)
ax2.fill_between(x[:,0], f - 2*np.sqrt(np.diag(var)), f + 2*np.sqrt(np.diag(var)), alpha = 0.5)
ax2.plot(X[:,0], Y[:,0], 'x')
plt.show()
#print gp.log_marginal_likelihood(np.array([m.kern.lengthscale[0], m.kern.variance[0], m.likelihood.gaussian_variance()[0]]))
