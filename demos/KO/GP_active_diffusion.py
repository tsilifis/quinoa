import numpy as np 
import kernel_py as kp
import scipy.stats as st
import matplotlib.pyplot as plt
import GPy as gpy
from scipy import linalg
import fipy as fp

#np.random.seed(31051985)


def f0(y):
    assert y.shape[0] == 3
    f = np.zeros((3,1))
    f[0,0] = y[1] * y[2]
    f[1,0] = y[0] * y[2]
    f[2,0] = - 2. * y[0] * y[1]
    return f

def f1(y):
    assert y.shape[0] == 3
    f = np.zeros((3,1))
    f[0,0] = y[0] * y[2]
    f[1,0] = - y[1] * y[2]
    f[2,0] = - y[0]**2 + y[1]**2
    return f
#def f(y0, t):
#    kappa = np.array([1., 1., -2.])
#    r = ode(f0).set_integrator('dopri5')
#    r.set_initial_value(y0, 0).set_f_params(kappa)
##    y_m = [y0[None, :]]
#    for tt in t[1:]:
#        r.integrate(tt)
#        y_m.append(r.y[None, :])
#    y_m = np.vstack(y_m)
#    return y_m#.flatten()


def RK4(y0, T, N):

    h = T / (N+1)
    y = np.zeros((3, N+1))
    y[:,0] = y0
    for i in range(1,N+1):
        k1 = f1(y[:,i-1])
        k2 = f1(y[:,i-1] + h*k1.flatten() / 2)
        k3 = f1(y[:,i-1] + h*k2.flatten() / 2)
        k4 = f1(y[:,i-1] + h*k3.flatten())
        y[:,i] = y[:,i-1] + h * (k1 + 2*k2 + 2*k3 + k4).flatten() / 6.
    return y






X = 2 * st.uniform.rvs(size = (4,2)) - 1.
Y = np.zeros((X.shape[0],1))

N = 1000
T = 10.
for i in range(X.shape[0]):
	y0 = np.zeros(3)
	y0[0] = 1.
	y0[1] = 0.1 * X[i, 0]
	y0[2] = X[i,1]
	Y[i,0] = RK4(y0, T, N)[1,-1]# + 0.1 * np.random.normal(size = (3,1))

kern = kp.RBF(2, 1, 1)
ker = gpy.kern.RBF(2, 1, 1)

m = gpy.models.GPRegression(X, Y, ker)


gp = kp.GP(X, Y, kern)

#x = np.linspace(-4., 4., 100).reshape(100,1)
x = np.linspace(-1, 1., 50)
y = np.linspace(-1, 1., 50)
xx, yy = np.meshgrid(x, y)
X_test = np.hstack([xx.flatten().reshape(2500,1), yy.flatten().reshape(2500,1)])

f, var = gp.predict(X_test)

m.optimize(messages = True)

print '-' * 30
print m.kern.lengthscale[0], m.kern.variance[0], m.likelihood.gaussian_variance()[0]
print '-' * 30


#m.plot()
#plt.show()
N_quad = 300

#print gp._kern._iso
gp.optimize()
f, var = gp.predict(X_test)
fig1 = plt.figure()
ax2 = fig1.add_subplot(111)
ax2.contourf(xx, yy, f.reshape(50,50), 30)
#ax2.fill_between(x[:,0], f - 2*np.sqrt(np.diag(var)), f + 2*np.sqrt(np.diag(var)), alpha = 0.5)
ax2.plot(X[:,0], X[:,1], 'wo')
#plt.colorbar()
plt.show()
#
sig = np.zeros(301)
sig_noise = np.zeros(301)
ell = np.zeros(301)
sig[0] = gp._kern._var
sig_noise[0] = gp._noise_var
ell[0] = gp._kern._lengthscale[0]
for i in range(300):
	x_new = gp.argmaxvar()
	print 'New design :' + str(x_new)
	print x_new.shape
	y0 = np.zeros(3)
	y0[0] = 1.
	y0[1] = 0.1 * x_new[0]
	y0[2] = x_new[1]
	y_new = RK4(y0, T, N)[1,-1].reshape((1,1))
	X = np.vstack([X, x_new])
	Y = np.vstack([Y, y_new])
	
	gp_new = kp.GP(X, Y, kern)
	gp_new.optimize()
	#gp_new._kern._lengthscale
	sig[i+1] = gp_new._kern._var
	sig_noise[i+1] = gp_new._noise_var
	ell[i+1] = gp_new._kern._lengthscale[0]
	gp = gp_new


#print gp._log_marginal_likelihood
#print m._log_marginal_likelihood
	#x =  np.linspace(np.min([x.min(), x_new[0]]), np.max([x.max(), x_new[0]]), 100).reshape(100,1)
	
	

	f, var = gp_new.predict(X_test)

	#fig1 = plt.figure(figsize = (11,5))
	#ax1 = fig1.add_subplot(121)
	#ax1.contourf(xx, yy, f.reshape(50,50), 30)
	#ax1.plot(X[:,0], X[:,1], 'wo')
	#ax2 = fig1.add_subplot(122)
	#ax2.contourf(xx, yy, np.diag(var).reshape(50,50), 30)
	#ax2.plot(X[:,0], X[:,1], 'wo')
	#plt.show()

	if i % 100 == 0:
		np.save('sig_batch_'+str(i)+'.npy', sig)
		np.save('ell_batch_'+str(i)+'.npy', ell)
		np.save('sig_noise_batch_'+str(i)+'.npy', sig_noise)
		np.save('X_batch_'+str(i)+'.npy', X)
		np.save('Y_batch_'+str(i)+'.npy', Y)

np.save('sig.npy', sig)
np.save('sig_noise.npy', sig_noise)
np.save('ell.npy', ell)
np.save('X.npy', X)
np.save('Y.npy', Y)
#print gp.log_marginal_likelihood(np.array([m.kern.lengthscale[0], m.kern.variance[0], m.likelihood.gaussian_variance()[0]]))
