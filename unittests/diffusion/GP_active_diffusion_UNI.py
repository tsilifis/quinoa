import numpy as np 
import kernel_py as kp
import scipy.stats as st
import matplotlib.pyplot as plt
import GPy as gpy
from scipy import linalg
import fipy as fp

#np.random.seed(31051985)


def collect_data(x_source):

	#xs = st.norm.cdf(x_source)
	xs = (x_source + 1.)/ 2.
	# Make the source 

	nx = 51
	ny = nx
	dx = 1./51 
	dy = dx
	rho = 0.05
	q0 = 1. / (np.pi * rho ** 2)
	T = 0.3
	mesh = fp.Grid2D(dx=dx, dy=dy, nx=nx, ny=ny)
	xs_1 = xs
 
	time = fp.Variable()
	sourceTerm_1 = fp.CellVariable(name = "Source term", mesh=mesh, value = 0.)
	#sourceTerm_2 = fp.CellVariable(name = "Source term", mesh=mesh, value = 0.)
	for i in range(sourceTerm_1().shape[0]):
		sourceTerm_1()[i] = q0 * np.exp( - ((mesh.cellCenters[0]()[i] - xs_1[0]) ** 2 
			+ (mesh.cellCenters[1]()[i] - xs_1[1]) ** 2 ) / (2 * rho **2)) * (time() < T)
	#sourceTerm_2()[i] = q0 * np.exp( - ((mesh.cellCenters[0]()[i] - xs_2[0]) ** 2 
	#                              + (mesh.cellCenters[1]()[i] - xs_2[1]) ** 2 ) / (2 * rho **2)) * (time() < T)

	# The equation
	eq = fp.TransientTerm() == fp.DiffusionTerm(coeff=1.) + sourceTerm_1# + sourceTerm_2

	# The solution variable
	phi = fp.CellVariable(name = "Concentration", mesh=mesh, value=0.)

	x = np.arange(0,51.)/51
	y = x

	data = []
	dt = 0.005
	steps = 60
	i = 1
	for step in range(steps):
		time.setValue(time() + dt)
		eq.solve(var=phi, dt=dt)
		#if __name__ == '__main__':
			#viewer.plot()
		if step == 14 or step == 29 or step == 44 or  step == 59:
			dl = phi()[0]
			dr = phi()[50] # phi()[100]
			ul = phi()[2550] # phi()[10100]
			ur = phi()[2600] # phi()[10200]
			data = np.hstack([data, np.array([dl, dr, ul, ur])])
			#data = np.hstack([data, np.array([dc, uc])])

			i = i + 1

	return data[-1]



#X = np.random.normal(scale = 1, size = (4,2))
X = 2. * st.uniform.rvs(size = (4,2)) - 1.
Y = np.zeros((X.shape[0],1))

for i in range(X.shape[0]):
	Y[i,0] = collect_data(X[i,:])# + 0.1 * np.random.normal(size = (3,1))

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
	x_new = gp.argmaxvar((-1.,1.))
	print 'New design :' + str(x_new)
	print x_new.shape
	y_new = collect_data(x_new).reshape((1,1))# + 0.1 * np.random.normal(size = (1,1))
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

	if i % 50 == 0:
		np.save('UU_sig_batch_'+str(i)+'.npy', sig)
		np.save('UU_ell_batch_'+str(i)+'.npy', ell)
		np.save('UU_sig_noise_batch_'+str(i)+'.npy', sig_noise)
		np.save('UU_X_batch_'+str(i)+'.npy', X)
		np.save('UU_Y_batch_'+str(i)+'.npy', Y)

np.save('UU_sig.npy', sig)
np.save('UU_sig_noise.npy', sig_noise)
np.save('UU_ell.npy', ell)
np.save('UU_X.npy', X)
np.save('UU_Y.npy', Y)

#print gp.log_marginal_likelihood(np.array([m.kern.lengthscale[0], m.kern.variance[0], m.likelihood.gaussian_variance()[0]]))


