import numpy as np
import kernel_py as kp
import scipy.stats as st 
from scipy import linalg
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#%matplotlib inline


def build_up_b(b, rho, dt, u, v, dx, dy):
	
	b[1:-1, 1:-1] = (rho * ( (1. / dt) * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2. * dx) + 
					(v[2:, 1:-1] - v[0:-2, 1:-1]) / (2. * dy)) - ((u[1:-1,2:] - u[1:-1,0:-2]) / (2. * dx)) ** 2 -  
					2. * ((u[2:,1:-1] - u[0:-2,1:-1]) / (2. * dy)) * ((v[1:-1, 2:] - v[1:-1,0:-2]) / (2. * dx)) - 
		 			((v[2:, 1:-1] - v[0:-2, 1:-1]) / (2. * dy))**2 ))

	return b

def pressure_poisson(p, dx, dy, b):
	pn = np.empty_like(p)
	pn = p.copy()
	for q in range(nit):
		pn = p.copy()
		p[1:-1, 1:-1] = ( ( (pn[1:-1, 2:] + pn[1:-1, 0:-2]) * dy**2 + (pn[2:, 1:-1] + pn[0:-2, 1:-1]) * dx**2) / (2. * (dx**2 + dy**2)) - dx**2 * dy**2 * b[1:-1,1:-1] / (2. * (dx**2 + dy**2)))

		p[:,-1] = p[:, -2] ## dp/dy = 0 at x = 2
		p[0, :] = p[1, :] ## dp/dy = 0 at y = 0
		p[:, 0] = p[:, 1] ## dp/dx = 0 at x = 0
		p[-1,:] = 0. ## p = 0 at y = 2

	return p

def cavity_flow(nt, u, v, dt, dx, dy, p, rho, xi):
	un = np.empty_like(u)
	vn = np.empty_like(v)
	b = np.zeros((ny, nx))

	nu = xi[-1]
	for n in range(nt):
		un = u.copy()
		vn = v.copy()
		b = build_up_b(b, rho, dt, u, v, dx, dy)
		p = pressure_poisson(p, dx, dy, b)
		#print p
		u[1:-1, 1:-1] = (un[1:-1, 1:-1] - un[1:-1, 1:-1] * (dt / dx) * (un[1:-1, 1:-1] - un[1:-1, 0:-2]) - 
						vn[1:-1, 1:-1] * (dt / dy) * (un[1:-1, 1:-1] - un[0:-2,1:-1]) - 
						(dt / (2.*rho*dx)) * (p[1:-1,2:] - p[1:-1,0:-2]) + 
					    nu * ( (dt / dx**2) * (un[1:-1, 2:] - 2 * un[1:-1, 1:-1] + un[1:-1, 0:-2]) + 
					    (dt / dy**2) * (un[2:, 1:-1] - 2. * un[1:-1,1:-1] + un[0:-2, 1:-1]) ) )

		v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - un[1:-1, 1:-1] * (dt / dx) * (vn[1:-1, 1:-1] - vn[1:-1, 0:-2]) - 
						vn[1:-1, 1:-1] * (dt / dy) * (vn[1:-1, 1:-1] - vn[0:-2, 1:-1]) - 
						(dt / (2.*rho*dy)) * (p[2:, 1:-1] - p[0:-2, 1:-1]) + 
						nu * ( (dt / dx**2) * (vn[1:-1, 2:] - 2 * vn[1:-1, 1:-1] + vn[1:-1, 0:-2]) + 
						(dt / dy**2) * (vn[2:, 1:-1] - 2. * vn[1:-1, 1:-1] + vn[0:-2, 1:-1]) ) )

		u[0, :] = -2 * xi[0] * np.sin(xi[2] * np.pi * np.linspace(0., 2., int(2/dx + 1)))
		u[:, 0] = 0.
		u[:, -1] = 0.
		u[-1, :] = 2 * xi[1] * np.sin(xi[3] * np.pi * np.linspace(0., 2., int(2/dx + 1))) # set velocity on cavity lid equal to 1
		v[0, :] = 0.
		v[-1, :] = 0.
		v[:, 0] = 0. # * np.exp(- xi[2] * np.linspace(0., 2., int(2/dx + 1)))
		v[:, -1] = 0. # * np.exp(- xi[3] * np.linspace(0., 2., int(2/dx + 1)))

	return u, v, p



nx = 101
ny = 101
#nt = 500
nit = 50
c = 1.
dx = 2. / (nx - 1)
dy = 2. / (ny - 1)

x = np.linspace(0, 2., nx)
y = np.linspace(0, 2., ny)
X, Y = np.meshgrid(x, y)

rho = 1.
#nu = .05
dt = .001


u = np.zeros((ny, nx))
v = np.zeros((ny, nx))
p = np.zeros((ny, nx))
b = np.zeros((ny, nx))
nt = 1000

dim = 5

#xi = st.uniform.rvs(size = (5,))
#xi[4] = xi[4] * 0.04 + 0.01
N_init = 20
XI = 2. * st.uniform.rvs(size = (N_init,dim)) - 1.
YI = np.zeros((N_init,1))

for i in range(XI.shape[0]):
	print 'Taking initial sample : ' + str(i)
	u = np.zeros((ny, nx))
	v = np.zeros((ny, nx))
	p = np.zeros((ny, nx))
	b = np.zeros((ny, nx))

	xi = 0.5 * (XI[i,:].copy() + 1.)

	xi[-1] = 0.04 * xi[-1] + 0.01 
	u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, xi)
	YI[i,0] = p[-2, -1]


print YI
kern = kp.RBF(dim, 1, 1)

gp = kp.GP(XI, YI, kern)

N_quad = 300
gp.optimize()

sig = np.zeros(N_quad + 1)
sig_noise = np.zeros(N_quad + 1)
ell = np.zeros(N_quad + 1)
sig[0] = gp._kern._var
sig_noise[0] = gp._noise_var
ell[0] = gp._kern._lengthscale[0]

kern._var = sig[0]
kern._lengthscale = [ell[0]] * dim

for i in range(N_quad):
	u = np.zeros((ny, nx))
	v = np.zeros((ny, nx))
	p = np.zeros((ny, nx))
	b = np.zeros((ny, nx))

	x_new = gp.argmaxvar((-1.,1.))
	print 'New design :' + str(x_new)
	print x_new.shape
	xi = 0.5 * (x_new.copy() + 1.)
	xi[-1] = 0.04 * xi[-1] + 0.01 
	print 'New input : ' + str(xi)
	u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, xi)
	#y_new = collect_data(x_new).reshape((1,1))# + 0.1 * np.random.normal(size = (1,1))
	y_new = p[-2, -1]
	XI = np.vstack([XI, x_new])
	YI = np.vstack([YI, y_new])
	
	gp_new = kp.GP(XI, YI, kern)
	gp_new._noise_var = gp._noise_var
	gp_new.optimize()
	#gp_new._kern._lengthscale
	sig[i+1] = gp_new._kern._var
	sig_noise[i+1] = gp_new._noise_var
	ell[i+1] = gp_new._kern._lengthscale[0]
	kern._var = sig[i+1]
	kern._lengthscale = [ell[i+1]] * dim
	gp = gp_new

	#f, var = gp_new.predict(X_test)

	if i % 50 == 0:
		np.save('sig_batch_'+str(i)+'.npy', sig)
		np.save('ell_batch_'+str(i)+'.npy', ell)
		np.save('sig_noise_batch_'+str(i)+'.npy', sig_noise)
		np.save('X_batch_'+str(i)+'.npy', XI)
		np.save('Y_batch_'+str(i)+'.npy', YI)
	print 'Took active data ' + str(i)


np.save('sig.npy', sig)
np.save('sig_noise.npy', sig_noise)
np.save('ell.npy', ell)
np.save('X.npy', XI)
np.save('Y.npy', YI)

#fig = plt.figure(figsize = (11, 7), dpi = 100)
# plotting the pressure field as a contour
#plt.contourf(X, Y, p, alpha = 0.5, cmap = cm.viridis)
#plt.colorbar()
# plotting the pressure field outlines
#plt.contour(X, Y, p, 30, cmap = cm.viridis)
# plotting velocity field
#plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.show()



