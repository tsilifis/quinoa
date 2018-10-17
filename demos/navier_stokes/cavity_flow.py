import numpy as np
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
		v[:, 0] = 0. #2 * np.exp(- xi[2] * np.linspace(0., 2., int(2/dx + 1)))
		v[:, -1] = 0. #2 * np.exp(- xi[3] * np.linspace(0., 2., int(2/dx + 1)))

	return u, v, p



nx = 101
ny = 101
nt = 500
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
nt = 1500
import scipy.stats as st 
#xi = st.uniform.rvs(size = (5,))
xi = np.zeros(5)
xi[0] = 1.
xi[1] = 1.
xi[2] = 1.
xi[3] = 1.
xi[-1] = 0.01

#xi[-1] = xi[-1] * 0.09 + 0.01

print 'Input point : ' + str(xi)
u, v, p = cavity_flow(nt, u, v, dt, dx, dy, p, rho, xi)

print 'QoI value : ' + str(p[-2,-1])
print p.argmax()

fig = plt.figure(figsize = (11, 7), dpi = 100)
# plotting the pressure field as a contour
plt.contourf(X, Y, p, alpha = 0.5, cmap = cm.viridis)
plt.colorbar()
# plotting the pressure field outlines
plt.contour(X, Y, p, 30, cmap = cm.viridis)
# plotting velocity field
plt.quiver(X[::2, ::2], Y[::2, ::2], u[::2, ::2], v[::2, ::2])
plt.xlabel('X')
plt.ylabel('Y')
plt.show()



