import numpy as np 
import chaos_toolbox as ct
import matplotlib.pyplot as plt 

Q = 10

[x_quad5, w5] = ct.util.QuadratureRule('CC').get_rule(2,6)
[x_quad6, w6] = ct.util.QuadratureRule('CC').get_rule(2,7)
[x_quad7, w7] = ct.util.QuadratureRule('CC').get_rule(2,8)
y_quad5 = np.load('../data_concentrations_CC_quadrature_lev6.npy')[:,-1]
y_quad6 = np.load('../data_concentrations_CC_quadrature_lev7.npy')[:,-1]
y_quad7 = np.load('../data_concentrations_CC_quadrature_lev8.npy')[:,-1]
print y_quad5.shape, y_quad6.shape, y_quad7.shape
coeffs5 = ct.chaos.PolyChaos().comp_coeffs(x_quad5, y_quad5, w5, Q, 'L')
coeffs6 = ct.chaos.PolyChaos().comp_coeffs(x_quad6, y_quad6, w6, Q, 'L')
coeffs7 = ct.chaos.PolyChaos().comp_coeffs(x_quad7, y_quad7, w7, Q, 'L')

c_n = coeffs6.shape[0]

all_m = np.load('U_all_m.npy')
#all_C = np.load('all_C.npy')

#for i in range(5):
#	plt.plot(range(100), all_m[:100,i], '-x')
#	plt.fill_between(range(100), all_m[:100,i]-2*np.sqrt(all_C[:100,i]), all_m[:100,i]+2*np.sqrt(all_C[:100,i]) )
#plt.plot(all_m[:100,:100], 'r-x')
#for i in range(10):
#	plt.plot(range(100), coeffs[i]* np.ones(100), 'k-', alpha = 0.7)

pol = ct.chaos.PolyBasis(2, Q, 'L')
xi = np.random.normal(size = (10000,2))
x = np.linspace(-1., 1., 50)
y = np.linspace(-1., 1., 50)
xx, yy = np.meshgrid(x, y)
X_test = np.hstack([xx.flatten().reshape(2500,1), yy.flatten().reshape(2500,1)])

P = pol(X_test)

y_bayes_1 = np.dot(P, all_m[10,:c_n])
y_bayes_2 = np.dot(P, all_m[20,:c_n])
y_bayes_3 = np.dot(P, all_m[-1,:c_n])
y5 = np.dot(P, coeffs5)
y6 = np.dot(P, coeffs6)
y7 = np.dot(P, coeffs7)
#print all_m[-1,:] - coeffs
#plt.show()
fig = plt.figure(figsize = (5,5))
#ax1 = fig.add_subplot(321)
plt.contourf(x, y, y_bayes_1.reshape(50,50), alpha = 0.8)
plt.xlabel(r'$\xi_1$', fontsize = 12)
plt.ylabel(r'$\xi_2$', fontsize = 12)
plt.savefig('./images/QoI_bayes_100_U.ps')
plt.savefig('./images/QoI_bayes_100_U.png')
plt.show()
#ax2 = fig.add_subplot(322)
fig = plt.figure(figsize = (5,5))
plt.contourf(x, y, y5.reshape(50,50), alpha = 0.8)
plt.xlabel(r'$\xi_1$', fontsize = 12)
plt.xlabel(r'$\xi_2$', fontsize = 12)
#plt.colorbar()
plt.savefig('./images/QoI_quad_lev6_U.ps')
plt.savefig('./images/QoI_quad_lev6_U.png')
plt.show()
#ax3 = fig.add_subplot(323)
fig = plt.figure(figsize = (5,5))
plt.contourf(x, y, y_bayes_2.reshape(50,50), alpha = 0.8)
plt.xlabel(r'$\xi_1$', fontsize = 12)
plt.ylabel(r'$\xi_2$', fontsize = 12)
plt.savefig('./images/QoI_bayes_200_U.ps')
plt.savefig('./images/QoI_bayes_200_U.png')
plt.show()
#ax4 = fig.add_subplot(324)
fig = plt.figure(figsize = (5,5))
plt.contourf(x, y, y6.reshape(50,50), alpha = 0.8)
plt.xlabel(r'$\xi_1$', fontsize = 12)
plt.ylabel(r'$\xi_2$', fontsize = 12)
#plt.colorbar()
plt.savefig('./images/QoI_quad_lev7_U.ps')
plt.savefig('./images/QoI_quad_lev7_U.png')
plt.show()
#ax5 = fig.add_subplot(325)
fig = plt.figure(figsize = (5,5))
plt.contourf(x, y, y_bayes_3.reshape(50,50), alpha = 0.8)
plt.xlabel(r'$\xi_1$', fontsize = 12)
plt.ylabel(r'$\xi_2$', fontsize = 12)
plt.savefig('./images/QoI_bayes_300_U.ps')
plt.savefig('./images/QoI_bayes_300_U.png')
plt.show()
#ax6 = fig.add_subplot(326)
fig = plt.figure(figsize = (5,5))
plt.contourf(x, y, y7.reshape(50,50), alpha = 0.8)
plt.xlabel(r'$\xi_1$', fontsize = 12)
plt.ylabel(r'$\xi_2$', fontsize = 12)
#plt.colorbar()
plt.savefig('./images/QoI_quad_lev8_U.ps')
plt.savefig('./images/QoI_quad_lev8_U.png')
plt.show()

