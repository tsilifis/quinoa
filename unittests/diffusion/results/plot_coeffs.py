import numpy as np 
import matplotlib.pyplot as plt 
import chaos_toolbox as ct

#Q = 20 # For hermite
Q = 10 # For Legendre

all_m = np.load('U_all_m.npy')
[x_quad, w] = ct.util.QuadratureRule('CC').get_rule(2,8)
print x_quad.shape[0]
y_quad = np.load('../data_concentrations_CC_quadrature_lev8.npy')[:,-1]
coeffs7 = ct.chaos.PolyChaos().comp_coeffs(x_quad, y_quad, w, Q, 'L')
print ct.chaos.PolyBasis(2, 6).mi_terms(6,2).shape
c_N = coeffs7.shape[0]

#plt.plot(all_m[20,:45], 'x')
#plt.plot(coeffs7[:45], '.')
#plt.show()

plt.plot(all_m[20,28:c_N], 'x')
plt.plot(coeffs7[28:], '.', label = 'Lev 8 (705 points)')
plt.xlabel('coefficient index', fontsize = 12)
plt.ylabel('Value', fontsize = 12)
plt.legend()
plt.tight_layout()
plt.savefig('./images/coeffs_lev8_U.png')
plt.savefig('./images/coeffs_lev8_U.ps')
plt.show()


sig = np.load('../sig_smooth_300.npy')
ell = np.load('../ell.npy')

fig = plt.figure()

ax1 = fig.add_subplot(111)
ax1.set_xlabel('Number of samples', fontsize = 12)
ax1.set_ylabel(r'$\sigma^2$', fontsize = 12)
ax1.yaxis.label.set_color('black')
line1 = ax1.plot(sig, color = 'C2', label = r'variance')
plt.legend(loc = 1)
#plt.style.use('seaborn-dark-palette')
ax2 = fig.add_subplot(111, sharex = ax1, frameon = False)
line2 = ax2.plot(ell, label = 'lengthscale')
ax2.set_ylabel(r'$\ell$', fontsize = 12)
ax2.yaxis.tick_right()
ax2.yaxis.set_label_position('right')
#ax2.yaxis.label.set_color('red')
plt.legend(loc = 3)
#line1 = ax1.plot(sig)
#plt.savefig('./images/sig_ell_plot.ps')
#plt.savefig('./images/sig_ell_plot.png')
plt.show()