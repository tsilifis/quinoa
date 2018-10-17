"""
Generate data for the diffusion forward model.

Author:
    Panagiotis Tsilifis

Date:
    6/12/2014

"""

import numpy as np
import fipy as fp
import os
import matplotlib.pyplot as plt


def collect_data(xs):

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
    #xs_1 = np.array([0.09, 0.23])
    #xs_2 = np.array([0.89, 0.75])
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


#    if __name__ == '__main__':
#        viewer = fp.Viewer(vars=phi, datamin=0., datamax=2.)
#        viewer.plot()

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
        
#        fig = plt.figure()
#        ax1 = fig.add_subplot(2,2,i)
#        ax1.contourf(x, y, phi().reshape(101,101),200)
#        plt.plot(0.5, 0.98, 'wo', markersize = 10, mew = 2)
#        plt.plot(0.5, 0.001, 'wo', markersize = 10, mew = 2)
#        ax1.set_title('Concentration map at t = ' + str(time()), fontsize = 16)
        #fig.suptitle('Concentration at t = ' + str(time()))
#        ax1.set_xlabel(r'$\xi_1$', fontsize = 16)
#        ax1.set_ylabel(r'$\xi_2$', fontsize = 16)

#        for tick in ax1.xaxis.get_major_ticks():
#            tick.label.set_fontsize(16)
#        for tick in ax1.yaxis.get_major_ticks():
#            tick.label.set_fontsize(16)
    #    png_file = os.path.join('figures', 'concentration'+ str(i) +'.png')
    #    plt.savefig(png_file)
#        plt.colorbar()
        #plt.show()
            i = i + 1
    
    return data

        
import scipy.stats as st
from chaos_toolbox import util
[x_quad, wghts] = util.QuadratureRule('CC').get_rule(2, 4)
sources = (x_quad + 1) / 2.

#[x_quad, wghts] = util.QuadratureRule().get_rule(2, 7)
#sources = st.norm.cdf(x_quad)

#sources = st.uniform.rvs(size = (5,2))
observ = np.zeros((sources.shape[0], 16))

for i in range(sources.shape[0]):
    observ[i,:] = collect_data(sources[i,:])
    print i


#print observ
#if __name__ == '__main__':
#    raw_input("Transient diffusion with source term. Press <return> to proceed")

np.save('data_concentrations_CC_quadrature_lev4.npy', observ)