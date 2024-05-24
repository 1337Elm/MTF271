import os.path
from matplotlib import ticker
import scipy.io as sio
import sys
import numpy as np
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from grad_xyz import dphidx,dphidy,dphidz,compute_face,dphidx_2d,dphidy_2d,compute_face_2d,compute_geometry_2d

name_dat = "Assignment 2/Assignment 2b/data/"
plt.rcParams.update({'font.size': 22})
plt.rcParams.update({'figure.max_open_warning': 0})
plt.interactive(True)

re =9.36e+5 
viscos =1./re

datax= np.loadtxt(name_dat + "x2d_hump_IDDES.dat")
x=datax[0:-1]
ni=int(datax[-1])
datay= np.loadtxt(name_dat + "y2d_hump_IDDES.dat")
y=datay[0:-1]
nj=int(datay[-1])

x2d=np.zeros((ni+1,nj+1))
y2d=np.zeros((ni+1,nj+1))

x2d=np.reshape(x,(ni+1,nj+1))
y2d=np.reshape(y,(ni+1,nj+1))

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])


itstep,nk,dz=np.load(name_dat + 'itstep-hump-IDDES.npy')


p2d=np.load(name_dat + 'p_averaged-hump-IDDES.npy')/itstep            #mean pressure
u2d=np.load(name_dat + 'u_averaged-hump-IDDES.npy')/itstep            #streamwise mean velocity
v2d=np.load(name_dat + 'v_averaged-hump-IDDES.npy')/itstep            #streamwise mean velocity
w2d=np.load(name_dat + 'w_averaged-hump-IDDES.npy')/itstep            #streamwise mean velocity
k_model2d=np.load(name_dat + 'k_averaged-hump-IDDES.npy')/itstep      #mean modeled turbulent kinetic energy velocity
vis2d=np.load(name_dat + 'vis_averaged-hump-IDDES.npy')/itstep        #mean modeled total viscosity
uu2d=np.load(name_dat + 'uu_stress-hump-IDDES.npy')/itstep
vv2d=np.load(name_dat + 'vv_stress-hump-IDDES.npy')/itstep
ww2d=np.load(name_dat + 'ww_stress-hump-IDDES.npy')/itstep            #spanwise resolved normal stress
uv2d=np.load(name_dat + 'uv_stress-hump-IDDES.npy')/itstep
psi2d=np.load(name_dat + 'fk_averaged-hump-IDDES.npy')/itstep         #ratio of RANS to LES lengthscale
eps2d=np.load(name_dat + 'eps_averaged-hump-IDDES.npy')/itstep        #mean modeled dissipion of turbulent kinetic energy 
s2_abs2d=np.load(name_dat + 'gen_averaged-hump-IDDES.npy')/itstep     #mean |S| (used in Smagorinsky model, the production term in k-eps model, IDDES ...)
s_abs2d=s2_abs2d**0.5

uu2d=uu2d-u2d**2                                #streamwise resolved normal stress
vv2d=vv2d-v2d**2                                #streamwise resolved normal stress
uv2d=uv2d-u2d*v2d                               #streamwise resolved shear stress


kres2d=0.5*(uu2d+vv2d+ww2d)                     


x065_off=np.genfromtxt(name_dat + "x065_off.dat", dtype=None,comments="%")

cyclic_x=False
cyclic_z=True

fx2d,fy2d,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d= compute_geometry_2d(x2d,y2d,xp2d,yp2d)

dudx= np.zeros((ni,nj))
dudy= np.zeros((ni,nj))

u_face_w,u_face_s=compute_face_2d(u2d,'n','n','d','d',x2d,y2d,fx2d,fy2d,cyclic_x,ni,nj)
v_face_w,v_face_s=compute_face_2d(v2d,'n','n','d','d',x2d,y2d,fx2d,fy2d,cyclic_x,ni,nj)

dudy=dphidy_2d(u_face_w,u_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)
dvdx=dphidx_2d(v_face_w,v_face_s,areawx_2d,areawy_2d,areasx_2d,areasy_2d,vol_2d)

#T.4 Location of interface
c_my = 0.09
C_w = 0.15
C_des = 0.65
kappa = 0.41
omega = eps2d/(c_my*(kres2d + k_model2d))

h_max = np.max([np.max(x2d[:,0])/ni, np.max(y2d[0,:])/nj,dz])

alpha = 0.25 - yp2d/h_max

f_B = np.zeros((ni,nj))
for i in range(ni):
    for j in range(nj):
        f_B[i,j] = np.min([2*np.exp(-9*alpha[i,j]**2),1])

r_dt = (vis2d-viscos)/(kappa**2 *yp2d**2 *np.max([s_abs2d,10**(-10) * np.ones((ni,nj))]))
f_dt = 1 - np.tanh((8*r_dt)**3)

f_d_hat = np.zeros((ni,nj))
delta = np.zeros((ni,nj))
L_rans = np.zeros((ni,nj))
for i in range(ni):
    for j in range(nj):
        f_d_hat[i,j] = np.max([(1-f_dt[i,j]),f_B[i,j]])
        delta[i,j] = np.min([np.max([C_w*yp2d[i,j], C_w*h_max, np.max(yp2d[0,:])/nj ]),h_max])
        L_rans[i,j] = np.min([c_my* (kres2d[i,j] + k_model2d[i,j])**(3/2) / eps2d[i,j],C_des*delta[i,j]])

L_les = C_des*delta
L_iddes = f_d_hat*L_rans + (1-f_d_hat)*L_les


psi = L_rans/L_iddes

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xp2d[:,:], yp2d[:,:], psi2d, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax.set_zlabel("$\\psi$")
ax.set_ylabel("y")
ax.set_xlabel("x")
plt.title("$\\psi$" + " in domain")
plt.savefig("Assignment 2/Assignment 2b/psi3d.png")
fig = plt.figure()
plt.contourf(xp2d,yp2d,psi2d)
plt.colorbar()
plt.ylabel("y")
plt.xlabel("x")
plt.title("$\\psi$" + " in domain")
plt.savefig("Assignment 2/Assignment 2b/psi2d.png")


lines = [0.65, 0.8, 0.9, 1, 1.1, 1.2, 1.3]
indx1 = np.abs(xp2d[:,0]- np.repeat(lines[0],len(xp2d[:,0]))).argmin()
indx2 = np.abs(xp2d[:,0]- np.repeat(lines[1],len(xp2d[:,0]))).argmin()
indx3 = np.abs(xp2d[:,0]- np.repeat(lines[2],len(xp2d[:,0]))).argmin()
indx4 = np.abs(xp2d[:,0]- np.repeat(lines[3],len(xp2d[:,0]))).argmin()
indx5 = np.abs(xp2d[:,0]- np.repeat(lines[4],len(xp2d[:,0]))).argmin()
indx6 = np.abs(xp2d[:,0]- np.repeat(lines[5],len(xp2d[:,0]))).argmin()
indx7 = np.abs(xp2d[:,0]- np.repeat(lines[6],len(xp2d[:,0]))).argmin()
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(f_d_hat[indx1,:],yp2d[indx1,:],'b',label="x_1")
plt.plot(f_d_hat[indx2,:],yp2d[indx2,:],'r',label="x_2")
plt.plot(f_d_hat[indx3,:],yp2d[indx3,:],'g',label="x_3")


plt.xlabel("$y$")
plt.legend()
plt.title("$\widetilde{f}_d$")
plt.savefig('Assignment 2/Assignment 2b/f_d.png')



fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(f_d_hat[indx4,:],yp2d[indx4,:],'b',label="x_4")
plt.plot(f_d_hat[indx5,:],yp2d[indx5,:],'r',label="x_5")
plt.plot(f_d_hat[indx6,:],yp2d[indx6,:],'g',label="x_6")
plt.plot(f_d_hat[indx7,:],yp2d[indx7,:],'k',label="x_7")


plt.xlabel("$y$")
plt.legend()
plt.title("$\widetilde{f}_d$")
plt.savefig('Assignment 2/Assignment 2b/f_d_plot2.png')



#T.5
L_des = f_B*L_rans + (1-f_B)*L_les
L_ddes = L_rans - f_d_hat*np.max([np.zeros((ni,nj)),L_rans - L_les])



fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xp2d[:,:], yp2d[:,:], L_rans/L_des, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax.set_zlabel("$\\psi$")
ax.set_ylabel("y")
ax.set_xlabel("x")
plt.title("$\\psi$" + " in domain")
plt.savefig("Assignment 2/Assignment 2b/psi_DES_3d.png")
fig = plt.figure()
plt.contourf(xp2d,yp2d,L_rans/L_des)
plt.colorbar()
plt.ylabel("y")
plt.xlabel("x")
plt.title("$\\psi$" + " in domain")
plt.savefig("Assignment 2/Assignment 2b/psi_DES_2d.png")




fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xp2d[:,:], yp2d[:,:], L_rans/L_ddes, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax.set_zlabel("$\\psi$")
ax.set_ylabel("y")
ax.set_xlabel("x")
plt.title("$\\psi$" + " in domain")
plt.savefig("Assignment 2/Assignment 2b/psi_DDES_3d.png")
fig = plt.figure()
plt.contourf(xp2d,yp2d,L_rans/L_ddes)
plt.colorbar()
plt.ylabel("y")
plt.xlabel("x")
plt.title("$\\psi$" + " in domain")
plt.savefig("Assignment 2/Assignment 2b/psi_DDES_2d.png")



#T.7

#Viscos
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(((vis2d-viscos)/viscos)[indx1,:],yp2d[indx1,:],'b',label="x_1")
plt.plot(((vis2d-viscos)/viscos)[indx2,:],yp2d[indx2,:],'r',label="x_2")
plt.plot(((vis2d-viscos)/viscos)[indx3,:],yp2d[indx3,:],'g',label="x_3")


plt.ylabel("$y$")
plt.legend()
plt.title("$\\langle \\nu_t \\rangle / \\nu$")
plt.savefig('Assignment 2/Assignment 2b/viscos_node_1_3.png')

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(((vis2d-viscos)/viscos)[indx4,:],yp2d[indx4,:],'b',label="x_4")
plt.plot(((vis2d-viscos)/viscos)[indx5,:],yp2d[indx5,:],'r',label="x_5")
plt.plot(((vis2d-viscos)/viscos)[indx6,:],yp2d[indx6,:],'g',label="x_6")
plt.plot(((vis2d-viscos)/viscos)[indx7,:],yp2d[indx7,:],'k',label="x_7")


plt.ylabel("$y$")
plt.legend()
plt.title("$\\langle \\nu_t \\rangle / \\nu$")
plt.savefig('Assignment 2/Assignment 2b/viscos_node_4_7.png')


#Shear stress

uv_mod = -(vis2d-viscos)*(dudy + dvdx)
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(np.abs((uv2d/(uv2d + uv_mod)))[indx1,:],yp2d[indx1,:],'b',label="x_1")
plt.plot(np.abs((uv2d/(uv2d + uv_mod)))[indx2,:],yp2d[indx2,:],'r',label="x_2")
plt.plot(np.abs((uv2d/(uv2d + uv_mod)))[indx3,:],yp2d[indx3,:],'g',label="x_3")


plt.ylabel("$y$")
plt.legend()
plt.title("$|\\langle \\tau_{12} \\rangle/(\\langle \\tau_{12}\\rangle + \\langle \overline{v}_1'\overline{v}_2'\\rangle|$")
plt.savefig('Assignment 2/Assignment 2b/uv_node_1_3.png')

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(np.abs((uv2d/(uv2d + uv_mod)))[indx4,:],yp2d[indx4,:],'b',label="x_4")
plt.plot(np.abs((uv2d/(uv2d + uv_mod)))[indx5,:],yp2d[indx5,:],'r',label="x_5")
plt.plot(np.abs((uv2d/(uv2d + uv_mod)))[indx6,:],yp2d[indx6,:],'g',label="x_6")
plt.plot(np.abs((uv2d/(uv2d + uv_mod)))[indx7,:],yp2d[indx7,:],'k',label="x_7")


plt.ylabel("$y$")
plt.legend()
plt.title("$|\\langle \\tau_{12} \\rangle/(\\langle \\tau_{12}\\rangle + \\langle \overline{v}_1'\overline{v}_2'\\rangle|$")
plt.savefig('Assignment 2/Assignment 2b/uv_node_4_7.png')


#Kinetic energy
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot((k_model2d/(k_model2d + kres2d))[indx1,:],yp2d[indx1,:],'b',label="x_1")
plt.plot((k_model2d/(k_model2d + kres2d))[indx2,:],yp2d[indx2,:],'r',label="x_2")
plt.plot((k_model2d/(k_model2d + kres2d))[indx3,:],yp2d[indx3,:],'g',label="x_3")


plt.ylabel("$y$")
plt.legend()
plt.title("$|\\langle k_{mod} \\rangle/(\\langle k_{mod}\\rangle + k_{res}$")
plt.savefig('Assignment 2/Assignment 2b/k_ratio_node_1_3.png')

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot((k_model2d/(k_model2d + kres2d))[indx4,:],yp2d[indx4,:],'b',label="x_4")
plt.plot((k_model2d/(k_model2d + kres2d))[indx5,:],yp2d[indx5,:],'r',label="x_5")
plt.plot((k_model2d/(k_model2d + kres2d))[indx6,:],yp2d[indx6,:],'g',label="x_6")
plt.plot((k_model2d/(k_model2d + kres2d))[indx7,:],yp2d[indx7,:],'k',label="x_7")


plt.ylabel("$y$")
plt.legend()
plt.title("$\\langle k_{mod} \\rangle/(\\langle k_{mod}\\rangle + k_{res}$")
plt.savefig('Assignment 2/Assignment 2b/k_ratio_node_4_7.png')


#Ratio of boundary layer thickness
interface = np.zeros(ni)
boundary_layer_smoothed = []
boundary_layer = []
smoothed_index = []
for i in range(ni):
    interface[i] = np.abs(((vis2d-viscos)/viscos)[i,:]- np.ones(nj)).argmin()
    boundary_layer.append(yp2d[i,np.abs(((vis2d-viscos)/viscos)[i,:]- np.ones(nj)).argmin()])
    if yp2d[i,np.abs(((vis2d-viscos)/viscos)[i,:]- np.ones(nj)).argmin()] > 0.07:
        boundary_layer_smoothed.append(yp2d[i,np.abs(((vis2d-viscos)/viscos)[i,:]- np.ones(nj)).argmin()])
        smoothed_index.append(i)

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(np.linspace(np.min(xp2d[:,0]),np.max(xp2d[:,-1]),len(boundary_layer_smoothed)),boundary_layer_smoothed)


plt.ylim([np.min(yp2d[-1,:]),np.max(yp2d[-1,:])])
plt.ylabel("$y$")
plt.title("Boundary layer approximation")
plt.savefig('Assignment 2/Assignment 2b/boundary_layer.png')

tauw=viscos*np.mean(u2d,axis = 0)[1]/y[1]
ustar=tauw**0.5
y_plus = np.zeros((ni,nj))
dx = np.zeros(ni)
limit_x = np.zeros(ni+1)
limit_z = np.zeros(ni+1)

for i in range(ni):
    y_plus[i,:] = (yp2d[i,:] - yp2d[i,0])*ustar/viscos

    if i == 0:
        dx[i] = 2*(xp2d[i+1,0]-xp2d[i,0])
    elif i == ni-1:
        dx[i] = 2*(xp2d[i,0] - xp2d[i-1,0])
    else:
        dx[i] = (xp2d[i+1,0]-xp2d[i,0])/2 + (xp2d[i,0]-xp2d[i-1,0])/2 


    limit_x[i] = y2d[i,0] + 10
    limit_z[i] = y2d[i,0] + 20

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(xp2d[:,0],boundary_layer/dx,'b',label="$\\delta / \\Delta x$")
#plt.plot(y_plus)
plt.plot(x2d[:,0],limit_x,'k',label="10")

#plt.ylim([np.min(yp2d[-1,:]),np.max(yp2d[-1,:])])
plt.xlabel("$x$")
#plt.legend()
plt.title("Boundary layer ratios")
plt.savefig('Assignment 2/Assignment 2b/boundary_layer_ratios_x.png')


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(xp2d[:,0],boundary_layer/dz, 'r', label="$\\delta /\\Delta z$")
#plt.plot(y_plus)

plt.plot(x2d[:,0],limit_z,'k-.',label="20")
#plt.ylim([np.min(yp2d[-1,:]),np.max(yp2d[-1,:])])
plt.xlabel("$x$")
#plt.legend()
plt.title("Boundary layer ratios")
plt.savefig('Assignment 2/Assignment 2b/boundary_layer_ratios_z.png')

#Shear stress Energy Comp
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot((k_model2d/(k_model2d + kres2d))[indx4,:],yp2d[indx4,:],'b',label="$k_{ratio}$")
plt.plot(np.abs((uv2d/(uv2d + uv_mod)))[indx4,:],yp2d[indx4,:],'r',label="$tau_{ratio}$")
plt.axvline(x = 0.2)


plt.ylabel("$y$")
plt.legend()
plt.title("Comparison of ratios")
plt.savefig('Assignment 2/Assignment 2b/k_tau_comp_node_4.png')