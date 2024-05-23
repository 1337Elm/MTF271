import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from dphidx_dy import dphidx_dy
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from grad_xyz import *
plt.rcParams.update({'font.size': 22})
plt.interactive(True)
name_dat = "Assignment 2/Assignment 2b/data/"
re =9.36e+5
viscos =1/re

xy_hump_fine = np.loadtxt(name_dat + "xy_hump.dat")
x=xy_hump_fine[:,0]
y=xy_hump_fine[:,1]

ni=314
nj=122

nim1=ni-1
njm1=nj-1

# read data file
vectz=np.genfromtxt(name_dat + "vectz_aiaa_paper.dat",comments="%")
ntstep=vectz[0]
n=len(vectz)

#            write(48,*)uvec(i,j)
#            write(48,*)vvec(i,j)
#            write(48,*)dummy(i,j)
#            write(48,*)uvec2(i,j)
#            write(48,*)vvec2(i,j)
#            write(48,*)wvec2(i,j)
#            write(48,*)uvvec(i,j)
#            write(48,*)p2D(i,j)
#            write(48,*)rk2D(i,j)
#            write(48,*)vis2D(i,j)  
#            write(48,*)dissp2D(i,j)
#            write(48,*)uvturb(i,j)



nn=12
nst=0
iu=range(nst+1,n,nn)
iv=range(nst+2,n,nn)
ifk=range(nst+3,n,nn)
iuu=range(nst+4,n,nn)
ivv=range(nst+5,n,nn)
iww=range(nst+6,n,nn)
iuv=range(nst+7,n,nn)
ip=range(nst+8,n,nn)
ik=range(nst+9,n,nn)
ivis=range(nst+10,n,nn)
idiss=range(nst+11,n,nn)
iuv_model=range(nst+12,n,nn)

u=vectz[iu]/ntstep
v=vectz[iv]/ntstep
fk=vectz[ifk]/ntstep
uu=vectz[iuu]/ntstep
vv=vectz[ivv]/ntstep
ww=vectz[iww]/ntstep
uv=vectz[iuv]/ntstep
p=vectz[ip]/ntstep
k_model=vectz[ik]/ntstep
vis=vectz[ivis]/ntstep
diss=vectz[idiss]/ntstep
uv_model=vectz[iuv_model]/ntstep

# uu is total inst. velocity squared. Hence the resolved turbulent resolved stresses are obtained as
uu=uu-u**2
vv=vv-v**2
uv=uv-u*v

p_2d=np.reshape(p,(ni,nj))
u_2d=np.reshape(u,(ni,nj))
v_2d=np.reshape(v,(ni,nj))
fk_2d=np.reshape(fk,(ni,nj))
uu_2d=np.reshape(uu,(ni,nj))
uv_2d=np.reshape(uv,(ni,nj))
vv_2d=np.reshape(vv,(ni,nj))
ww_2d=np.reshape(ww,(ni,nj))
k_model_2d=np.reshape(k_model,(ni,nj))
vis_2d=np.reshape(vis,(ni,nj)) #this is to total viscosity, i.e. vis_tot=vis+vis_turb
diss_2d=np.reshape(diss,(ni,nj)) 
x_2d=np.transpose(np.reshape(x,(nj,ni)))
y_2d=np.transpose(np.reshape(y,(nj,ni)))

# set fk_2d=1 at upper boundary
fk_2d[:,nj-1]=fk_2d[:,nj-2]

x065_off=np.genfromtxt(name_dat + "x065_off.dat",comments="%")

# the funtion dphidx_dy wants x and y arrays to be one cell smaller than u2d. Hence I take away the last row and column below
x_2d_new=np.delete(x_2d,-1,0)
x_2d_new=np.delete(x_2d_new,-1,1)
y_2d_new=np.delete(y_2d,-1,0)
y_2d_new=np.delete(y_2d_new,-1,1)
# compute the gradient
dudx,dudy=dphidx_dy(x_2d_new,y_2d_new,u_2d)


#*************************
# plot u
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
xx=0.65;
i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(u_2d[i1,:],y_2d[i1,:],'b-')
plt.plot(x065_off[:,2],x065_off[:,1],'bo')
plt.xlabel("$U$")
plt.ylabel("$y-y_{wall}$")
plt.title("$x=0.65$")
plt.axis([0, 1.3,0,0.3])
plt.savefig('Assignment 2/Assignment 2b/u065_hump_python_eps.png')

#*************************
# plot vv
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
xx=0.65;
i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(vv_2d[i1,:],y_2d[i1,:],'b-')
plt.plot(x065_off[:,5],x065_off[:,1],'bo')
plt.xlabel("$\overline{v'v'}$")
plt.ylabel("$y-y_{wall}$")
plt.title("$x=0.65$")
plt.axis([0, 0.01,0,0.3])
plt.savefig('Assignment 2/Assignment 2b/vv065_hump_python_eps.png')

#*************************
# plot uu
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
xx=0.65;
i1 = (np.abs(xx-x_2d[:,1])).argmin()  # find index which closest fits xx
plt.plot(uu_2d[i1,:],y_2d[i1,:],'b-')
plt.plot(x065_off[:,4],x065_off[:,1],'bo')
plt.xlabel("$\overline{u'u'}$")
plt.ylabel("$y-y_{wall}$")
plt.title("$x=0.65$")
plt.axis([0, 0.05,0,0.3])
plt.savefig('Assignment 2/Assignment 2b/uu065_hump_python_eps.png')

################################ contour plot
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.contourf(x_2d,y_2d,uu_2d, 50)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.clim(0,0.05)
plt.axis([0.6,1.5,0,1])
plt.title("contour $\overline{u'u'}$")
plt.savefig('Assignment 2/Assignment 2b/piso_python_eps.png')

################################ vector plot
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
k=6# plot every forth vector
ss=3.2 #vector length
plt.quiver(x_2d[::k,::k],y_2d[::k,::k],u_2d[::k,::k],v_2d[::k,::k],width=0.01)
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.axis([0.6,1.5,0,1])
plt.title("vector plot")
plt.savefig('Assignment 2/Assignment 2b/vect_python_eps.png')



#T.1 Discretization themes
lines = [0.65, 0.8, 0.9, 1, 1.10, 1.20, 1.30]
indx1 = np.abs(x_2d[:,0]- np.repeat(lines[0],len(x_2d[:,0]))).argmin()
indx2 = np.abs(x_2d[:,0]- np.repeat(lines[1],len(x_2d[:,0]))).argmin()
indx3 = np.abs(x_2d[:,0]- np.repeat(lines[2],len(x_2d[:,0]))).argmin()
indx4 = np.abs(x_2d[:,0]- np.repeat(lines[3],len(x_2d[:,0]))).argmin()
indx5 = np.abs(x_2d[:,0]- np.repeat(lines[4],len(x_2d[:,0]))).argmin()
indx6 = np.abs(x_2d[:,0]- np.repeat(lines[5],len(x_2d[:,0]))).argmin()
indx7 = np.abs(x_2d[:,0]- np.repeat(lines[6],len(x_2d[:,0]))).argmin()



fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(uv_2d[indx1,:],y_2d[indx1,:],'b',label="x_1")
plt.plot(uv_2d[indx2,:],y_2d[indx2,:],'r',label="x_2")
plt.plot(uv_2d[indx3,:],y_2d[indx3,:],'k',label="x_3")

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.title("")
plt.savefig('Assignment 2/Assignment 2b/uv_nodes1-3.png')



fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(uv_2d[indx4,:],y_2d[indx4,:],'g',label="x_4")
plt.plot(uv_2d[indx5,:],y_2d[indx5,:],'b',label="x_5")
plt.plot(uv_2d[indx6,:],y_2d[indx6,:],'r',label="x_6")
plt.plot(uv_2d[indx7,:],y_2d[indx7,:],'k',label="x_7")

plt.xlabel("$x$")
plt.ylabel("$y$")
plt.legend()
plt.title("")
plt.savefig('Assignment 2/Assignment 2b/uv_nodes4-7.png')


#T.2 The modeled turbulent shear stress

v_t_v = (vis_2d-viscos)/viscos

#Closest index above 1, rather plot more than boundary layer than not all of boundary layer
boundary_thickness_indx1 = np.abs(v_t_v[indx1,:]- np.repeat(1,len(v_t_v[indx1,:]))).argmin() + 1
boundary_thickness_indx2 = np.abs(v_t_v[indx2,:]- np.repeat(1,len(v_t_v[indx2,:]))).argmin()
boundary_thickness_indx3 = np.abs(v_t_v[indx3,:]- np.repeat(1,len(v_t_v[indx3,:]))).argmin() - 1
boundary_thickness_indx4 = np.abs(v_t_v[indx4,:]- np.repeat(1,len(v_t_v[indx4,:]))).argmin() 
boundary_thickness_indx5 = np.abs(v_t_v[indx5,:]- np.repeat(1,len(v_t_v[indx5,:]))).argmin() - 1
boundary_thickness_indx6 = np.abs(v_t_v[indx6,:]- np.repeat(1,len(v_t_v[indx6,:]))).argmin() + 1
boundary_thickness_indx7 = np.abs(v_t_v[indx7,:]- np.repeat(1,len(v_t_v[indx7,:]))).argmin() - 1

"""
print(v_t_v[indx1,boundary_thickness_indx1])
print(v_t_v[indx2,boundary_thickness_indx2])
print(v_t_v[indx3,boundary_thickness_indx3])
print(v_t_v[indx4,boundary_thickness_indx4])
print(v_t_v[indx5,boundary_thickness_indx5])
print(v_t_v[indx6,boundary_thickness_indx6])
print(v_t_v[indx7,boundary_thickness_indx7])
"""

uv_model_2d = np.reshape(uv_model,(ni,nj))


#1
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(uv_2d[indx1,:boundary_thickness_indx1],y_2d[indx1,:boundary_thickness_indx1],'b',label="$\overline{u'v'}_{Res}$")
plt.plot(uv_model_2d[indx1,:boundary_thickness_indx1],y_2d[indx1,:boundary_thickness_indx1],'r',label="$\overline{u'v'}_{Mod}$")

plt.xlabel("$\overline{u'v'}$")
plt.ylabel("$y$")
plt.legend()
plt.title("Shear stress, x = " + str(lines[0]))
plt.savefig('Assignment 2/Assignment 2b/shear_stress_comparoison_node1.png')

#2
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(uv_2d[indx2,:boundary_thickness_indx2],y_2d[indx2,:boundary_thickness_indx2],'b',label="$\overline{u'v'}_{Res}$")
plt.plot(uv_model_2d[indx2,:boundary_thickness_indx2],y_2d[indx2,:boundary_thickness_indx2],'r',label="$\overline{u'v'}_{Mod}$")


plt.xlabel("$\overline{u'v'}$")
plt.ylabel("$y$")
plt.title("Shear stress, x = " + str(lines[1]))
plt.legend()
plt.savefig('Assignment 2/Assignment 2b/shear_stress_comparoison_node2.png')


#3
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(uv_2d[indx3,:boundary_thickness_indx3],y_2d[indx3,:boundary_thickness_indx3],'b',label="$\overline{u'v'}_{Res}$")
plt.plot(uv_model_2d[indx3,:boundary_thickness_indx3],y_2d[indx3,:boundary_thickness_indx3],'r',label="$\overline{u'v'}_{Mod}$")


plt.xlabel("$\overline{u'v'}$")
plt.ylabel("$y$")
plt.title("Shear stress, x = " + str(lines[2]))
plt.legend()
plt.savefig('Assignment 2/Assignment 2b/shear_stress_comparoison_node3.png')


#4
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(uv_2d[indx4,:boundary_thickness_indx4],y_2d[indx4,:boundary_thickness_indx4],'b',label="$\overline{u'v'}_{Res}$")
plt.plot(uv_model_2d[indx4,:boundary_thickness_indx4],y_2d[indx4,:boundary_thickness_indx4],'r',label="$\overline{u'v'}_{Mod}$")


plt.xlabel("$\overline{u'v'}$")
plt.ylabel("$y$")
plt.title("Shear stress, x = " + str(lines[3]))
plt.legend()
plt.savefig('Assignment 2/Assignment 2b/shear_stress_comparoison_node4.png')


#5
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(uv_2d[indx5,:boundary_thickness_indx5],y_2d[indx5,:boundary_thickness_indx5],'b',label="$\overline{u'v'}_{Res}$")
plt.plot(uv_model_2d[indx5,:boundary_thickness_indx5],y_2d[indx5,:boundary_thickness_indx5],'r',label="$\overline{u'v'}_{Mod}$")


plt.xlabel("$\overline{u'v'}$")
plt.ylabel("$y$")
plt.title("Shear stress, x = " + str(lines[4]))
plt.legend()
plt.savefig('Assignment 2/Assignment 2b/shear_stress_comparoison_node5.png')


#6
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(uv_2d[indx6,:boundary_thickness_indx6],y_2d[indx6,:boundary_thickness_indx6],'b',label="$\overline{u'v'}_{Res}$")
plt.plot(uv_model_2d[indx6,:boundary_thickness_indx6],y_2d[indx6,:boundary_thickness_indx6],'r',label="$\overline{u'v'}_{Mod}$")


plt.xlabel("$\overline{u'v'}$")
plt.ylabel("$y$")
plt.title("Shear stress, x = " + str(lines[5]))
plt.legend()
plt.savefig('Assignment 2/Assignment 2b/shear_stress_comparoison_node6.png')


#7
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(uv_2d[indx7,:boundary_thickness_indx7],y_2d[indx7,:boundary_thickness_indx7],'b',label="$\overline{u'v'}_{Res}$")
plt.plot(uv_model_2d[indx7,:boundary_thickness_indx7],y_2d[indx7,:boundary_thickness_indx7],'r',label="$\overline{u'v'}_{Mod}$")


plt.xlabel("$\overline{u'v'}$")
plt.ylabel("$y$")
plt.title("Shear stress, x = " + str(lines[6]))
plt.legend()
plt.savefig('Assignment 2/Assignment 2b/shear_stress_comparoison_node7.png')



#T.3 The turbulent viscosity


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(v_t_v[indx4,:boundary_thickness_indx4],y_2d[indx4,:boundary_thickness_indx4],'b',label="$\\nu_t$")


plt.xlabel("$\\nu_t$")
plt.ylabel("$y$")
plt.title("Turbulent viscosity, x = " + str(lines[3]))
plt.legend()
plt.savefig('Assignment 2/Assignment 2b/viscos_turb_node4.png')

duudx,duudy=dphidx_dy(x_2d_new,y_2d_new,uu_2d)
dvvdx,dvvdy=dphidx_dy(x_2d_new,y_2d_new,vv_2d)
duvdx,duvdy=dphidx_dy(x_2d_new,y_2d_new,uv_2d)


dvdx,dvdy = dphidx_dy(x_2d_new,y_2d_new,v_2d)

term_1_1 = dphidx_dy(x_2d_new,y_2d_new,v_t_v*dudx)[0]
term_1_2 = dphidx_dy(x_2d_new,y_2d_new,v_t_v*dudy)[1]
term_1_3 = dphidx_dy(x_2d_new,y_2d_new,v_t_v*dvdx)[0]
term_1_4 = dphidx_dy(x_2d_new,y_2d_new,v_t_v*dvdy)[1]
term_1 = term_1_1 + term_1_2 + term_1_3 + term_1_4

term_2 = -duudx - duvdy - duvdx - dvvdy

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(term_1[indx4,:boundary_thickness_indx4],y_2d[indx4,:boundary_thickness_indx4],'b',label="$\partial / \partial x_j (\\langle \\nu_t \partial \overline{v}_i/\partial x_j\\rangle)$")

plt.xlabel("")
plt.ylabel("$y$")
plt.title("Modeled, x = " + str(lines[3]))
plt.legend()
plt.savefig('Assignment 2/Assignment 2b/term_1_node4.png')

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(term_2[indx4,:boundary_thickness_indx4],y_2d[indx4,:boundary_thickness_indx4],'b',label="$-\partial \\langle v_i'v_j' \\rangle/\partial x_j$")

plt.xlabel("")
plt.ylabel("$y$")
plt.title("Resolved, x = " + str(lines[3]))
plt.legend()
plt.savefig('Assignment 2/Assignment 2b/term_2_node4.png')


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(term_1[indx4,:boundary_thickness_indx4],y_2d[indx4,:boundary_thickness_indx4],'b',label="$\partial / \partial x_j (\\langle \\nu_t \partial \overline{v}_i/\partial x_j\\rangle)$")
plt.plot(term_2[indx4,:boundary_thickness_indx4],y_2d[indx4,:boundary_thickness_indx4],'r',label="$-\partial \\langle v_i'v_j' \\rangle/\partial x_j$")

plt.xlabel("")
plt.ylabel("$y$")
plt.title("Comparison, x = " + str(lines[3]))
plt.legend()
plt.savefig('Assignment 2/Assignment 2b/term_12_node4.png')


# read data file
vectz=np.genfromtxt(name_dat + "vectz_aiaa_journal.dat",comments="%")
ntstep=vectz[0]
n=len(vectz)

#            write(48,*)uvec(i,j)
#            write(48,*)vvec(i,j)
#            write(48,*)dummy(i,j)
#            write(48,*)uvec2(i,j)
#            write(48,*)vvec2(i,j)
#            write(48,*)wvec2(i,j)
#            write(48,*)uvvec(i,j)
#            write(48,*)p2D(i,j)
#            write(48,*)rk2D(i,j)
#            write(48,*)vis2D(i,j)  
#            write(48,*)dissp2D(i,j)
#            write(48,*)uvturb(i,j)

nn=12
nst=0
ivis=range(nst+10,n,nn)

vis_journal=vectz[ivis]/ntstep
vis_2d_journal=np.reshape(vis_journal,(ni,nj)) #this is to total viscosity, i.e. vis_tot=vis+vis_turb

v_t_v_journal = (vis_2d_journal-viscos)/viscos

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(v_t_v[indx4,:boundary_thickness_indx4],y_2d[indx4,:boundary_thickness_indx4],'b',label="$\\nu_{t-214}$")
plt.plot(v_t_v_journal[indx4,:boundary_thickness_indx4],y_2d[indx4,:boundary_thickness_indx4],'r',label="$\\nu_{t-179}$")

plt.xlabel("")
plt.ylabel("$y$")
plt.title("Comparison, x = " + str(lines[3]))
plt.legend()
plt.savefig('Assignment 2/Assignment 2b/turb_viscosity_comparison.png')