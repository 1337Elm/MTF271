import scipy.io as sio
import sys
import numpy as np
import matplotlib.pyplot as plt
from gradients import compute_face_phi,dphidx,dphidy,init
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

plt.rcParams.update({'font.size': 22})

plt.interactive(True)
plt.close('all')

viscos=1/550

name = './Contraction/'

datax= np.loadtxt(str(name)+"x2d.dat")
x=datax[0:-1]
ni=int(datax[-1])
datay= np.loadtxt(str(name)+"y2d.dat")
y=datay[0:-1]
nj=int(datay[-1])

x2d=np.zeros((ni+1,nj+1))
y2d=np.zeros((ni+1,nj+1))

x2d=np.reshape(x,(ni+1,nj+1))
y2d=np.reshape(y,(ni+1,nj+1))

# compute cell centers
xp2d=0.25*(x2d[0:-1,0:-1]+x2d[0:-1,1:]+x2d[1:,0:-1]+x2d[1:,1:])
yp2d=0.25*(y2d[0:-1,0:-1]+y2d[0:-1,1:]+y2d[1:,0:-1]+y2d[1:,1:])

x=xp2d[:,0]
y=yp2d[0,:]

ywall_s=0.5*(y2d[0:-1,0]+y2d[1:,0])
dist_s=yp2d-ywall_s[:,None]


#z grid
zmax, nk=np.loadtxt(str(name)+'z.dat')
nk=int(nk)
zp = np.linspace(0, zmax, nk)

itstep,nk,dz=np.load(str(name)+'itstep.npy')
p2d=np.load(str(name)+'p_averaged.npy')/itstep
u2d=np.load(str(name)+'u_averaged.npy')/itstep
v2d=np.load(str(name)+'v_averaged.npy')/itstep
w2d=np.load(str(name)+'w_averaged.npy')/itstep
uu2d=np.load(str(name)+'uu_stress.npy')/itstep
vv2d=np.load(str(name)+'vv_stress.npy')/itstep
ww2d=np.load(str(name)+'ww_stress.npy')/itstep
uv2d=np.load(str(name)+'uv_stress.npy')/itstep

uu2d=uu2d-u2d**2
vv2d=vv2d-v2d**2
uv2d=uv2d-u2d*v2d

diss2d=np.load(str(name)+'diss_mean.npy')

ubulk = np.trapz(u2d[0,:],yp2d[0,:])/max(y2d[0,:])
print('ubulk',ubulk)
ustar2=viscos*u2d[:,0]/dist_s[:,0]
yplus2d=np.ones((ni,nj))
for i in range(0,ni):
   yplus2d[i,:]=(abs(ustar2[i]))**0.5*yp2d[i,:]/viscos
cf=(abs(ustar2))**0.5*np.sign(ustar2)/ubulk**2/0.5
ustar=(abs(ustar2))**0.5


kres_2d=0.5*(uu2d+vv2d+ww2d)


#
# compute re_delta1 for boundary layer flow
dx=x[3]-x[2]
re_disp_bl=np.zeros(ni)
delta_disp=np.zeros(ni)
for i in range (0,ni-1):
   d_disp=0
   for j in range (1,nj-1):
      up=u2d[i,j]/u2d[i,-1]
      dy=y2d[i,j]-y2d[i,j-1]
      d_disp=d_disp+(1.-min(up,1.))*dy

   delta_disp[i]=d_disp
   re_disp_bl[i]=d_disp*u2d[i,-1]/viscos

re_disp_bl[-1]=re_disp_bl[-1-1]
delta_disp[-1]=delta_disp[-1-1]

# Load DNS channel at Re_tau = 550
# ----------------------------------------------------------------------------------------------------------------------------------------
DNS_mean=np.genfromtxt(name + "Re550.dat",comments="%")
y_DNS=DNS_mean[:,0]
yplus_DNS=DNS_mean[:,1]
u_DNS=DNS_mean[:,2]
u2_DNS=DNS_mean[:,3]**2
v2_DNS=DNS_mean[:,4]**2
w2_DNS=DNS_mean[:,5]**2
uv_DNS=DNS_mean[:,10]

k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)

dudy_DNS = np.gradient(u_DNS,y_DNS)

vist_DNS = abs(uv_DNS)/dudy_DNS

#%       y/h               y+           dissip          produc         p-strain          p-diff          t-diff         v-diff            bal          tp-kbal

DNS_k_bal=np.genfromtxt(name + "Re550_bal_kbal.dat",comments="%")
diss_DNS=-DNS_k_bal[:,2]/viscos # it is scaled with ustar**4/viscos
prod_DNS=DNS_k_bal[:,3]/viscos # it is scaled with ustar**4/viscos




# compute geometric quantities
areaw,areawx,areawy,areas,areasx,areasy,vol,fx,fy,as_bound = init(x2d,y2d,xp2d,yp2d)

# compute face value of U and V
zero_bc=np.zeros(ni)
u2d_face_w,u2d_face_s=compute_face_phi(u2d,fx,fy,ni,nj,zero_bc)
v2d_face_w,v2d_face_s=compute_face_phi(v2d,fx,fy,ni,nj,zero_bc)
p2d_face_w,p2d_face_s=compute_face_phi(p2d,fx,fy,ni,nj,p2d[:,0])

# x derivatives
dudx=dphidx(u2d_face_w,u2d_face_s,areawx,areasx,vol)
dvdx=dphidx(v2d_face_w,v2d_face_s,areawx,areasx,vol)
dpdx=dphidx(p2d_face_w,p2d_face_s,areawx,areasx,vol)

# y derivatives
dudy=dphidy(u2d_face_w,u2d_face_s,areawy,areasy,vol)
dvdy=dphidy(v2d_face_w,v2d_face_s,areawy,areasy,vol)
dpdy=dphidy(p2d_face_w,p2d_face_s,areawy,areasy,vol)

#%%%%%%%%%%%%%%%%%%%%% grid
fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
for i in range(0,len(x2d[:,0])+1,5):
   plt.plot(x2d[i,:],y2d[i,:])
plt.plot(x2d[-1,:],y2d[-1,:])

for j in range(0,len(y2d[0,:])+1,5):
   plt.plot(x2d[:,j],y2d[:,j])
plt.plot(x2d[:,-1],y2d[:,-1])
plt.axis([-0.5,5,0,1])
plt.axis('equal')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('grid-hill.png',bbox_inches='tight')

##########################################  P
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.26,bottom=0.20)
cp=p2d[:,0]/ubulk**2/0.5
plt.plot(xp2d[:,1],cp,'b-',label='hill')

plt.ylabel("$c_P$")
plt.axis([-0.5,5,1.1*np.min(cp),0.9*np.max(cp)])
plt.savefig('cp-hill-wave.png',bbox_inches='tight')


##########################################  cf
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.26,bottom=0.20)
plt.plot(xp2d[:,1],cf,'b-',label='hill')
plt.ylabel("$C_f$")
plt.xlabel("$x$")
plt.axis([-0.5,5,1.1*np.min(cf),1.1*np.max(cf)])
plt.legend(loc='best',fontsize=16)
plt.savefig('cf-hill-wave.png',bbox_inches='tight')


########################################## diss at inlet
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i=0
plt.plot(yplus2d[i,:],diss2d[i,:],'b-',label='pyCALC')
plt.plot(yplus_DNS,diss_DNS,'r-',label='DNS')
plt.ylabel(r"$\varepsilon$")
plt.xlabel("$y^+$")
plt.legend(loc='best',fontsize=14)
plt.axis([0, 500, 0, 50])
plt.savefig('diss.png',bbox_inches='tight')


########################################## diss-zoom at inlet
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
i=0
plt.plot(yplus2d[i,:],diss2d[i,:],'b-',label='pyCALC')
plt.plot(yplus_DNS,diss_DNS,'r-',label='DNS')
plt.ylabel(r"$\varepsilon$")
plt.xlabel("$y^+$")
plt.legend(loc='best',fontsize=14)
plt.axis([0, 50, 0, 150])
plt.savefig('diss-zoom.png',bbox_inches='tight')


#%%%%%%%%%%%%%%%%%%%%% grid
fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
for i in range(0,len(x2d[:,0])+1,5):
   plt.plot(x2d[i,:],y2d[i,:])
plt.plot(x2d[-1,:],y2d[-1,:])

for j in range(0,len(y2d[0,:])+1,5):
   plt.plot(x2d[:,j],y2d[:,j])
plt.plot(x2d[:,-1],y2d[:,-1])
plt.axis([-0.5,5,0,1])
plt.axis('equal')
plt.gca().set_aspect('equal', adjustable='box')
plt.savefig('grid.png',bbox_inches='tight')



#%%%%%%%%%
#Plot reynolds stresses
grid_line_1 = 3
grid_line_2 = -10

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

y2d = y2d[:,1:]
plt.plot(uu2d[grid_line_1,:],yplus2d[grid_line_1,:],label = "$\overline{u'u'}$")
plt.plot(vv2d[grid_line_1,:],yplus2d[grid_line_1,:],label = "$\overline{v'v'}$")
plt.plot(ww2d[grid_line_1,:],yplus2d[grid_line_1,:],label = "$\overline{w'w'}$")

plt.plot(uv2d[grid_line_1,:],yplus2d[grid_line_1,:],label = "$\overline{u'v'}$")

plt.axis([-1,8,0,500])
plt.legend()
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title(f"Reynolds Stresses along x = {np.round(x2d[grid_line_1,5],3)}")
plt.savefig('stresses1.png',bbox_inches = 'tight')

fig59,ax1 = plt.subplots()
plt.plot(uu2d[grid_line_2,:],yplus2d[grid_line_2,:],label = "$\overline{u'u'}$")
plt.plot(vv2d[grid_line_2,:],yplus2d[grid_line_2,:],label = "$\overline{v'v'}$")
plt.plot(ww2d[grid_line_2,:],yplus2d[grid_line_2,:],label = "$\overline{w'w'}$")

plt.plot(uv2d[grid_line_2,:],yplus2d[grid_line_2,:],label = "$\overline{u'v'}$")

plt.axis([-3,15,225,500])
plt.legend()
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title(f"Reynolds Stresses along x = {np.round(x2d[grid_line_2,5],3)}")
plt.savefig('stresses2.png',bbox_inches = 'tight')


#%%%%%%
#Plot all terms in v_1 equation
fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

#Derivatives
uu2d_face_w,uu2d_face_s = compute_face_phi(uu2d,fx,fy,ni,nj,zero_bc)
uv2d_face_w,uv2d_face_s = compute_face_phi(uv2d,fx,fy,ni,nj,zero_bc)

duu2ddx = dphidx(uu2d_face_w,uu2d_face_s,areawy,areasy,vol)
duv2ddx = dphidx(uv2d_face_w,uv2d_face_s,areawy,areasy,vol)

#Second Derivatives
dudx_face_w,dudx_face_s = compute_face_phi(dudx,fx,fy,ni,nj,zero_bc)
dudy_face_w,dudy_face_s = compute_face_phi(dudy,fx,fy,ni,nj,zero_bc)

d2udx2 = dphidx(dudx_face_w,dudx_face_s,areawy,areasy,vol)
d2udy2 = dphidy(dudy_face_w,dudy_face_s,areawy,areasy,vol)

plt.plot(u2d[grid_line_1,:]*dudx[grid_line_1,:],yplus2d[grid_line_1,:],label = "$\overline{v}_1 \partial \overline{v}_1 / \partial x_1$")
plt.plot(v2d[grid_line_1,:]*dudy[grid_line_1,:],yplus2d[grid_line_1,:],label = "$\overline{v}_2 \partial \overline{v}_1 / \partial x_2$")
plt.plot(-dpdx[grid_line_1,:],yplus2d[grid_line_1,:],label = "$\partial \overline{p}/\partial x_1$")
plt.plot(viscos* d2udx2[grid_line_1,:],yplus2d[grid_line_1,:],label = "$\\nu \partial^2 \overline{v}_1/\partial x_1^2$")


plt.axis([-10,10,0,500])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title("Terms in " + "$\overline{v}_1$-equation" + f" along x = {np.round(x2d[grid_line_1,5],3)}")
plt.savefig('termsV_1-1.png',bbox_inches = 'tight')

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(-duu2ddx[grid_line_1,:],yplus2d[grid_line_1,:],label = "$ -\partial \overline{v'^2}_1/\partial x_1$")
plt.plot(-duv2ddx[grid_line_1,:],yplus2d[grid_line_1,:],label = "$ -\partial \overline{v'_1v'_2}/\partial x_1$")
plt.axis([-550,100,0,500])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title("Terms in " + "$\overline{v}_1$-equation" + f" along x = {np.round(x2d[grid_line_1,5],3)}")
plt.savefig('termsV_1-1_large_terms.png',bbox_inches = 'tight')

fig59,ax1 = plt.subplots()
plt.plot(u2d[grid_line_2,:]*dudx[grid_line_2,:],yplus2d[grid_line_2,:],label = "$\overline{v}_1 \partial \overline{v}_1 / \partial x_1$")
plt.plot(v2d[grid_line_2,:]*dudy[grid_line_2,:],yplus2d[grid_line_2,:],label = "$\overline{v}_2 \partial \overline{v}_1 / \partial x_2$")
plt.plot(-dpdx[grid_line_2,:],yplus2d[grid_line_2,:],label = "$\partial \overline{p}/\partial x_1$")
plt.plot(viscos* d2udx2[grid_line_2,:],yplus2d[grid_line_2,:],label = "$\\nu \partial^2 \overline{v}_1/\partial x_1^2$")

plt.axis([-10,10,225,500])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title("Terms in " + "$\overline{v}_1$-equation" + f" along x = {np.round(x2d[grid_line_2,5],3)}")
plt.savefig('termsV_1-2.png',bbox_inches = 'tight')


fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(-duu2ddx[grid_line_2,:],yplus2d[grid_line_2,:],label = "$ -\partial \overline{v'^2}_1/\partial x_1$")
plt.plot(-duv2ddx[grid_line_2,:],yplus2d[grid_line_2,:],label = "$ -\partial \overline{v'_1v'_2}/\partial x_1$")
plt.axis([-1550,300,225,500])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title("Terms in " + "$\overline{v}_1$-equation" + f" along x = {np.round(x2d[grid_line_2,5],3)}")
plt.savefig('termsV_1-2_large_terms.png',bbox_inches = 'tight')

#%%%%
#Plot Production term

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus2d[grid_line_1,:],-uv2d[grid_line_1,:]*dudy[grid_line_1,:],label = "$P^k$" + f" x = {np.round(x2d[grid_line_1,2],3)}")
plt.plot(yplus2d[grid_line_2,:],-uv2d[grid_line_2,:]*dudy[grid_line_2,:],label = "$P^k$" + f" x = {np.round(x2d[grid_line_2,2],3)}")
plt.legend(fontsize = "10")
plt.xlabel("$y^+$")
plt.ylabel("Amplitude")
plt.title("Production")
plt.savefig('prod.png',bbox_inches = 'tight')

#Plot Production term and dissipation
fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus2d[grid_line_1,:],-uv2d[grid_line_1,:]*dudy[grid_line_1,:],label = "$P^k$" + f" x = {np.round(x2d[grid_line_1,2],3)}")
plt.plot(yplus2d[grid_line_1,:],diss2d[grid_line_1,:],label = "$\\varepsilon$" + f" x = {np.round(x2d[grid_line_1,2],3)}")

plt.plot(yplus2d[grid_line_2,:],-uv2d[grid_line_2,:]*dudy[grid_line_2,:],label = "$P^k" + f" x = {np.round(x2d[grid_line_2,2],3)}$")
plt.plot(yplus2d[grid_line_2,:],diss2d[grid_line_2,:],label = "$\\varepsilon$" + f" x = {np.round(x2d[grid_line_2,2],3)}")
plt.legend(fontsize = "10")
plt.xlabel("$y^+$")
plt.ylabel("Amplitude")
plt.title("Production")
plt.savefig('prod_diss.png',bbox_inches = 'tight')
