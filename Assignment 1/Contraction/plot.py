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
plt.savefig(name + 'grid-hill.png',bbox_inches='tight')

##########################################  P
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.26,bottom=0.20)
cp=p2d[:,0]/ubulk**2/0.5
plt.plot(xp2d[:,1],cp,'b-',label='hill')

plt.ylabel("$c_P$")
plt.axis([-0.5,5,1.1*np.min(cp),0.9*np.max(cp)])
plt.savefig(name + 'cp-hill-wave.png',bbox_inches='tight')


##########################################  cf
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.26,bottom=0.20)
plt.plot(xp2d[:,1],cf,'b-',label='hill')
plt.ylabel("$C_f$")
plt.xlabel("$x$")
plt.axis([-0.5,5,1.1*np.min(cf),1.1*np.max(cf)])
plt.legend(loc='best',fontsize=16)
plt.savefig(name + 'cf-hill-wave.png',bbox_inches='tight')


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
plt.savefig(name + 'diss.png',bbox_inches='tight')


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
plt.savefig(name + 'diss-zoom.png',bbox_inches='tight')


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
plt.savefig(name + 'grid.png',bbox_inches='tight')



#%%%%%%%%%
#1.1 Plot reynolds stresses
grid_line_1 = 3
grid_line_2 = -10

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(uu2d[grid_line_1,:],yplus2d[grid_line_1,:],'b',markersize = 2,label = "$\overline{u'u'}$")
plt.plot(vv2d[grid_line_1,:],yplus2d[grid_line_1,:],'r--',markersize = 2,label = "$\overline{v'v'}$")
plt.plot(ww2d[grid_line_1,:],yplus2d[grid_line_1,:],'k-.',markersize = 2,label = "$\overline{w'w'}$")

plt.plot(uv2d[grid_line_1,:],yplus2d[grid_line_1,:],'g:',fillstyle = 'none',markersize = 2,label = "$\overline{u'v'}$")

plt.axis([-1,8,0,500])
plt.legend()
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title(f"Reynolds Stresses along x = {np.round(x2d[grid_line_1,5],3)}")
plt.savefig(name + 'stresses1.png',bbox_inches = 'tight')

fig59,ax1 = plt.subplots()
plt.plot(uu2d[grid_line_2,:],yplus2d[grid_line_2,:],'b',markersize = 2,label = "$\overline{u'u'}$")
plt.plot(vv2d[grid_line_2,:],yplus2d[grid_line_2,:],'r--',markersize = 2,label = "$\overline{v'v'}$")
plt.plot(ww2d[grid_line_2,:],yplus2d[grid_line_2,:],'k-.',markersize = 2,label = "$\overline{w'w'}$")

plt.plot(uv2d[grid_line_2,:],yplus2d[grid_line_2,:],'g:',fillstyle = 'none',markersize = 2,label = "$\overline{u'v'}$")

plt.axis([-3,15,225,500])
plt.legend()
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title(f"Reynolds Stresses along x = {np.round(x2d[grid_line_2,5],3)}")
plt.savefig(name + 'stresses2.png',bbox_inches = 'tight')


#%%%%%%
#1.2 Plot all terms in v_1 equation
fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

#Derivatives
uu2d_face_w,uu2d_face_s = compute_face_phi(uu2d,fx,fy,ni,nj,zero_bc)
uv2d_face_w,uv2d_face_s = compute_face_phi(uv2d,fx,fy,ni,nj,zero_bc)

duu2ddx = dphidx(uu2d_face_w,uu2d_face_s,areawx,areasx,vol)
duu2ddy = dphidy(uu2d_face_w,uu2d_face_s,areawy,areasy,vol)
duv2ddx = dphidx(uv2d_face_w,uv2d_face_s,areawx,areasx,vol)
duv2ddy = dphidy(uv2d_face_w,uv2d_face_s,areawy,areasy,vol)

duu2ddx_face_w,duu2ddx_face_s = compute_face_phi(duu2ddx,fx,fy,ni,nj,zero_bc)

duu2ddxdy = dphidy(duu2ddx_face_w,duu2ddx_face_s,areawy,areasy,vol)
d2uu2ddx2 = dphidx(duu2ddx_face_w,duu2ddx_face_s,areawx,areasx,vol)
d2uu2ddy2 = dphidy(duu2ddx_face_w,duu2ddx_face_s,areawy,areasy,vol)
#Second Derivatives
dudx_face_w,dudx_face_s = compute_face_phi(dudx,fx,fy,ni,nj,zero_bc)
dudy_face_w,dudy_face_s = compute_face_phi(dudy,fx,fy,ni,nj,zero_bc)

d2udx2 = dphidx(dudx_face_w,dudx_face_s,areawx,areasx,vol)
d2udy2 = dphidy(dudy_face_w,dudy_face_s,areawy,areasy,vol)

plt.plot(u2d[grid_line_1,:]*dudx[grid_line_1,:],yplus2d[grid_line_1,:],'b',markersize = 2,label = "$\overline{v}_1 \partial \overline{v}_1 / \partial x_1$")
plt.plot(v2d[grid_line_1,:]*dudy[grid_line_1,:],yplus2d[grid_line_1,:],'r--',markersize = 2,label = "$\overline{v}_2 \partial \overline{v}_1 / \partial x_2$")
plt.plot(-dpdx[grid_line_1,:],yplus2d[grid_line_1,:],'k-.',markersize = 2,label = "$\partial \overline{p}/\partial x_1$")
plt.plot(viscos* d2udx2[grid_line_1,:],yplus2d[grid_line_1,:],'g:',markersize = 2,label = "$\\nu \partial^2 \overline{v}_1/\partial x_1^2$")
plt.plot(-duu2ddx[grid_line_1,:],yplus2d[grid_line_1,:],'bo',fillstyle = 'none',markersize = 2,label = "$ -\partial \overline{v'^2}_1/\partial x_1$")

plt.axis([-10,10,0,500])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title("Terms in " + "$\overline{v}_1$-equation" + f" along x = {np.round(x2d[grid_line_1,5],3)}")
plt.savefig(name + 'termsV_1-1.png',bbox_inches = 'tight')

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(-duv2ddy[grid_line_1,:],yplus2d[grid_line_1,:],'r',label = "$ -\partial \overline{v'_1v'_2}/\partial x_1$")
plt.plot(viscos* d2udy2[grid_line_1,:],yplus2d[grid_line_1,:],'g:',markersize = 2,label = "$\\nu \partial^2 \overline{v}_1/\partial x_2^2$")
#plt.axis([-550,100,0,500])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.xlim([-50,50])
plt.title("Terms in " + "$\overline{v}_1$-equation" + f" along x = {np.round(x2d[grid_line_1,5],3)}")
plt.savefig(name + 'termsV_1-1_large_terms.png',bbox_inches = 'tight')

fig59,ax1 = plt.subplots()
plt.plot(u2d[grid_line_2,:]*dudx[grid_line_2,:],yplus2d[grid_line_2,:],'b',markersize = 2,label = "$\overline{v}_1 \partial \overline{v}_1 / \partial x_1$")
plt.plot(v2d[grid_line_2,:]*dudy[grid_line_2,:],yplus2d[grid_line_2,:],'r--',markersize = 2,label = "$\overline{v}_2 \partial \overline{v}_1 / \partial x_2$")
plt.plot(-dpdx[grid_line_2,:],yplus2d[grid_line_2,:],'k-.',markersize = 2,label = "$\partial \overline{p}/\partial x_1$")
plt.plot(viscos* d2udx2[grid_line_2,:],yplus2d[grid_line_2,:],'g:',markersize = 2,label = "$\\nu \partial^2 \overline{v}_1/\partial x_1^2$")
plt.plot(-duu2ddx[grid_line_2,:],yplus2d[grid_line_2,:],'bo',fillstyle = 'none',markersize = 2,label = "$ -\partial \overline{v'^2}_1/\partial x_1$")

plt.axis([-10,10,225,500])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title("Terms in " + "$\overline{v}_1$-equation" + f" along x = {np.round(x2d[grid_line_2,5],3)}")
plt.savefig(name + 'termsV_1-2.png',bbox_inches = 'tight')


fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(-duv2ddy[grid_line_2,:],yplus2d[grid_line_2,:],'r',label = "$ -\partial \overline{v'_1v'_2}/\partial x_2$")
plt.plot(viscos* d2udy2[grid_line_2,:],yplus2d[grid_line_2,:],markersize = 2,label = "$\\nu \partial^2 \overline{v}_1/\partial x_2^2$")
#plt.axis([-1550,300,225,500])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.xlim([-120,120])
plt.title("Terms in " + "$\overline{v}_1$-equation" + f" along x = {np.round(x2d[grid_line_2,5],3)}")
plt.savefig(name + 'termsV_1-2_large_terms.png',bbox_inches = 'tight')

#%%%%
# 1.3 Plot Production term
P = -uu2d*dudx - vv2d*dvdy - uv2d*dudy - uv2d*dvdx

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus2d[grid_line_1,:],P[grid_line_1,:],'b',markersize = 2,label = "$P^k$" + f" x = {np.round(x2d[grid_line_1,2],3)}")
plt.legend(fontsize = "10")
plt.xlabel("$y^+$")
plt.ylabel("Amplitude")
plt.title("Production")
plt.savefig(name + 'prod_1.png',bbox_inches = 'tight')



fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus2d[grid_line_2,:],P[grid_line_2,:],'b',markersize = 2,label = "$P^k$" + f" x = {np.round(x2d[grid_line_2,2],3)}")
plt.legend(fontsize = "10")
plt.xlabel("$y^+$")
plt.ylabel("Amplitude")
plt.title("Production")
plt.savefig(name + 'prod_2.png',bbox_inches = 'tight')


fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus2d[grid_line_1,:],np.abs(uu2d[grid_line_1,:]),'b',markersize = 2,label = "$|\overline{u'^2}|$")
plt.plot(yplus2d[grid_line_1,:],np.abs(2*uv2d[grid_line_1,:]),'r--',markersize = 2,label = "$|2 \overline{u'v'}|$")
plt.plot(yplus2d[grid_line_1,:],np.abs(vv2d[grid_line_1,:]),'k-.',markersize = 2,label = "$|\overline{v'^2}|$")
plt.legend(fontsize = "10")
plt.xlabel("$y^+$")
plt.ylabel("Amplitude")
plt.title("Contribution")
plt.savefig(name + 'contributions_production_1.png',bbox_inches = 'tight')

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus2d[grid_line_2,:],np.abs(uu2d[grid_line_2,:]),'b',markersize = 2,label = "$|\overline{u'^2}|$")
plt.plot(yplus2d[grid_line_2,:],np.abs(2*uv2d[grid_line_2,:]),'r--',markersize = 2,label = "$|2 \overline{u'v'}|$")
plt.plot(yplus2d[grid_line_2,:],np.abs(vv2d[grid_line_2,:]),'k-.',markersize = 2,label = "$|\overline{v'^2}|$")
plt.legend(fontsize = "10")
plt.xlabel("$y^+$")
plt.ylabel("Amplitude")
plt.title("Contribution")
plt.savefig(name + 'contributions_production_2.png',bbox_inches = 'tight')

#1.4 Plot Production term and dissipation

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus2d[grid_line_1,:],P[grid_line_1,:],'b',markersize = 5,label = "$P^k$" + f" x = {np.round(x2d[grid_line_1,2],3)}")
plt.plot(yplus2d[grid_line_1,:],-diss2d[grid_line_1,:],'r--',markersize = 5,label = "$-\\varepsilon$" + f" x = {np.round(x2d[grid_line_1,2],3)}")
plt.legend(fontsize = "10")
plt.xlabel("$y^+$")
plt.ylabel("Amplitude")
plt.title("Production & Dissipation")
plt.savefig(name + 'prod_diss_1.png',bbox_inches = 'tight')

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(yplus2d[grid_line_2,:],P[grid_line_2,:],'b',markersize = 5,label = "$P^k" + f" x = {np.round(x2d[grid_line_2,2],3)}$")
plt.plot(yplus2d[grid_line_2,:],-diss2d[grid_line_2,:],'r--',markersize = 5,label = "$-\\varepsilon$" + f" x = {np.round(x2d[grid_line_2,2],3)}")
plt.legend(fontsize = "10")
plt.xlabel("$y^+$")
plt.ylabel("Amplitude")
plt.title("Production & Dissipation")
plt.savefig(name + 'prod_diss_2.png',bbox_inches = 'tight')



#%%%%%
#1.5 Plot Reynolds stress equation terms

#Choose i = k = 1, j = 2 => u'v'

c_mu, c_1, c_2, c_1w, c_2w, sigma_k, c_k = 0.09, 1.5, 0.6, 0.5, 0.3, 1, 0.2

term_1 = u2d[grid_line_1,:]*duv2ddx[grid_line_1,:] + v2d[grid_line_1,:]*duv2ddy[grid_line_1,:]
term_2 = viscos*(duv2ddx[grid_line_1,:] + duv2ddy[grid_line_1,:])
term_3_prod = -uu2d[grid_line_1,:]*dvdx[grid_line_1,:] - uv2d[grid_line_1,:]*dvdy[grid_line_1,:] -uv2d[grid_line_1,:]*dudx[grid_line_1,:] - vv2d[grid_line_1,:]*dudy[grid_line_1,:]

D_face_w,D_face_s = compute_face_phi(c_k*vv2d*(kres_2d/diss2d)*duv2ddx,fx,fy,ni,nj,zero_bc)
dDdy = dphidy(D_face_w,D_face_s,areawy,areasy,vol)
dDdx = dphidx(D_face_w,D_face_s,areawx,areasx,vol)

kres_face_w,kres_face_s = compute_face_phi(kres_2d,fx,fy,ni,nj,zero_bc)
dkdy = dphidy(kres_face_w,kres_face_s,areawy,areasy,vol)

dkdy_face_w,dkdy_face_s = compute_face_phi(dkdy,fx,fy,ni,nj,zero_bc)
d2kdy2 = dphidy(dkdy_face_w,dkdy_face_s,areawy,areasy,vol)

term_5_y = dDdy[grid_line_1,:] + dDdx[grid_line_1,:]
term_6_diss = [-(2/3)*diss2d[grid_line_1,j]  + viscos*d2kdy2[grid_line_1,j] if grid_line_1 == j else viscos*d2kdy2[grid_line_1,j] for j in range(len(term_5_y))]

n_1 = np.zeros((ni,nj))
n_2 = np.zeros((ni,nj))
f = np.zeros((ni,nj))
dist = np.zeros((ni,nj))
eps = np.zeros((ni,nj))
for i in range(1,ni-1):
   for j in range(1,nj-1):
      d = np.sqrt((xp2d[i+1,0] - xp2d[i,0])**2 + (yp2d[i+1,0] - yp2d[i,0])**2)
      s1 = (xp2d[i+1,0] -xp2d[i,0])/d
      s2 = (y2d[i,j] - y2d[i-1,j])/d
      n_1[i,j] = s2
      n_2[i,j] = -s1
      

      dist[i,j] = np.sqrt((x2d[i,j] - x2d[i,1])**2 + (y2d[i,j] - y2d[i,1])**2)
      if i == grid_line_1 and i == j:
         eps[i,j] = -diss2d[i,j] + viscos*d2kdy2[grid_line_1,j]
      else:
         eps[i,j] = viscos*d2kdy2[grid_line_1,j]

      f[i,j] = np.min([kres_2d[i,j]**(3/2)/(2.55*np.abs(n_1[i,j]*(xp2d[i,j] - xp2d[i,0]) + n_2[i,j]*(yp2d[i,j] - yp2d[i,0]))*eps[i,j]),1.0])

phi_12_1 = -c_1*diss2d/kres_2d*uv2d
phi_12_1w = -(3/2)*c_1w*diss2d/kres_2d*uv2d*f
phi_12_2 = -c_2*(-uu2d*dvdx - uv2d*dvdy - uv2d*dudx - vv2d*dudy)

phi_11_2 = -c_2*(-uu2d*dudx - uv2d*dudy - uu2d*dudx - uv2d*dudy - (2/3)*term_3_prod)
phi_22_2 = -c_2*(-uv2d*dvdx - vv2d*dvdy - uv2d*dvdx - vv2d*dvdy)
phi_12_2w = c_2w*(-(3/2)*(phi_11_2*n_1*n_2 + phi_12_2*n_2*n_2) - (3/2)*(phi_12_2*n_1*n_1 + phi_22_2*n_1*n_2))*f

phi = phi_12_1 + phi_12_2 + phi_12_1w + phi_12_2w



fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(term_1,yplus2d[grid_line_1,:],'b',markersize = 2,label = "$\overline{u} \partial \overline{u'v'}/\partial x$")
plt.plot(term_2,yplus2d[grid_line_1,:],'r--',markersize=2,label = "$\\nu \partial^2 \overline{u'v'}/\partial x^2$")
plt.plot(term_3_prod,yplus2d[grid_line_1,:],'k-.',markersize = 2,label = "$P_{i,j}$")
plt.plot(term_5_y,yplus2d[grid_line_1,:],'g-.',fillstyle = 'none',markersize = 2,linewidth = 2,label = "$D_{i,j,2},m=2$")
plt.plot(term_6_diss,yplus2d[grid_line_1,:],linewidth = 2,label = "$\\varepsilon - \\nu \partial^2 k/\partial y^2$")
#plt.axis([-1550,300,225,500])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("$\overline{u'v'}$")
plt.title("Terms in " + "Reynolds stress equation" + f" along x = {np.round(x2d[grid_line_1,5],3)}")
plt.savefig(name + 'terms_reynolds_uv.png',bbox_inches = 'tight')

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(phi[grid_line_1,:],yplus2d[grid_line_1,:],linewidth = 2,label = "$\\Phi$")
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("$\overline{u'v'}$")
plt.title("Terms in " + "Reynolds stress equation" + f" along x = {np.round(x2d[grid_line_1,5],3)}")
plt.savefig(name + 'terms_6_uv.png',bbox_inches = 'tight')


#Choose i = j = 1 => u'u'

c_mu, c_1, c_2, c_1w, c_2w, sigma_k, c_k = 0.09, 1.5, 0.6, 0.5, 0.3, 1, 0.2

term_1 = u2d[grid_line_1,:]*duu2ddx[grid_line_1,:] + v2d[grid_line_1,:]*duu2ddy[grid_line_1,:]
term_2 = viscos*(d2uu2ddx2[grid_line_1,:] + d2uu2ddy2[grid_line_1,:])
term_3_prod = -uu2d[grid_line_1,:]*dudx[grid_line_1,:] - uv2d[grid_line_1,:]*dudy[grid_line_1,:] -uu2d[grid_line_1,:]*dudx[grid_line_1,:] - uv2d[grid_line_1,:]*dudy[grid_line_1,:]

D_face_w,D_face_s = compute_face_phi(c_k*vv2d*(kres_2d/diss2d)*duu2ddx,fx,fy,ni,nj,zero_bc)
dDdy = dphidy(D_face_w,D_face_s,areawy,areasy,vol)
dDdx = dphidx(D_face_w,D_face_s,areawx,areasx,vol)

kres_face_w,kres_face_s = compute_face_phi(kres_2d,fx,fy,ni,nj,zero_bc)
dkdy = dphidy(kres_face_w,kres_face_s,areawy,areasy,vol)

dkdy_face_w,dkdy_face_s = compute_face_phi(dkdy,fx,fy,ni,nj,zero_bc)
d2kdy2 = dphidy(dkdy_face_w,dkdy_face_s,areawy,areasy,vol)

term_5_y = dDdy[grid_line_1,:] + dDdx[grid_line_1,:]
term_6_diss = [-(2/3)*diss2d[grid_line_1,j]  + viscos*d2kdy2[grid_line_1,j] if grid_line_1 == j else viscos*d2kdy2[grid_line_1,j] for j in range(len(term_5_y))]

n_1 = np.zeros((ni,nj))
n_2 = np.zeros((ni,nj))
f = np.zeros((ni,nj))
dist = np.zeros((ni,nj))
eps = np.zeros((ni,nj))
for i in range(1,ni-1):
   for j in range(1,nj-1):
      d = np.sqrt((xp2d[i+1,0] - xp2d[i,0])**2 + (yp2d[i+1,0] - yp2d[i,0])**2)
      s1 = (xp2d[i+1,0] -xp2d[i,0])/d
      s2 = (y2d[i,j] - y2d[i-1,j])/d
      n_1[i,j] = s2
      n_2[i,j] = -s1
      

      dist[i,j] = np.sqrt((x2d[i,j] - x2d[i,1])**2 + (y2d[i,j] - y2d[i,1])**2)
      if i == grid_line_1 and i == j:
         eps[i,j] = -diss2d[i,j] + viscos*d2kdy2[grid_line_1,j]
      else:
         eps[i,j] = viscos*d2kdy2[grid_line_1,j]

      f[i,j] = np.min([kres_2d[i,j]**(3/2)/(2.55*np.abs(n_1[i,j]*(xp2d[i,j] - xp2d[i,0]) + n_2[i,j]*(yp2d[i,j] - yp2d[i,0]))*eps[i,j]),1.0])

phi_11_1 = -c_1*diss2d/kres_2d*(uu2d  -(2/3)*kres_2d)
phi_11_1w = c_1w*diss2d/kres_2d*vv2d*f
phi_12_1 = -c_1*diss2d/kres_2d*uv2d
phi_12_1w = -(3/2)*c_1w*diss2d/kres_2d*uv2d*f
phi_12_2 = -c_2*(-uu2d*dvdx - uv2d*dvdy - uv2d*dudx - vv2d*dudy)

phi_11_2 = -c_2*(-uu2d*dudx - uv2d*dudy - uu2d*dudx - uv2d*dudy - (2/3)*term_3_prod)
phi_22_2 = -c_2*(-uv2d*dvdx - vv2d*dvdy - uv2d*dvdx - vv2d*dvdy)
phi_12_2w = c_2w*(-(3/2)*(phi_11_2*n_1*n_2 + phi_12_2*n_2*n_2) - (3/2)*(phi_12_2*n_1*n_1 + phi_22_2*n_1*n_2))*f
phi_11_2w = c_2w*(phi_11_2*n_1*n_1 + phi_12_2*n_2*n_1 + phi_12_2*n_1*n_2 + phi_22_2*n_2*n_2 - (3/2)*(phi_11_2*n_1*n_1 + phi_12_2*n_1*n_2) - (3/2)*(phi_11_2*n_1*n_1 + phi_12_2*n_1*n_2))

phi = phi_11_1 + phi_11_2 + phi_11_1w + phi_11_2w



fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(term_1,yplus2d[grid_line_1,:],'b',markersize = 2,label = "$\overline{u} \partial \overline{u'v'}/\partial x$")
plt.plot(term_2,yplus2d[grid_line_1,:],'r--',markersize=2,label = "$\\nu \partial^2 \overline{u'v'}/\partial x^2$")
plt.plot(term_3_prod,yplus2d[grid_line_1,:],'k-.',markersize = 2,label = "$P_{i,j}$")
plt.plot(term_5_y,yplus2d[grid_line_1,:],'g-.',fillstyle = 'none',markersize = 2,linewidth = 2,label = "$D_{i,j,2},m=2$")
plt.plot(term_6_diss,yplus2d[grid_line_1,:],linewidth = 2,label = "$\\varepsilon - \\nu \partial^2 k/\partial y^2$")
#plt.axis([-1550,300,225,500])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("$\overline{u'^2}$")
plt.title("Terms in " + "Reynolds stress equation" + f" along x = {np.round(x2d[grid_line_1,5],3)}")
plt.savefig(name + 'terms_reynolds_uu.png',bbox_inches = 'tight')

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(phi[grid_line_1,:],yplus2d[grid_line_1,:],linewidth = 2,label = "$\\Phi$")
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("$\overline{u'^2}$")
plt.title("Terms in " + "Reynolds stress equation" + f" along x = {np.round(x2d[grid_line_1,5],3)}")
plt.savefig(name + 'terms_6_uu.png',bbox_inches = 'tight')

#%%%%
#1.6 Compare Boussinesq withh database values

uv2d_boussinesq = [-2*c_mu*(kres_2d[grid_line_1,j]**2 / diss2d[grid_line_1,j])*(kres_2d[grid_line_1,j] / diss2d[grid_line_1,j] * 0.5* (dudy[grid_line_1,j] + dvdx[grid_line_1,j]))\
                  + (2/3)*kres_2d[grid_line_1,j] if j == grid_line_1 else -2*c_mu*(kres_2d[grid_line_1,j]**2 / diss2d[grid_line_1,j])*(kres_2d[grid_line_1,j] / diss2d[grid_line_1,j] * 0.5* (dudy[grid_line_1,j] + dvdx[grid_line_1,j])) \
                  for j in range(len(term_5_y))]

uu2d_boussinesq = [-2*c_mu*(kres_2d[grid_line_1,j]**2 / diss2d[grid_line_1,j])*(kres_2d[grid_line_1,j] / diss2d[grid_line_1,j] * 0.5* (dudx[grid_line_1,j] + dudx[grid_line_1,j]))\
                  + (2/3)*kres_2d[grid_line_1,j] if j == grid_line_1 else -2*c_mu*(kres_2d[grid_line_1,j]**2 / diss2d[grid_line_1,j])*(kres_2d[grid_line_1,j] / diss2d[grid_line_1,j] * 0.5* (dudx[grid_line_1,j] + dudx[grid_line_1,j])) \
                  for j in range(len(term_5_y))]

vv2d_boussinesq = [-2*c_mu*(kres_2d[grid_line_1,j]**2 / diss2d[grid_line_1,j])*(kres_2d[grid_line_1,j] / diss2d[grid_line_1,j] * 0.5* (dvdy[grid_line_1,j] + dvdy[grid_line_1,j]))\
                  + (2/3)*kres_2d[grid_line_1,j] if j == grid_line_1 else -2*c_mu*(kres_2d[grid_line_1,j]**2 / diss2d[grid_line_1,j])*(kres_2d[grid_line_1,j] / diss2d[grid_line_1,j] * 0.5* (dvdy[grid_line_1,j] + dvdy[grid_line_1,j])) \
                  for j in range(len(term_5_y))]

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(uv2d_boussinesq,yplus2d[grid_line_1,:],'b--',markersize = 2,label = "$\overline{u'v'}_{boussinesq}$")
plt.plot(uv2d[grid_line_1,:],yplus2d[grid_line_1,:],'b',markersize=2,label = "$\overline{u'v'}$")
plt.plot(uu2d_boussinesq,yplus2d[grid_line_1,:],'r--',markersize = 2,label = "$\overline{u'u'}_{boussinesq}$")
plt.plot(uu2d[grid_line_1,:],yplus2d[grid_line_1,:],'r',fillstyle = 'none',markersize = 2,linewidth = 2,label = "$\overline{u'u'}$")
plt.plot(vv2d_boussinesq,yplus2d[grid_line_1,:],'k-.',markersize = 2,label = "$\overline{v'v'}_{boussinesq}$")
plt.plot(vv2d[grid_line_1,:],yplus2d[grid_line_1,:],'k',fillstyle = 'none',markersize = 2,linewidth = 2,label = "$\overline{v'v'}$")
#plt.axis([-1550,300,225,500])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title("Boussinesq stresses compared to database" + f" along x = {np.round(x2d[grid_line_1,5],3)}")
plt.savefig(name + 'boussinesq.png',bbox_inches = 'tight')


fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(uv2d_boussinesq,yplus2d[grid_line_1,:],'b--',markersize = 2,label = "$\overline{u'v'}_{boussinesq}$")
plt.plot(uv2d[grid_line_1,:],yplus2d[grid_line_1,:],'b',markersize=2,label = "$\overline{u'v'}$")
plt.plot(uu2d_boussinesq,yplus2d[grid_line_1,:],'r--',markersize = 2,label = "$\overline{u'u'}_{boussinesq}$")
plt.plot(uu2d[grid_line_1,:],yplus2d[grid_line_1,:],'r',fillstyle = 'none',markersize = 2,linewidth = 2,label = "$\overline{u'u'}$")
plt.plot(vv2d_boussinesq,yplus2d[grid_line_1,:],'k-.',markersize = 2,label = "$\overline{v'v'}_{boussinesq}$")
plt.plot(vv2d[grid_line_1,:],yplus2d[grid_line_1,:],'k',fillstyle = 'none',markersize = 2,linewidth = 2,label = "$\overline{v'v'}$")
plt.axis([-1,2.5,0,50])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title("Boussinesq stresses close to the wall")
plt.savefig(name + 'boussinesq_wall_zoom.png',bbox_inches = 'tight')


#%%%%
#1.7 Plot modelled production term

P = -uv2d*dudy

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xp2d[:,:], yp2d[:,:], P, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax.set_zlabel("Production")
ax.set_ylabel("y")
ax.set_xlabel("x")
plt.title("Exact production in the domain")
plt.savefig(name + "production3d.png")


#%%
#1.8 Eigenvalues
ny_t = c_mu*kres_2d**2/diss2d

ny_t_grid_1 = ny_t[grid_line_1,:]
ny_t_grid_2 = ny_t[grid_line_2,:]


s11 = dudx
s12 = 0.5*(dudy + dvdx)
s22 = dvdy
sij = s11 + 2*s12 + s22

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(kres_2d[grid_line_1,:]/3 * np.sqrt(2/(sij[grid_line_1,:]*sij[grid_line_1,:])),yplus2d[grid_line_1,:],'b--',markersize = 4,label = "$ k / 3*(2/\hat{s}_{i,j} \hat{s}_{i,j})^{1/2}$")
plt.plot(ny_t_grid_1,yplus2d[grid_line_1,:],'g-.',markersize = 2,label = "$\\nu_t$")
#plt.xlim([0,1])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title("Eigenvalues of strain-rate tensor " + "$\hat{s}_{i,j}$"+ f" along x = {np.round(xp2d[grid_line_1,5],3)}")
plt.savefig(name + 'eigen_values_grid_1.png',bbox_inches = 'tight')

fig59,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
plt.plot(kres_2d[grid_line_2,:]/3 *  np.sqrt(2/(sij[grid_line_2,:]*sij[grid_line_2,:])),yplus2d[grid_line_2,:],'b--',markersize = 4,label = "$ k / 3*(2/\hat{s}_{i,j} \hat{s}_{i,j})^{1/2}$")
plt.plot(ny_t_grid_2,yplus2d[grid_line_2,:],'g-.',markersize = 2,label = "$\\nu_t$")
#plt.xlim([0,1])
plt.legend(fontsize = "10")
plt.ylabel("$y^+$")
plt.xlabel("Amplitude")
plt.title("Eigenvalues of strain-rate tensor " + "$\hat{s}_{i,j}$"+ f" along x = {np.round(xp2d[grid_line_2,5],3)}")
plt.savefig(name + 'eigen_values_grid_2.png',bbox_inches = 'tight')

limiter = kres_2d/3 * np.sqrt(2/(sij*sij))

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xp2d[:,:], yp2d[:,:], limiter, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax.set_zlim([0,0.5])

ax.set_ylabel("y")
ax.set_xlabel("x")
plt.title("Limiter in the domain")
plt.savefig(name + "limiter3d.png")

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(xp2d[:,:], yp2d[:,:], ny_t, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax.set_ylabel("y")
ax.set_xlabel("x")
plt.title("$\\nu_t$" + " in the domain")
plt.savefig(name + "ny_t3d.png")