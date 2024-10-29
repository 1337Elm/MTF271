import scipy.io as sio
import numpy as np
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
plt.rcParams.update({'font.size': 22})

plt.interactive(True)
name_dat = "Assignment 2/Assignment 2a/data/"

dx=0.05
dz=0.025

ni=34
nj=49
nk=34


viscos=1./5200.

# loop over nfiles 
nfiles=16
#initialize fields
u3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))
v3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))
w3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))
w3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))
te3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))
eps3d_nfiles=np.zeros((ni,nj,nk,nfiles+1))

for n in range(1,nfiles+1):
   print('file no: n=',n)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  read v_1 & transform v_1 to a 3D array (file 1)
   uvw = sio.loadmat(name_dat + 'u'+str(n)+'_IDD_PANS.mat')
   ttu=uvw['u'+str(n)+'_IDD_PANS']
   u3d_nfiles[:,:,:,n]= np.reshape(ttu,(nk,nj,ni))  #v_1 velocity
# N.B.- We don't have to swich axex since python and fortran stores an array in the same way

   uvw = sio.loadmat(name_dat + 'v'+str(n)+'_IDD_PANS.mat')
   tt=uvw['v'+str(n)+'_IDD_PANS']
   v3d_nfiles[:,:,:,n]= np.reshape(tt,(nk,nj,ni))  #v_2 velocity

   uvw = sio.loadmat(name_dat + 'w'+str(n)+'_IDD_PANS.mat')
   tt=uvw['w'+str(n)+'_IDD_PANS']
   w3d_nfiles[:,:,:,n]= np.reshape(tt,(nk,nj,ni))  #v_3 velocity

   uvw = sio.loadmat(name_dat + 'te'+str(n)+'_IDD_PANS.mat')
   tt=uvw['te'+str(n)+'_IDD_PANS']
   te3d_nfiles[:,:,:,n]= np.reshape(tt,(nk,nj,ni))  #modeled turbulent kinetic energy

   uvw = sio.loadmat(name_dat + 'eps'+str(n)+'_IDD_PANS.mat')
   tt=uvw['eps'+str(n)+'_IDD_PANS']
   eps3d_nfiles[:,:,:,n]= np.reshape(tt,(nk,nj,ni))  #modeled turbulent dissipation


# merge nfiles. This means that new ni = nfiles*ni
u3d=u3d_nfiles[:,:,:,1]
v3d=v3d_nfiles[:,:,:,1]
w3d=w3d_nfiles[:,:,:,1]
te3d=te3d_nfiles[:,:,:,1]
eps3d=eps3d_nfiles[:,:,:,1]
for n in range(1,nfiles+1):
   u3d=np.concatenate((u3d, u3d_nfiles[:,:,:,n]), axis=0)
   v3d=np.concatenate((v3d, v3d_nfiles[:,:,:,n]), axis=0)
   w3d=np.concatenate((w3d, w3d_nfiles[:,:,:,n]), axis=0)
   te3d=np.concatenate((te3d, te3d_nfiles[:,:,:,n]), axis=0)
   eps3d=np.concatenate((eps3d, eps3d_nfiles[:,:,:,n]), axis=0)



# x coordinate direction = index 0, first index
# y coordinate direction = index 1, second index
# z coordinate direction = index 2, third index



ni=len(u3d)

x=dx*ni
z=dz*nk


umean=np.mean(u3d, axis=(0,2))
vmean=np.mean(v3d, axis=(0,2))
wmean=np.mean(w3d, axis=(0,2))
temean=np.mean(te3d, axis=(0,2))
epsmean=np.mean(eps3d, axis=(0,2))

# face coordinates
yc = np.loadtxt(name_dat + "yc.dat")

# cell cener coordinates
y= np.zeros(nj)
dy=np.zeros(nj)
for j in range (1,nj-1):
# dy = cell width
   dy[j]=yc[j]-yc[j-1]
   y[j]=0.5*(yc[j]+yc[j-1])

y[nj-1]=yc[nj-1]
tauw=viscos*umean[1]/y[1]
ustar=tauw**0.5
yplus=y*ustar/viscos

DNS=np.genfromtxt(name_dat + "LM_Channel_5200_mean_prof.dat", dtype=None,comments="%")
y_DNS=DNS[:,0]
yplus_DNS=DNS[:,1]
u_DNS=DNS[:,2]

DNS=np.genfromtxt(name_dat + "LM_Channel_5200_vel_fluc_prof.dat", dtype=None,comments="%")

u2_DNS=DNS[:,2]
v2_DNS=DNS[:,3]
w2_DNS=DNS[:,4]
uv_DNS=DNS[:,5]

k_DNS=0.5*(u2_DNS+v2_DNS+w2_DNS)

# load RANS data
rans = np.loadtxt(name_dat + 'y_u_k_eps_uv_5200-RANS-code.txt')
y_rans = rans[:,0]
u_rans = rans[:,1]
k_rans = rans[:,2]
eps_rans = rans[:,3]

# find equi.distant DNS cells in log-scale
xx=0.
jDNS=[1]*40
for i in range (0,40):
   i1 = (np.abs(10.**xx-yplus_DNS)).argmin()
   jDNS[i]=int(i1)
   xx=xx+0.2


#S.2 Mean velocity profile
############################### U log
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.semilogx(yplus,umean/ustar,'b-')
plt.semilogx(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
plt.axis([1, 8000, 0, 31])
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")

plt.savefig('Assignment 2/Assignment 2a/u_log_python_eps.png')


############################### U linear plus zoom
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(yplus,umean/ustar,'b-')
plt.plot(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
plt.axis([1, 5200, 0, 31])
plt.ylabel("$U^+$")
plt.xlabel("$y^+$")


# make a zoom
axins1 = inset_axes(ax1, width="50%", height="50%", loc='lower right', borderpad=1) # borderpad = space to the main figure
# reduce fotnsize 
axins1.tick_params(axis = 'both', which = 'major', labelsize = 10)
#axins1.yaxis.set_label_position("left")
axins1.xaxis.set_label_position("top")  # put x labels at the top
axins1.xaxis.tick_top()                 # put x ticks at the top
#axins1.yaxis.tick_left()
#axins1.xaxis.tick_bottom()
plt.plot(yplus,umean/ustar,'b-')
plt.plot(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
plt.axis([0, 30, 0, 14])

# Turn ticklabels of insets off
#axins1.tick_params(labelleft=False, labelbottom=False)
#axins1.tick_params(labelleft=False)


plt.savefig('Assignment 2/Assignment 2a/u_linear-zoom_python_eps.png')



#S.3 Resolved stresses
uv=np.zeros(nj)
for k in range (0,nk):
    for j in range (0,nj):
        for i in range (0,ni):
            ufluct=u3d[i,j,k]-umean[j]
            vfluct=v3d[i,j,k]-vmean[j]
            wfluct=w3d[i,j,k]-wmean[j]
            uv[j]=uv[j]+ufluct*vfluct

uv=uv/ni/nk

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.semilogx(yplus,uv,'b-')
plt.semilogx(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
#plt.axis([1, 8000, 0, 31])
plt.ylabel("$\\langle \overline{v}_1,\overline{v}_2 \\rangle$")
plt.xlabel("$y^+$")
plt.title("$\\langle \overline{v}_1,\overline{v}_2 \\rangle$" + " log plot")
plt.savefig('Assignment 2/Assignment 2a/uv_log.png')

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(yplus,uv,'b-')
#plt.semilogx(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
#plt.axis([1, 8000, 0, 31])
plt.ylabel("$\\langle \overline{v}_1,\overline{v}_2 \\rangle$")
plt.xlabel("$y^+$")
plt.title("$\\langle \overline{v}_1,\overline{v}_2 \\rangle$" + " linear plot")

plt.savefig('Assignment 2/Assignment 2a/uv_linear.png')


#S.4 Turbulent kinetic energy

uu=np.zeros(nj)
vv=np.zeros(nj)
ww=np.zeros(nj)
for k in range (0,nk):
    for j in range (0,nj):
        for i in range (0,ni):
            ufluct=u3d[i,j,k]-umean[j]
            vfluct=v3d[i,j,k]-vmean[j]
            wfluct=w3d[i,j,k]-wmean[j]
            uu[j]=uu[j]+ufluct*ufluct
            uu[j]=uu[j]+vfluct*vfluct
            uu[j]=uu[j]+wfluct*wfluct

uu=uu/ni/nk
vv=vv/ni/nk
ww=ww/ni/nk

k_res = 0.5*(uu + vv + ww)

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(yplus,temean,'b-',label="$k_{mod}$")
plt.plot(yplus,k_res,'r-',label="$k_{res}$")

#plt.semilogx(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
#plt.axis([-100, 5000, 0, 10])
plt.ylabel("$k$")
plt.xlabel("$y^+$")
plt.legend()
plt.title("Modeled and resolved energy")

plt.savefig('Assignment 2/Assignment 2a/k_res_mod_linear.png')


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(yplus,k_res + temean,'bo',label="$k_{res} + k_{mod}$")
plt.plot(yplus_DNS[jDNS],k_DNS[jDNS],'b',label="$k_{DNS}$")
#plt.semilogx(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
#plt.axis([-100, 5000, 0, 10])
plt.ylabel("$k$")
plt.xlabel("$y^+$")
plt.legend()
plt.title("Comparison")

plt.savefig('Assignment 2/Assignment 2a/k_DNS_comparison.png')


#S.5 The modelled turbulent shear stress (WHAT IS THE RESOLVED STRESS??)
c_my = 0.09
U_eps = (epsmean[1:]*viscos)**(1/4)
y_star = (U_eps*y[1:])/viscos
R_t = temean[1:]**2 / (viscos*epsmean[1:])
f_my = (1-np.exp(-y_star/14))**2 * (1 + (5/(R_t**(3/4)))*np.exp(-(R_t/200)**2))
viscos_t = c_my*f_my*temean[1:]**2/epsmean[1:]


dudx, dudy, dudz=np.gradient(u3d,dx,y,dz)
dvdx, dvdy, dvdz=np.gradient(v3d,dx,y,dz)
dwdx, dwdy, dwdz=np.gradient(w3d,dx,y,dz)

dudy_mean = np.mean(dudy, axis=(0,2))
dvdx_mean = np.mean(dvdx, axis=(0,2))
#Time average

#
tau_12 = -viscos_t*(dudy_mean[1:] + dvdx_mean[1:])

f_k = temean/(k_res + temean)
interface = np.abs(f_k- np.repeat(0.4,len(f_k))).argmin() -1


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(yplus[1:],tau_12,'b',label="$\\tau_{mod}$")
plt.plot(yplus,uv,'r',label="$\overline{u'v'}$")
plt.axvline(x=yplus[interface])
#plt.semilogx(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
#plt.axis([-100, 5000, 0, 10])
#plt.ylabel("$\\langle \\tau_{12} \\rangle$")
#plt.axis([0,5000,-0.8,0.01])
plt.xlabel("$y^+$")
plt.legend()
plt.title("Comparison")

plt.savefig('Assignment 2/Assignment 2a/tau_comparison.png')

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(yplus[1:],tau_12+uv[1:],'b',label="$\\tau_{mod} + \overline{u'v'}$")
plt.plot(yplus_DNS[jDNS],uv_DNS[jDNS],'r',label="$\overline{u'v'}_{DNS}$")
plt.axvline(x=yplus[interface])

#plt.semilogx(yplus_DNS[jDNS],u_DNS[jDNS],'bo')
#plt.axis([-100, 5000, 0, 10])
#plt.ylabel("$\\langle \\tau_{12} \\rangle$")
plt.xlabel("$y^+$")
plt.legend()
plt.title("Comparison")

plt.savefig('Assignment 2/Assignment 2a/tau_comparison_DNS.png')


#S.6 Location of interface in DES and DDES (mean of dy??)
beta_star = 0.09
a1 = 0.3
sigma_omega_k_eps = 0.5
alpha_k_omega = 5/9
alpha_k_eps = 0.44
omega = eps3d/(c_my*te3d)
omegamean = np.mean(omega,axis=(0,2))

f_k = temean/(k_res + temean)
C_des = 0.65
delta = np.max([dx,np.mean(dy),dz])


#SA-DES
switch_SA_DES = 0
for i in range(nj):
    if yplus[i] == C_des*delta:
        switch_SA_DES = yplus[i]

#SST-DES
F_des = np.zeros(nj-1)
for i in range(nj-1):
    F_des[i] = np.max([temean[i+1]**0.5/(C_des*beta_star*delta*omegamean[i+1]),1])

#DDES
xsi = np.max([(2*temean[1:]**0.5)/(beta_star*omegamean[1:]*yplus[1:]),(500*viscos)/(yplus[1:]**2 * omegamean[1:])])
F_s = np.tanh(xsi**2)
L_t = temean[1:]**0.5 /(beta_star*omegamean[1:])

F_ddes = np.zeros(nj-2)
for i in range(nj-2):
    F_ddes[i] = np.max([L_t[i+1]/(C_des*delta)*(1-F_s),1])


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(yplus,np.repeat(C_des*delta,len(yplus)),'b',label="$C_{DES}\\Delta$")
plt.plot(yplus,yplus,'k',label="$y^+$")

plt.axis([0,0.1,0.03,0.05])
plt.xlabel("$y$")
plt.legend()
plt.title("Interface SA-DES")

plt.savefig('Assignment 2/Assignment 2a/interface_SA-DES.png')


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(yplus[1:],F_des,'b',label="$F_{SST-DES}$")
plt.plot(yplus,np.ones(len(yplus)),'k-.')
plt.xlabel("$y^+$")
plt.legend()
plt.title("Interface")

plt.savefig('Assignment 2/Assignment 2a/interface.png')

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(yplus,f_k,'b',label="$f_k$")
plt.plot(yplus,np.repeat(0.4,len(yplus)),'k')
plt.xlabel("$y^+$")
plt.legend()
plt.title("Interface")

plt.savefig('Assignment 2/Assignment 2a/interface_f_k.png')


fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(yplus[2:],F_ddes,'b',label="$F_{DDES}$")
plt.plot(yplus,np.ones(len(yplus)),'k-.')

plt.xlabel("$y^+$")
plt.legend()
plt.title("Interface")

plt.savefig('Assignment 2/Assignment 2a/interface_ddes.png')

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(yplus[2:],F_ddes,'b',label="$F_{DDES}$")
plt.plot(yplus[1:],F_des,'r',label="$F_{DES}$")

plt.plot(yplus,np.ones(len(yplus)),'k-.')

plt.xlabel("$y^+$")
plt.legend()
plt.title("Interface")

plt.savefig('Assignment 2/Assignment 2a/interface_ddes_des.png')


#S.7 SAS turbulent length scales
kappa = 0.41

du_meandy = np.gradient(umean,y)
d2u_meandy2 = np.gradient(du_meandy,y)

L_steady  = kappa*np.abs(du_meandy,d2u_meandy2)


s11 = 0.5*(dudx + dudx)
s11 = dudx
s12 = 0.5*(dudy + dvdx)
s22 = dvdy
S = (2*(s11**2 + 2*s12**2 + s22**2))**0.5


dudx, dudy, dudz=np.gradient(u3d,dx,y,dz)
dvdx, dvdy, dvdz=np.gradient(v3d,dx,y,dz)
dwdx, dwdy, dwdz=np.gradient(w3d,dx,y,dz)


#Something weird with second derivatives??
second_derivative = np.gradient(dudx,dx,y,dz)
d2udx2 = second_derivative[0]

second_derivative = np.gradient(dudy,dx,y,dz)
d2udy2 = second_derivative[1]

second_derivative = np.gradient(dudz,dx,y,dz)
d2udz2 = second_derivative[2]



second_derivative = np.gradient(dvdx,dx,y,dz)
d2vdx2 = second_derivative[0]

second_derivative = np.gradient(dvdy,dx,y,dz)
d2vdy2 = second_derivative[1]

second_derivative = np.gradient(dvdz,dx,y,dz)
d2vdz2 = second_derivative[2]



second_derivative = np.gradient(dwdx,dx,y,dz)
d2wdx2 = second_derivative[0]

second_derivative = np.gradient(dwdy,dx,y,dz)
d2wdy2 = second_derivative[1]

second_derivative = np.gradient(dwdz,dx,y,dz)
d2wdz2 = second_derivative[2]

U_biss = ((d2udx2 + d2udy2 + d2udz2)*(d2udx2 + d2udy2 + d2udz2) + \
          (d2vdx2 + d2vdy2 + d2vdz2)*(d2vdx2 + d2vdy2 + d2vdz2) + \
            (d2wdx2 + d2wdy2 + d2wdz2)*(d2wdx2 + d2wdy2 + d2wdz2))

L_inst = kappa*np.abs(S/U_biss)

dkdx, dkdy, dkdz = np.gradient(te3d,dx,y,dz)
domegadx, domegady, domegadz = np.gradient(omega,dx,y,dz)

dkdx_mean = np.mean(dkdx,axis=(0,2))
dkdy_mean = np.mean(dkdy,axis=(0,2))
dkdz_mean = np.mean(dkdz,axis=(0,2))

domegadx_mean = np.mean(domegadx,axis=(0,2))
domegady_mean = np.mean(domegady,axis=(0,2))
domegadz_mean = np.mean(domegadz,axis=(0,2))

xsi = np.zeros(nj-1)
for i in range(1,nj-1):
    xsi[i] = np.min(np.max([temean[i]**0.5 / (beta_star * omegamean[i] * yplus[i]), \
                    (500*viscos)/(yplus[i]**2 * omegamean[i])])*(4*sigma_omega_k_eps*temean[i])\
                    /(np.max([2*sigma_omega_k_eps * (1/omegamean[i])*(dkdx_mean[i]*domegadx_mean[i] + dkdy_mean[i]*domegady_mean[i] + dkdz_mean[i]*domegadz_mean[i]),10**(-10)])*yplus[i]**2))
F1 = np.tanh(xsi**4)

eta = np.max([(2*temean[1:]**(1/2))/(beta_star*omegamean[1:]*yplus[1:]) , (500*viscos)/(yplus[1:]**2 *omegamean[1:])])
F2 = np.tanh(eta**2)

viscos_t_omega = (a1*temean[1:])/(np.max([a1*omegamean[1:],np.abs(2*(np.mean(s11,axis=(0,2))[1:]**2 + 2*np.mean(s12,axis=(0,2))[1:]**2 + np.mean(s22,axis=(0,2))[1:]**3)**0.5)*F2]))

alpha = F1*alpha_k_omega + (1-F1)*alpha_k_eps

P_k = 2*temean[1:]**0.5 * np.min([c_my* (temean[1:]**(3/2))/epsmean[1:], np.repeat(C_des*delta,len(temean)-1)]) * (np.mean(s11,axis=(0,2))[1:]**2 + 2*np.mean(s12,axis=(0,2))[1:]**2 + np.mean(s22,axis=(0,2))[1:]**2)
P_omega = alpha*P_k/viscos_t_omega


F_sas = 1
xsi_2 = 1.5

L= temean[1:]**0.5 / (omegamean[1:]* c_my**(0.25))

T1_inst = xsi_2*kappa*np.mean(S,axis=(0,2))[1:]**2 * (L/np.mean(L_inst,axis=(0,2))[1:])
T1_steady = xsi_2*kappa*np.mean(S,axis=(0,2))[1:]**2 * (L/L_steady[1:])

P_sas_inst = F_sas*T1_inst
P_sas_steady = F_sas*T1_steady

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(yplus[1:],P_sas_inst,'b',label="$P_{SAS-inst}$")
plt.plot(yplus[1:],P_sas_steady,'r',label="$P_{SAS-steady}$")
plt.plot(yplus[1:],P_omega,'k',label="$P^{\omega}$")
plt.ylim([0,5])

plt.xlim([150,5000])

plt.xlabel("$y^+$")
plt.legend()


plt.savefig('Assignment 2/Assignment 2a/production.png')