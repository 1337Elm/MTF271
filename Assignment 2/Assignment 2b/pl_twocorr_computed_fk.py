import scipy.io as sio
import numpy as np
import sys
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 22})
name_dat = "Assignment 2/Assignment 2b/data/"

# x=0.65
#   do itstep= ....
#         i=36
#         do j=10,100,20
#         do k=2,nk/2
#            write(67,*) phi(i,j,k,w)
#         end do
#         end do
#         write(67,*)ivisz
#   end do


w_time = np.loadtxt(name_dat + "w_time_z65.dat")

ntot=len(w_time)
nt=int(w_time[-1])  #Number of time steps
nj=5 # j=10,30,50,70,90
nk=16 # k=1-16

w_time_org=w_time

# every 81 element (nj*nk) is the timestep number
n=int(ntot/81)
idelete = np.linspace(80,ntot-1,n,dtype=int)

# remove the timestep numbers

w_time=np.delete(w_time, idelete)

w_y_z_t= np.reshape(w_time,(nt,nj,nk))
# swap axis
w_y_z_t_65= np.swapaxes(w_y_z_t,0,2)  # the order of the indices are (k,j,t)
w_y_z_t_65= np.swapaxes(w_y_z_t_65,0,1)


# x=0.8
w_time = np.loadtxt(name_dat + "w_time_z80.dat")

ntot=len(w_time)
nt=int(w_time[-1])  #Number of time steps
nj=5 # j=10,30,50,70,90
nk=16 # k=1-16

# every 81 element (nj*nk) is the timestep number
n=int(ntot/81)
idelete = np.linspace(80,ntot-1,n,dtype=int)

# remove the timestep numbers

w_time=np.delete(w_time, idelete)

w_y_z_t= np.reshape(w_time,(nt,nj,nk))
# swap axis
w_y_z_t_80= np.swapaxes(w_y_z_t,0,2)
w_y_z_t_80= np.swapaxes(w_y_z_t_80,0,1)

# x=1.1 
w_time = np.loadtxt(name_dat + "w_time_z110.dat")

ntot=len(w_time)
nt=int(w_time[-1])  #Number of time steps
nj=5 # j=10,30,50,70,90
nk=16 # k=1-16

# every 81 element (nj*nk) is the timestep number
n=int(ntot/81)
idelete = np.linspace(80,ntot-1,n,dtype=int)

# remove the timestep numbers

w_time=np.delete(w_time, idelete)

w_y_z_t= np.reshape(w_time,(nt,nj,nk))
# swap axis
w_y_z_t_110= np.swapaxes(w_y_z_t,0,2)
w_y_z_t_110= np.swapaxes(w_y_z_t_110,0,1)

# x=1.3 
w_time = np.loadtxt(name_dat + "w_time_z130.dat")

ntot=len(w_time)
nt=int(w_time[-1])  #Number of time steps
nj=5 # j=10,30,50,70,90
nk=16 # k=1-16

# every 81 element (nj*nk) is the timestep number
n=int(ntot/81)
idelete = np.linspace(80,ntot-1,n,dtype=int)

# remove the timestep numbers

w_time=np.delete(w_time, idelete)

w_y_z_t= np.reshape(w_time,(nt,nj,nk))
# swap axis
w_y_z_t_130= np.swapaxes(w_y_z_t,0,2) #nk,nj,nt
w_y_z_t_130= np.swapaxes(w_y_z_t_130,0,1)

x_pos = [0.65,0.8,1.1,1.3]
B33_norm065 = np.zeros((nj,nk))
B33_norm08 = np.zeros((nj,nk))
B33_norm11 = np.zeros((nj,nk))
B33_norm13 = np.zeros((nj,nk))
for j in range(nj):
   for k in range(nk):
      for n in range(nt):
            B33_norm065[j,k] += w_y_z_t_65[j,0,n]*w_y_z_t_65[j,k,n]/nt
            B33_norm08[j,k] += w_y_z_t_80[j,0,n]*w_y_z_t_80[j,k,n]/nt
            B33_norm11[j,k] += w_y_z_t_110[j,0,n]*w_y_z_t_110[j,k,n]/nt
            B33_norm13[j,k] += w_y_z_t_130[j,0,n]*w_y_z_t_130[j,k,n]/nt


   B33_norm065[j,:] /= B33_norm065[j,0]
   B33_norm08[j,:] /= B33_norm08[j,0]
   B33_norm11[j,:] /= B33_norm11[j,0]
   B33_norm13[j,:] /= B33_norm13[j,0]

   for p in range(nk):
      if B33_norm065[j,p] < 0:
         B33_norm065[j,p] = 0
      if B33_norm08[j,p] < 0:
         B33_norm08[j,p] = 0
      if B33_norm11[j,p] < 0:
         B33_norm11[j,p] = 0
      if B33_norm13[j,p] < 0:
         B33_norm13[j,p] = 0


z = np.linspace(0,0.2,nk)
for fig in range(4):
   plt.figure()
   plt.subplots_adjust(left=0.20,bottom=0.20)

   if fig == 0:
      plt.plot(z,B33_norm065[0,:],"b",label="$j=10$")
      plt.plot(z,B33_norm065[1,:],"r",label="$j=30$")
      plt.plot(z,B33_norm065[2,:],"k",label="$j=50$")
      plt.plot(z,B33_norm065[3,:],"g",label="$j=70$")
      plt.plot(z,B33_norm065[4,:],"b-.",label="$j=90$")
      plt.title(f"Two point correlation, x = {x_pos[fig]}")
      plt.savefig("Assignment 2/Assignment 2b/2point" + str(x_pos[fig]) + ".png")
   elif fig == 1:
      plt.plot(z,B33_norm08[0,:],"b",label="$j=10$")
      plt.plot(z,B33_norm08[1,:],"r",label="$j=30$")
      plt.plot(z,B33_norm08[2,:],"k",label="$j=50$")
      plt.plot(z,B33_norm08[3,:],"g",label="$j=70$")
      plt.plot(z,B33_norm08[4,:],"b-.",label="$j=90$")
   elif fig == 2:
      plt.plot(z,B33_norm11[0,:],"b",label="$j=10$")
      plt.plot(z,B33_norm11[1,:],"r",label="$j=30$")
      plt.plot(z,B33_norm11[2,:],"k",label="$j=50$")
      plt.plot(z,B33_norm11[3,:],"g",label="$j=70$")
      plt.plot(z,B33_norm11[4,:],"b-.",label="$j=90$")
   else:
      plt.plot(z,B33_norm13[0,:],"b",label="$j=10$")
      plt.plot(z,B33_norm13[1,:],"r",label="$j=30$")
      plt.plot(z,B33_norm13[2,:],"k",label="$j=50$")
      plt.plot(z,B33_norm13[3,:],"g",label="$j=70$")
      plt.plot(z,B33_norm13[4,:],"b-.",label="$j=90$")

   plt.xlim([z[0],z[-1]+0.01])
   plt.legend(fontsize = "xx-small")
   plt.xlabel("z")
   plt.ylabel("$B_{33}^{norm} ( x_3^A,\hat{x}_3)$")

   plt.title(f"Two point correlation, x = {x_pos[fig]}")
   plt.savefig("Assignment 2/Assignment 2b/2point" + str(x_pos[fig]) + ".png")
         

xy= np.loadtxt(name_dat + "hump_grid_nasa_les_coarse_noflow.dat")
x1=xy[:,0]
y1=xy[:,1]

nim1=int(x1[0])
njm1=int(y1[0])

ni=nim1+1
nj=njm1+1


x=x1[1:]
y=y1[1:]

x_2d=np.reshape(x,(njm1,nim1))
y_2d=np.reshape(y,(njm1,nim1))

x_2d=np.transpose(x_2d)
y_2d=np.transpose(y_2d)

# compute cell centers
xp2d= np.zeros((ni,nj))
yp2d= np.zeros((ni,nj))

for jj in range (0,nj):
   for ii in range (0,ni):

      im1=max(ii-1,0)
      jm1=max(jj-1,0)

      i=min(ii,nim1-1)
      j=min(jj,njm1-1)


      xp2d[ii,jj]=0.25*(x_2d[i,j]+x_2d[im1,j]+x_2d[i,jm1]+x_2d[im1,jm1])
      yp2d[ii,jj]=0.25*(y_2d[i,j]+y_2d[im1,j]+y_2d[i,jm1]+y_2d[im1,jm1])
