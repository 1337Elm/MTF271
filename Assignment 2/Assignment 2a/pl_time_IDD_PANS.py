import numpy as np
import matplotlib.pyplot as plt
import sys
#from scipy.signal import welch, hanning
from scipy.signal import welch, hann
plt.rcParams.update({'font.size': 22})

# ***** read u
# /chalmers/users/lada/python-DES-code/channel-5200-IDD-PANS-inlet-synt-MTF271/u-time-history-i70.dat

plt.interactive(True)
name = "Assignment 2/Assignment 2a/data/"
n1=1
n2=15000

data = np.genfromtxt(name + "u-time-history-i70.dat", dtype=None)
t=data[n1:n2,0] #time
u1=data[n1:n2,1]   #v_1 at point 1
u2=data[n1:n2,2]   #v_1 at point 2
u3=data[n1:n2,3]   #v_1 at point 3
u4=data[n1:n2,4]   #v_1 at point 4
u5=data[n1:n2,5]   #v_1 at point 5

data = np.genfromtxt(name + "w-time-history-i70.dat", dtype=None)
w1=data[n1:n2,1]   #v_3 at point 1
w2=data[n1:n2,2]   #v_3 at point 2
w3=data[n1:n2,3]   #v_3 at point 3
w4=data[n1:n2,4]   #v_3 at point 4
w5=data[n1:n2,5]   #v_3 at point 5

dx=0.1
dz=0.05
dt=t[1]-t[0]    

#S.1 Time History
# plot time history u1 BLUE, u2 RED, u3 BLACK, u4 GREEN, u5 YELLOW
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(t,u1,'b-')
plt.plot(t[::30],u1[::30],'bo',label="$\overline{v}_1 / u_\\tau$")

plt.plot(t,u2,'r-')
plt.plot(t[::30],u2[::30],'ro',label="$\overline{v}_1 / u_\\tau$")

plt.plot(t,u3,'k-')
plt.plot(t[::30],u3[::30],'ko',label="$\overline{v}_1 / u_\\tau$")


plt.plot(t,u4,'g-')
plt.plot(t[::30],u4[::30],'go',label="$\overline{v}_1 / u_\\tau$")


plt.plot(t,u5,'y-')
plt.plot(t[::30],u5[::30],'yo',label="$\overline{v}_1 / u_\\tau$")

plt.xlabel('$t$')
plt.axis([10, 11, 9,25])
plt.ylabel('$u$')
plt.title("History of " + "$\overline{v}_1 / u_\\tau$" + " at 5 nodes")
plt.savefig('Assignment 2/Assignment 2a/u-time_eps.png')


#W plots
fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)

plt.plot(t,w1,'b-')
plt.plot(t[::30],w1[::30],'bo',label="$\overline{v}_3 / u_\\tau$")

plt.plot(t,w2,'r-')
plt.plot(t[::30],w2[::30],'ro',label="$\overline{v}_3 / u_\\tau$")

plt.plot(t,w3,'k-')
plt.plot(t[::30],w3[::30],'ko',label="$\overline{v}_3 / u_\\tau$")

plt.plot(t,w4,'g-')
plt.plot(t[::30],w4[::30],'go',label="$\overline{v}_3 / u_\\tau$")

plt.plot(t,w5,'y-')
plt.plot(t[::30],w5[::30],'yo',label="$\overline{v}_3 / u_\\tau$")

plt.xlabel('$t$')
#plt.axis([10,11,-1.5, 1.5])
plt.xlim([10, 11])
plt.ylabel('$w$')
plt.title("History of " + "$\overline{v}_3 / u_\\tau$" + " at 5 nodes")
plt.savefig('Assignment 2/Assignment 2a/w-time_eps.png')

#Compute autocorrelation
u1_fluct=u1-np.mean(u1)
two=np.correlate(u1_fluct,u1_fluct,"full")
# find max
nmax=np.argmax(two)
# and its value
two_max=np.max(two)
# Pick the right half and normalize
two_sym_norm=two[nmax:]/two_max
#dt=t[1]-t[0]
node1v1_sym_norm_INT = np.zeros(nmax)

nmax_auto_1 = 0
for i in range(nmax):
    node1v1_sym_norm_INT[i] = two_sym_norm[i]
    if two_sym_norm[i] < two_sym_norm[i+1]:
        nmax_auto_1 = i        
        break

int_T_1=np.trapz(node1v1_sym_norm_INT)*dt


u2_fluct=u2-np.mean(u2)
two_2=np.correlate(u2_fluct,u2_fluct,"full")
# find max
nmax_2=np.argmax(two_2)
# and its value
two_max_2=np.max(two_2)
# Pick the right half and normalize
two_sym_norm_2=two_2[nmax_2:]/two_max_2

node1v1_sym_norm_INT = np.zeros(nmax_2)

nmax_auto_1 = 0
for i in range(nmax):
    node1v1_sym_norm_INT[i] = two_sym_norm_2[i]
    if two_sym_norm_2[i] < two_sym_norm_2[i+1]:
        nmax_auto_1 = i        
        break

int_T_2=np.trapz(node1v1_sym_norm_INT)*dt

u3_fluct=u3-np.mean(u3)
two_3=np.correlate(u3_fluct,u3_fluct,"full")
# find max
nmax_3=np.argmax(two_3)
# and its value
two_max_3=np.max(two_3)
# Pick the right half and normalize
two_sym_norm_3=two_3[nmax_3:]/two_max_3

node1v1_sym_norm_INT = np.zeros(nmax_3)

nmax_auto_1 = 0
for i in range(nmax):
    node1v1_sym_norm_INT[i] = two_sym_norm_3[i]
    if two_sym_norm_3[i] < two_sym_norm_3[i+1]:
        nmax_auto_1 = i        
        break

int_T_3=np.trapz(node1v1_sym_norm_INT)*dt

u4_fluct=u4-np.mean(u4)
two_4=np.correlate(u4_fluct,u4_fluct,"full")
# find max
nmax_4=np.argmax(two_4)
# and its value
two_max_4=np.max(two_4)
# Pick the right half and normalize
two_sym_norm_4=two_4[nmax_4:]/two_max_4

node1v1_sym_norm_INT = np.zeros(nmax_4)

nmax_auto_1 = 0
for i in range(nmax):
    node1v1_sym_norm_INT[i] = two_sym_norm_4[i]
    if two_sym_norm_4[i] < two_sym_norm_4[i+1]:
        nmax_auto_1 = i        
        break

int_T_4=np.trapz(node1v1_sym_norm_INT)*dt

u5_fluct=u5-np.mean(u5)
two_5=np.correlate(u5_fluct,u5_fluct,"full")
# find max
nmax_5=np.argmax(two_5)
# and its value
two_max_5=np.max(two_5)
# Pick the right half and normalize
two_sym_norm_5=two_5[nmax_5:]/two_max_5

node1v1_sym_norm_INT = np.zeros(nmax_5)

nmax_auto_1 = 0
for i in range(nmax):
    node1v1_sym_norm_INT[i] = two_sym_norm_5[i]
    if two_sym_norm_5[i] < two_sym_norm_5[i+1]:
        nmax_auto_1 = i        
        break

int_T_5=np.trapz(node1v1_sym_norm_INT)*dt

fig1,ax1 = plt.subplots()
plt.subplots_adjust(left=0.20,bottom=0.20)
imax=500;
plt.plot(t[0:imax],two_sym_norm[0:imax],"b-",label="node 1")
plt.plot(t[0:imax],two_sym_norm_2[0:imax],"r-",label="node 2")
plt.plot(t[0:imax],two_sym_norm_3[0:imax],"k-",label="node 3")
plt.plot(t[0:imax],two_sym_norm_4[0:imax],"g-",label="node 4")
plt.plot(t[0:imax],two_sym_norm_5[0:imax],"y-",label="node 5")
plt.xlabel("t")
plt.ylabel("$B_{uu}$")
plt.legend(fontsize=10)
plt.title("Autocorrelation")
plt.savefig("Assignment 2/Assignment 2a/Autocorrelation_u1.png")

print("Integral time scale for " + "$B_{uu}$" + " at node 1," +  "$T_{int}$" + f" = {int_T_1}")
print("Integral time scale for " + "$B_{uu}$" + " at node 2," +  "$T_{int}$" + f" = {int_T_2}")
print("Integral time scale for " + "$B_{uu}$" + " at node 3," +  "$T_{int}$" + f" = {int_T_3}")
print("Integral time scale for " + "$B_{uu}$" + " at node 4," +  "$T_{int}$" + f" = {int_T_4}")
print("Integral time scale for " + "$B_{uu}$" + " at node 5," +  "$T_{int}$" + f" = {int_T_5}")
