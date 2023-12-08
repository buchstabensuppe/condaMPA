from gekko import GEKKO
import pandas as pd
import matplotlib.pyplot as plt
import CSTRgekko
from CSTRgekko import simCSTR1
import numpy as np
# load data and parse into columns
# url = 'http://apmonitor.com/do/uploads/Main/tclab_dyn_data2.txt'
# data = pd.read_csv(url)
# t = data['Time']
# u = data[['H1','H2']]
# y = data[['T1','T2']]

seconds =int(50)
tstep = 0.1 #[s]
n_variables = 3
x0s = [100, 50, 0]
x_train, y_train, t = simCSTR1(seconds, tstep, n_variables, x0s)
u = np.array(x_train)
y = np.array(y_train)


# generate time-series model
m = GEKKO(remote=True) # remote=True for MacOS

# system identification
na = 3  # output coefficients
nb = 3  # input coefficients
yp,p,K = m.sysid(t,u,y,na,nb,diaglevel=1,pred='meas') #source: https://apmonitor.com/pds/notebooks/12_time_series.html

plt.figure(figsize=(10,6))
plt.subplot(2,1,1)
plt.plot(t,u)
plt.legend([r'$u_0$',r'$u_1$'])
plt.ylabel('MVs')
plt.subplot(2,1,2)
plt.plot(t,y)
plt.plot(t,yp)
plt.legend([r'$y_0$',r'$y_1$',r'$z_0$',r'$z_1$'])
plt.ylabel('CVs')
plt.xlabel('Time')
plt.savefig('sysid.png')
plt.show()

breakbreak = True