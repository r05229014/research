import numpy as np
import matplotlib.pyplot as plt

data = np.load('../data/data_1.npy')
# 1
plt.figure(1, figsize = (8,6))
plt.scatter(data[:,2]/1000000000, data[:,5])
plt.title('Cloud size distribution', fontsize=25)
plt.xlabel('cloud size (km^3)', fontsize=18)
plt.ylabel('core number', fontsize=18)
plt.xlim(0,)
plt.ylim(0)
plt.savefig('../img/cloud_size')
# 2
plt.figure(2, figsize = (8,6))
plt.scatter(data[:,3]/1000, data[:,5])
plt.title('Cloud Top height distribution', fontsize=25)
plt.xlabel('cloud top height (km)', fontsize=18)
plt.ylabel('core number', fontsize=18)
plt.xlim(0,)
plt.ylim(0)
plt.savefig('../img/cloud_height')
# 3
plt.figure(3, figsize = (8,6))
plt.scatter(data[:,4]/1000, data[:,5])
plt.title('Cloud Base height distribution', fontsize=25)
plt.xlabel('cloud base height (km)', fontsize=18)
plt.ylabel('core number', fontsize=18)
plt.xlim(0,)
plt.ylim(0)
plt.savefig('../img/cloud_base')
# 4
plt.figure(4, figsize = (8,6))
plt.scatter(data[:,6], data[:,5])
plt.title('Cloud Precipitation distribution', fontsize=25)
plt.xlabel('precipitation (day)', fontsize=18)
plt.ylabel('core number', fontsize=18)
plt.xlim(0,)
plt.ylim(0)
plt.savefig('../img/cloud_pcp')

plt.show()
