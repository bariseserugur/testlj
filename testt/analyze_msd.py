import numpy as np

f = open('random/3/msd.out','rb')

msd = np.load(f)
mean_msd = np.mean(msd,axis=0)

print(list(mean_msd))
