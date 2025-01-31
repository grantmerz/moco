import numpy as np


frac_train=0.01
path = '/home/g4merz/data/sdss.npz'
data = np.load(path, allow_pickle=True)
n_gal = len(data["labels"])
indices = np.random.permutation(n_gal)
ind_split_train = int(np.ceil(frac_train * n_gal))
#ind_split_dev = ind_split_train + int(np.ceil(frac_dev * n_gal))
images = data["cube"][indices[:ind_split_train]]
np.save('/home/g4merz/data/sdss_sample.npy',images)
