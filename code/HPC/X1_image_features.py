import numpy as np

# concatenation using HPC
np_fake = np.load("fake_imgs.npy")
np_real = np.load("real_imgs.npy")

X1_images = np.concatenate((np_fake, np_real), axis = 0)
print(X1_images.shape)
np.save("X1_images.npy", X1_images)