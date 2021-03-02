# %%
import numpy as np
import matplotlib.pyplot as plt
from utils import *
# %%
filename = "./data/10.npz"
t,features,linear_velocity,angular_velocity,K,b,imu_T_cam = load_data(filename, load_features = True)
# %%

# %%
