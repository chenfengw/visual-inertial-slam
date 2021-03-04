import numpy as np

class KalmanFilter:
    def __init__(self):
        pass
            
    @staticmethod
    def soft_max(x):
        temp = np.exp(x)
        return temp / temp.sum()