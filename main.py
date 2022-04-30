import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, Dense, Input
from tensorflow.models import Model
from tensorflow.keras.optimizers import Adam

class Agent:
    def __init__(self, alpha, beta, gamma=0.99, n_actions=4, fc1_size=1024, fc2_size=512,
                 input_dims=8):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_size
        self.fc2_dims = fc2_size

