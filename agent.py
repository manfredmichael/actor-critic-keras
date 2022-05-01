import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input
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

        self.actor, self.critic, self.policy = self.build_actor_critic_networks()
        self.action_space = [i for i in range(self.n_actions)]

    def build_actor_critic_networks(self):
        input_ = Input(shape=[self.input_dims])
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input_)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)

        probs = Dense(self.n_actions, activation='softmax')
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_likelihood = y_true* K.log(out)

            return K.sum(-log_likelihood * delta)

        actor = Model(inputs=[input_, delta], ouputs=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)
