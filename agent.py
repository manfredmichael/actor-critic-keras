import numpy as np

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import tensorflow as tf
tf.compat.v1.disable_eager_execution()

from utils import LearningReport

class Agent:
    def __init__(self, alpha, beta, gamma=0.99, n_actions=4, fc1_size=1024, fc2_size=512,
                 input_dims=8):
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.input_dims = input_dims
        self.fc1_dims = fc1_size
        self.fc2_dims = fc2_size
        self.n_actions = n_actions

        self.report = LearningReport()

        self.actor, self.critic, self.policy = self.build_actor_critic_networks()
        self.action_space = [i for i in range(self.n_actions)]

    def build_actor_critic_networks(self):
        input_ = Input(shape=[self.input_dims])
        delta = Input(shape=[1])
        dense1 = Dense(self.fc1_dims, activation='relu')(input_)
        dense2 = Dense(self.fc2_dims, activation='relu')(dense1)

        probs = Dense(self.n_actions, activation='softmax')(dense2)
        values = Dense(1, activation='linear')(dense2)

        def custom_loss(y_true, y_pred):
            out = K.clip(y_pred, 1e-8, 1-1e-8)
            log_likelihood = y_true * K.log(out)

            return K.sum(-log_likelihood * delta)

        actor = Model(inputs=[input_, delta], outputs=[probs])
        actor.compile(optimizer=Adam(lr=self.alpha), loss=custom_loss)

        critic = Model(inputs=[input_], outputs=[values])
        critic.compile(optimizer=Adam(lr=self.alpha), loss="mse")

        policy = Model(inputs=[input_], outputs=[probs])

        return actor, critic, policy

    def choose_action(self, observation):
        state = np.expand_dims(observation, axis=0)
        probabilities = self.policy.predict(state)[0]
        action = np.random.choice(self.action_space, p=probabilities)

        return action

    def learn(self, state, action, reward, state_, done):
        self.report.add_reward(reward)

        state = np.expand_dims(state, axis=0)
        state_ = np.expand_dims(state_, axis=0)

        critic_value = self.critic.predict(state)
        critic_value_ = self.critic.predict(state_)

        target = reward + self.gamma * critic_value_  * (1-int(done))
        delta = target - critic_value

        actions = np.zeros([1, self.n_actions])
        actions[np.arange(1), action] = 1.0
        
        self.actor.fit([state, delta], actions, verbose=0)
        self.critic.fit(state, target, verbose=0)
