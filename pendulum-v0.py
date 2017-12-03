# coding: utf-8
# imports

import tensorflow as tf
import sonnet as snt
import numpy as np
import gym
import random
from collections import deque

class QNetwork:
    def __init__(self, action_size, state_size):
        self.session = tf.Session()
        self.action_size = action_size
        self.learning_rate = 0.001
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.gamma = 0.95
        self.tau = 0.125
        self.memory = deque(maxlen=2000)
        self.
    
    def actor_model(self):
        hidden_units = 24
        linear = snt.Linear(self.action_size)
        l1 = snt.Linear(hidden_units)
        l2 = tf.nn.relu(l1)
        l3 = tf.nn.relu(l2)
        # TODO: Find out if this is the same as:: output = snt.Linear(self.action_size)(l3)
        output = linear(l3)
