# coding: utf-8

# import modules
import random
from collections import deque
import gym
import numpy as np
import tensorflow as tf
import sonnet as snt

# Variables
EPISODES = 1000

# DQN
class DQN:
    def __init__(self, state_size, action_size):
        self.session = tf.Session()
        self.state = tf.placeholder(tf.float32, shape=[None, state_size])
        self.target = tf.placeholder(tf.float32, shape=[None, action_size])
        self.action = tf.placeholder(tf.int64, shape=[None])
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.action_values = self.model(self.state)
        self.loss = tf.reduce_mean((self.action_values - self.target)**2)
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
        init = tf.global_variables_initializer()
        self.session.run(init)
    
    def _build_model(self):
        hidden_units = 8
        first_layer = snt.Linear(hidden_units)
        second_layer = snt.Linear(2)
        
        def action_values(state):
            hidden_layers = 8
            input_layer = first_layer(state)
            hidden_layer = tf.nn.relu(input_layer)
            output_layer = second_layer(hidden_layer)
            return output_layer

        return action_values
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
        
    def act(self, state):
        if np.random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        
        act_values = self.session.run(self.action_values, feed_dict={self.state: state})
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_actions_values = self.session.run(self.action_values, {self.state: next_state})
                target = reward + self.gamma * np.amax(next_actions_values[0])
            target_f = self.session.run(self.action_values, {self.state: state})
            #target_f = self.model(state)
            target_f[0][action] = target
            values = self.session.run({'_': self.train_op, 'loss': self.loss, 'action_value': self.action_values}, 
                                      feed_dict={self.state: state, self.target: target_f})
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def main():
    env = gym.make('CartPole-v1')
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQN(state_size, action_size)
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        accumulated_reward = 0
        for time in range(500):
            env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            accumulated_reward += reward
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                     .format(e, EPISODES, accumulated_reward, agent.epsilon))
                break
if __name__ == "__main__":
    main()
