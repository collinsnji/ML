import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

data = np.genfromtxt('data/linearRegression.csv', delimiter=',')
learning_rate = 0.0001

x = tf.placeholder('float')
y = tf.placeholder('float')

W = tf.Variable(tf.ones([1, 1]))
b = tf.Variable(tf.zeros([1]))
H = tf.matmul(x, W) + b

loss = tf.reduce_mean(tf.pow((y - H), 2))
model = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

def run():
    session = tf.Session()
    session.run(init)
    xi = data[:, :1]
    yi = data[:, 1:]
    training_data = {x: xi, y: yi}

    for i in range(1000): # for i in range(len(data)): 
        session.run(model, feed_dict=training_data)

    print 'Iterations: %d' %i
    print 'W: %f' % session.run(W)
    print 'b: %f' % session.run(b)
    print 'Loss: %f' % session.run(loss, feed_dict=training_data)

    plt.plot(xi, yi, 'ro')
    x_ = np.linspace(0, 100, 200)
    plt.plot(x_, session.run(W)[0, 0] * x_ + session.run(b)[0])
    plt.show()
if __name__ == '__main__':
    run()
