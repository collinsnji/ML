import numpy as np
import tensorflow as tf
import sonnet as snt

data = np.genfromtxt('data/linearRegression.csv', delimiter=',')
test = np.genfromtxt('data/test.csv', delimiter=',')
learning_rate = 0.0001

x = tf.placeholder('float', shape=[None, 1])
y = tf.placeholder('float', shape=[None, 1])

#W = tf.Variable(tf.ones([1, 1]))
#b = tf.Variable(tf.zeros([1]))
#H = tf.matmul(x, W) + b

linear = snt.Linear(1)
H = linear(x)

loss = tf.reduce_mean(tf.squared_difference(y, H))
model = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

def run():
    session = tf.Session()
    session.run(init)
    xi = data[:, :1]
    yi = data[:, 1:]
    training_data = {x: xi, y: yi}
    test_data = {x: test[:, :1], y: test[:, 1:]}

    for i in range(1000): # for i in range(len(data)): 
        session.run(model, feed_dict=training_data)

    print 'Iterations: %d' %i
    print 'W: %f' % session.run(linear.w)
    print 'b: %f' % session.run(linear.b)
    print 'Loss: %f' % session.run(loss, feed_dict=training_data)
    print 'Eval: %f' % session.run(loss, feed_dict=test_data)

if __name__ == '__main__':
    run()
