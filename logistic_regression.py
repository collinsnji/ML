import numpy as np
import tensorflow as tf
import sonnet as snt

data = np.genfromtxt('data/iris_training.csv', delimiter=',')
test = np.genfromtxt('data/iris_test.csv', delimiter=',')
learning_rate = 0.0001

x = tf.placeholder('float', shape=[None, 4])
y = tf.placeholder('int32', shape=[None])

linear = snt.Linear(3)
H = linear(x)

cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=H)
loss = tf.reduce_mean(cross_entropy)
model = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.global_variables_initializer()

# ignore
def sigmoid(x, w):
    return 1/(1+np.exp(1)**-(w*x))

def softmax(x):
    S = np.exp(np.array(x))
    return S / np.sum(S)

def run():
    session = tf.Session()
    session.run(init)
    xi = data[:, :4]
    yi = data[:, -1]
    training_data = {x: xi, y: yi}
    test_data = {x: test[:, :4], y: test[:, -1]}

    for i in range(1000): # for i in range(len(data)): 
        session.run(model, feed_dict=training_data)
        print 'Loss: %f' % session.run(loss, feed_dict=training_data)

    print 'Iterations: %d' %i
    print 'W:', session.run(linear.w)
    print 'b:', session.run(linear.b)
    print 'Eval: %f' % session.run(loss, feed_dict=test_data)

if __name__ == '__main__':
    run()
