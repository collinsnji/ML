import numpy as np
import tensorflow as tf
import sonnet as snt

data = np.genfromtxt('data/iris_training.csv', delimiter=',')
test = np.genfromtxt('data/iris_test.csv', delimiter=',')
learning_rate = 0.001

x = tf.placeholder('float', shape=[None, 4])
y = tf.placeholder('int64', shape=[None])

num_hidden = 100

linear1 = snt.Linear(num_hidden, name='layer1')
linear2 = snt.Linear(3, name='layer2')
hidden = tf.sigmoid(linear1(x))
outputs = linear2(hidden)


def softmax(x):
    S = tf.exp(x)
    return S / tf.expand_dims(tf.reduce_sum(S, axis=1), axis=-1)

# cross entropy
one_hot = tf.one_hot(y, 3)
probs = softmax(outputs)
CE = tf.reduce_sum(one_hot * tf.log(probs), axis=-1)
loss = -1 * tf.reduce_mean(CE)

# accuracy
prediction_indices = tf.argmax(probs, axis=1)
# import pdb; pdb.set_trace()
batch_correct = tf.equal(prediction_indices, y)
average_accuracy = tf.reduce_mean(tf.cast(batch_correct, tf.float32),
                                  name='accuracy')

# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=H)
# loss = tf.reduce_mean(cross_entropy)

# train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
train_op = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(loss)

init = tf.global_variables_initializer()

def run():
    session = tf.Session()
    session.run(init)
    
    training_data = {x: data[:, :4], y: data[:, -1]}
    test_data = {x: test[:, :4], y: test[:, -1]}

    for i in range(10000):
        train_results = session.run({'_': train_op, 'loss': loss, 'accuracy': average_accuracy},
                                    feed_dict=training_data)
        if i % 100 == 0:
            test_results = session.run({'loss': loss, 'accuracy': average_accuracy},
                                       feed_dict=test_data)
            print('%d: train loss = %f train accuracy = %f test loss = %f test accuracy = %f' %
                  (i, train_results['loss'], 100 * train_results['accuracy'],
                   test_results['loss'], 100 * test_results['accuracy']))                      

    print 'Iterations: %d' %i
    test_results = session.run({'loss': loss, 'accuracy': average_accuracy},
                               feed_dict=test_data)
    print 'test loss: %f, test accuracy: %f%%' % (test_results['loss'],
                                                  test_results['accuracy'] * 100)

if __name__ == '__main__':
    run()
