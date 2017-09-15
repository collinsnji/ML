from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import sonnet as snt
# import pdb; pdb.set_trace()

def softmax(x):
    S = tf.exp(x)
    return S / tf.reduce_sum(S, keep_dims=True)

def main():
    data = input_data.read_data_sets('data/MNIST_data', one_hot=False)
    learning_rate = 0.3

    x = tf.placeholder(tf.float32, shape=[None, 784], name='x_values')
    y = tf.placeholder(tf.int64, shape=[None], name='y_values')

    num_hidden = 800
    layer1 = snt.Linear(num_hidden, name='linear1')(x)
    hidden1 = tf.nn.relu(layer1)
    hidden2 = tf.nn.relu(hidden1)
    outputs = snt.Linear(10)(hidden2)

    # Loss
    one_hot = tf.one_hot(y, 10)
    probs = softmax(outputs)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=one_hot, logits=outputs))

    # train
    train_op = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(loss)
    init = tf.global_variables_initializer()

    # accuracy
    prediction_indices = tf.argmax(probs, axis=1)
    batch_correct = tf.equal(prediction_indices, y)
    average_accuracy = tf.reduce_mean(tf.cast(batch_correct, tf.float32), name='accuracy')

    session = tf.Session()
    session.run(init)

    for i in range(5000):
        _x, _y = data.train.next_batch(100)
        training_data = {x: _x, y: _y}

        train_results = session.run({'_': train_op, 'loss': loss, 'accuracy': average_accuracy}, feed_dict=training_data)
        if i % 100 == 0:
            test_data = {x: data.test.images, y: data.test.labels}
            test_results = session.run({'loss': loss, 'accuracy': average_accuracy}, feed_dict=test_data)
            print('%d: train loss = %f train accuracy = %f test loss = %f test accuracy = %f' % (i, train_results['loss'], 100 * train_results['accuracy'], test_results['loss'], 100 * test_results['accuracy']))                      

    print 'Test loss: %f, Test accuracy: %f%%' % (test_results['loss'], test_results['accuracy'] * 100)

if __name__ == '__main__':
    main()