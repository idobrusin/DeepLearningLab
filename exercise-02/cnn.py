import csv
import gzip
import os
import pickle

import argparse
import tensorflow as tf
import tempfile
import numpy as np
import time


def arg_parser():
    parser = argparse.ArgumentParser("Neural network for classifying the MNIST data set")
    parser.add_argument('-l', '--learningrate',
                        dest='learning_rate',
                        type=float,
                        default=0.001,
                        help='Learning rate for the optimizer')
    parser.add_argument('-c', '--cpu',
                        dest='run_cpu',
                        action='store_true',
                        help='If set, generated output will be appended with CPU instead of GPU. '
                             'Use LD_LIBRARY_PATH = "" environment variable to control running on CPU/GPU')
    parser.add_argument('-f', '--filternumber',
                        dest='num_filters',
                        type=int,
                        default=16,
                        help='Learning rate for the optimizer')
    parser.add_argument('-e', '--epochs',
                        dest='num_epochs',
                        type=int,
                        default=500,
                        help='Number of training epochs')
    parser.add_argument('-ld', '--logdir',
                        dest='log_dir',
                        type=str,
                        default=None,
                        help='TensorFlow log directory')
    parser.add_argument('-dd', '--datadir',
                        dest='data_dir',
                        type=str,
                        default=None,
                        help='TensorFlow log directory')
    parser.add_argument('-b', '--batchsize',
                        dest='batch_size',
                        type=int,
                        default=100,
                        help='Number of batches for stochastic gradient descent optimization')
    parser.add_argument('-s', '--silent',
                        dest='silent_mode',
                        action='store_true',
                        help='If set, all outputs are suppressed, except the final accuracy')
    return parser


def mnist(datasets_dir='./data'):
    """
    Loads the MNIST data set
    The data set is stored in the 'datasets_dir', if not yet present
    :param datasets_dir: location for storing the data set
    :return: [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    """
    if not os.path.exists(datasets_dir):
        os.mkdir(datasets_dir)
    data_file = os.path.join(datasets_dir, 'mnist.pkl.gz')
    if not os.path.exists(data_file):
        print('... downloading MNIST from the web')
        try:
            import urllib
            urllib.urlretrieve('http://google.com')
        except AttributeError:
            import urllib.request as urllib
        url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'
        urllib.urlretrieve(url, data_file)

    print('... loading data')
    # Load the dataset
    f = gzip.open(data_file, 'rb')
    try:
        train_set, valid_set, test_set = pickle.load(f, encoding="latin1")
    except TypeError:
        train_set, valid_set, test_set = pickle.load(f)
    f.close()

    test_x, test_y = test_set
    test_x = test_x.astype('float32')
    test_x = test_x.astype('float32').reshape(test_x.shape[0], 1, 28, 28)
    test_y = test_y.astype('int64')
    valid_x, valid_y = valid_set
    valid_x = valid_x.astype('float32')
    valid_x = valid_x.astype('float32').reshape(valid_x.shape[0], 1, 28, 28)
    valid_y = valid_y.astype('int64')
    train_x, train_y = train_set
    train_x = train_x.astype('float32').reshape(train_x.shape[0], 1, 28, 28)
    train_y = train_y.astype('int64')

    train_x = train_x.reshape((train_x.shape[0], -1))
    train_y = one_hot(train_y)
    valid_x = valid_x.reshape((valid_x.shape[0], -1))
    valid_y = one_hot(valid_y)
    test_x = test_x.reshape((test_x.shape[0], -1))
    test_y = one_hot(test_y)
    rval = [(train_x, train_y), (valid_x, valid_y), (test_x, test_y)]
    print('... done loading data')
    print(train_x.shape)
    print(valid_x.shape)
    print(test_x.shape)
    return rval


def save_data(data_dir, num_filters, learning_rate, test_accuracy, run_on_cpu, num_variables, num_epochs):
    """
    Saves data to data_dir
    :param data_dir: directory to store data in
    :param num_filters: number of filter
    :param learning_rate: learning rate
    :param test_accuracy: test accuracy
    :param run_on_cpu: true if training was on cpu
    :param num_variables: number of parameters in network
    :param num_epochs: number of training epochs
    """
    run_mode = "GPU" if not run_on_cpu else "CPU"

    # ---- data
    file_name = "%s/data-learnrate_%s-num-filters_%s-run-on_%s.txt" % \
                (data_dir, learning_rate, num_filters, run_mode)
    values = np.array(train_accuracies)
    np.savetxt(file_name, values, fmt='%1.5f')

    # ---- durations
    duration_file_name = "%s/durations.csv" % data_dir

    fields = [num_variables, "%.2f" % duration, run_mode, num_epochs, learning_rate]
    with open(duration_file_name, 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(fields)

    accuracy_file_name = "%s/accuracies.csv" % data_dir
    fields = [num_variables, "%.5f" % learning_rate, "%.5f" % test_accuracy, run_mode, num_filters, num_epochs]
    with open(accuracy_file_name, 'a') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(fields)


def one_hot(labels):
    """this creates a one hot encoding from a flat vector:
    i.e. given y = [0,2,1]
     it creates y_one_hot = [[1,0,0], [0,0,1], [0,1,0]]
    """
    classes = np.unique(labels)
    n_classes = classes.size
    one_hot_labels = np.zeros(labels.shape + (n_classes,))
    for c in classes:
        one_hot_labels[labels == c, c] = 1
    return one_hot_labels


def neural_net(num_filters=16, filter_size=3):
    # Input / Output
    # x = tf.placeholder(tf.float32, [None, 784])
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])

    with tf.name_scope('reshape'):
        x_img = tf.reshape(x, [-1, 28, 28, 1])
        # tf.summary.image('input', x_img, 10)  # store image examples (increases log size!)

    with tf.name_scope('conv1'):
        # Output shape: [28, 28, 16] (with default num_filters)
        conv1 = tf.layers.conv2d(
            inputs=x_img,
            filters=num_filters,
            kernel_size=filter_size,
            padding="same",
            activation=tf.nn.relu)

    with tf.name_scope('maxpool1'):
        # Output shape: [14, 14, 16]
        pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2, 2], strides=2)

    with tf.name_scope('conv2'):
        # Output shape: [14, 14, 16] (with default num_filters)
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=num_filters,
            kernel_size=filter_size,
            padding="same",
            activation=tf.nn.relu)

    with tf.name_scope('maxpool2'):
        # Output shape: [7, 7, 16]
        pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2, 2], strides=2)

    with tf.name_scope('fullCon'):
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * num_filters])
        dense = tf.layers.dense(inputs=pool2_flat, units=128, activation=tf.nn.relu)

    # Logits Layer
    with tf.name_scope('output'):
        output = tf.layers.dense(inputs=dense, units=10)
    return output, x, y_


if __name__ == '__main__':
    parser = arg_parser()
    args = parser.parse_args()

    # ----- Configuration -----------
    # Training params
    num_train_epochs = args.num_epochs
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    num_filters = args.num_filters
    filter_size = 3

    # Calculation params
    run_on_cpu = args.run_cpu

    # Output
    if not args.log_dir:
        log_dir = tempfile.mkdtemp()
    else:
        log_dir = args.log_dir

    data_dir = args.data_dir
    silent_mode = args.silent_mode

    print("Configuration")
    print("   Number of Epochs: ", num_train_epochs)
    print("   Batch size      : ", batch_size)
    print("   Learning rate   : ", learning_rate)
    print("   Number of filter: ", num_filters)
    print("   Filter size     : ", filter_size)
    print("   Run on CPU      : ", run_on_cpu)
    print("   TF Log directory: ", log_dir)
    print("   Data directory  : ", data_dir)
    print("   Silent Mode     : ", silent_mode)

    # ----- Load MNIST data --------------
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = mnist()

    # ------ Neural Network ----------
    # Create NN graph
    output, x, y_ = neural_net(num_filters=num_filters)

    # Optimization: cross entropy loss with stochastic gradient descent
    with tf.name_scope('crossEntropy'):
        cross_entropy = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=output))
        train_step = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # Accuracy
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # create a summary for our cost and accuracy
    tf.summary.scalar("cost", cross_entropy)
    tf.summary.scalar("accuracy", accuracy)
    summary_op = tf.summary.merge_all()

    # ----------- Train network -----------
    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())
        run_name = "learnrate_%s-num-filters_%s-run-on-cpu_%s" % (learning_rate, num_filters, run_on_cpu)
        writer = tf.summary.FileWriter("%s/%s" % (log_dir, run_name), sess.graph)

        t0 = time.time()

        # perform training cycles
        train_accuracies = []
        for epoch in range(num_train_epochs):
            n_batches = int(len(X_train) / batch_size)  # number of batches in one epoch

            for i in range(n_batches):
                batch_begin = i * batch_size
                batch_end = batch_begin + batch_size
                X_batch = X_train[batch_begin:batch_end]
                Y_batch = y_train[batch_begin:batch_end]
                _, summary = sess.run([train_step, summary_op], feed_dict={x: X_batch, y_: Y_batch})
            _, summary = sess.run([accuracy, summary_op], feed_dict={x: X_valid, y_: y_valid})
            writer.add_summary(summary, epoch)
            train_accuracies.append(accuracy.eval(feed_dict={x: X_valid, y_: y_valid}))
            if not silent_mode:
                print('step %d, training accuracy %g' % (epoch, train_accuracies[-1]))
        t1 = time.time()
        duration = t1 - t0
        print('Duration: {:.1f}s'.format(duration))

        test_accuracy = accuracy.eval(feed_dict={x: X_test, y_: y_test})
        print("Accuracy: ", test_accuracy)

        num_variables = np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()])

        # Export data to file for plotting
        if data_dir and not silent_mode:
            save_data(data_dir, num_filters, learning_rate, test_accuracy, run_on_cpu, num_variables, num_train_epochs)
