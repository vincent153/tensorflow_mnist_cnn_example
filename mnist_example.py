import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import sys
from pathlib import Path
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
model_path = Path('model')
if not model_path.is_dir():
    model_path.mkdir()

def compute_accuracy(v_xs, v_ys):
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1), tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys, keep_prob: 1})
    return result

def weight_variable(shape):
    inital = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(inital)

def bias_variable(shape):
    inital = tf.constant(0.1, shape=shape)
    return tf.Variable(inital)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME',use_cudnn_on_gpu=True)

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def build_conv_layer(prev_layer,filter_shape,filter_num):
    
    W_conv = weight_variable([filter_shape[0], filter_shape[1], prev_layer.shape[3].value, filter_num]) 
    b_conv = bias_variable([filter_num])
    h_conv = tf.nn.relu(conv2d(prev_layer, W_conv) + b_conv)
    return max_pool_2x2(h_conv)
    pass

def build_fc_layer(prev_layer,out_neurons,drop_rate):
    
    fc_len = prev_layer.shape[1]*prev_layer.shape[2]*prev_layer.shape[3]
    W_fc = weight_variable([fc_len.value, out_neurons])
    b_fc = bias_variable([out_neurons])
    prev_flat = tf.reshape(prev_layer, [-1, fc_len.value]) #[n_samples, 7,7,64]  => [n_samples, 7*7*64]
    h_fc = tf.nn.relu(tf.matmul(prev_flat, W_fc) + b_fc)
    return tf.nn.dropout(h_fc, drop_rate)
    pass

# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784]) # 28x28

ys = tf.placeholder(tf.float32, [None, 10])

keep_prob = tf.placeholder(tf.float32)#drop out rate
x_image = tf.reshape(xs, [-1, 28, 28, 1])

conv1 = build_conv_layer(x_image,[5,5],32)
conv2 = build_conv_layer(conv1,[5,5],64)
conv3 = build_conv_layer(conv2,[1,1],48)
fc1 = build_fc_layer(conv3,1024,keep_prob)
## output layer ##


classes = 10
W_out = weight_variable([1024, classes])
b_out = bias_variable([classes])

logits = tf.matmul(fc1, W_out) + b_out

prediction = tf.nn.softmax(logits)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=ys)
train_step = tf.train.AdamOptimizer(1e-4).minimize(tf.reduce_mean(cross_entropy))

saver = tf.train.Saver()

with tf.Session(config=None) as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys, keep_prob:0.5})
        if i % 50 == 0:
            acc = compute_accuracy(mnist.test.images[:3000], mnist.test.labels[:3000])
            print('epoch:{},acc:{}'.format(i,acc))
        saver.save(sess, "./model/mnist_cnn3.ckpt")
