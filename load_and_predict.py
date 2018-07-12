import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import cv2
#mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

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

conv1 = build_conv_layer(x_image[:5000],[5,5],32)
conv2 = build_conv_layer(conv1,[5,5],64)
conv3 = build_conv_layer(conv2,[1,1],48)
fc1 = build_fc_layer(conv3,1024,keep_prob)


## output layer ##
classes = 10
W_out = weight_variable([1024, classes])
b_out = bias_variable([classes])
prediction = tf.nn.softmax(tf.matmul(fc1, W_out) + b_out)


#img_tf = tf.Variable(img)

saver = tf.train.Saver()
def classify(img,inverse=False):
    print(img.shape)
    img = cv2.resize(img,(28,28),interpolation=cv2.INTER_CUBIC)
    if inverse:
        img = 255-img
    img = img/255
    #cv2.imshow('img',img)
    #cv2.waitKey()
    img = img.reshape([-1,img.shape[0],img.shape[1],1])
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, 'model/mnist_cnn3.ckpt')#load model
        res = sess.run(prediction, feed_dict={x_image: img,keep_prob:1})
        internal_out = sess.run(conv1, feed_dict={x_image: img,keep_prob:1})
        #print(internal_out[0,:,:,0].shape)
           
             
        for i in range(internal_out.shape[3]):
            img = internal_out[0,:,:,i]
            cv2.imwrite('{}.png'.format(i),img)
        
        digit = sess.run(tf.argmax(res,1))
        print(digit)#print predict
        return digit[0]
if __name__ =='__main__':
    import sys

    print(classify(cv2.imread(sys.argv[1],0),int(sys.argv[2])))
    #classify('tmp.png')

