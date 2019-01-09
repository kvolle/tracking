from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import system things
import tensorflow as tf
import numpy as np
import random
image_size = 28

#import helpers
import model

def _parse_function(example_proto):
    feature = {
        'img_a': tf.FixedLenFeature([], tf.string),
        'img_b': tf.FixedLenFeature([], tf.string),
        'match': tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_single_example(example_proto, feature)
    # Convert the image data from string back to the numbers
    image_a = tf.decode_raw(features['img_a'], tf.int64, name="Steve")
    image_b = tf.decode_raw(features['img_b'], tf.int64, name="Greg")
    match = tf.cast(features['match'], tf.int32)

    # Reshape image data into the original shape
    image_a = tf.reshape(image_a, [image_size, image_size, 1])
    image_b = tf.reshape(image_b, [image_size, image_size, 1])

    return image_a, image_b, match

# prepare data and tf.session
data_path = 'datasets/gray.tfrecords'
dataset = tf.data.TFRecordDataset(data_path)
dataset = dataset.map(_parse_function)  # Parse the record into tensors.
dataset = dataset.shuffle(buffer_size=1000)
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
[x1, x2, y] = iterator.get_next()

sess = tf.InteractiveSession(config=tf.ConfigProto(log_device_placement=True))
sess.run(iterator.initializer)
#blah1 = sess.run(tf.reduce_mean(x1))
#blah2 = sess.run(x2)
# setup siamese network
network = model.siamese(x1, x2, y, [1024, 1024, 2])
s1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer1')
s2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'siamese.layer2')

mod = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
saver = tf.train.Saver(mod, max_to_keep=15)
tf.initialize_all_variables().run()

"""if tf.train.checkpoint_exists("./model/Final"):
    print("Model exists")
    response = input("Load saved model? (Y/n)")
    if (response == 'Y') or (response == 'y'):
        saver.restore(sess, './model/Final')# Sloppy and dangerous
else:
    print("Model not found")
"""
vars = tf.trainable_variables()

train_step = tf.train.GradientDescentOptimizer(0.0000001).minimize(network.loss,var_list=vars)

writer = tf.summary.FileWriter("log/", sess.graph)

# serialize the graph
graph_def = tf.get_default_graph().as_graph_def()

N = 1#150000
# Create a coordinator and run all QueueRunner objects
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(coord=coord)
for step in range(N):
    """_, loss_v = sess.run([train_step, network.loss], feed_dict={
                        network.x1: batch_x1,
                        network.x2: batch_x2,
                        network.y_: batch_y})"""
    _, loss_v = sess.run([train_step, network.loss])
    if step % 100 == 0:
        print(str(step) + ", " +str(loss_v))
    if np.isnan(loss_v):
        print('Model diverged with loss = NaN')
        quit()
    #if step % 10 == 0:
    #    [loss_sum] = sess.run([network.acc], feed_dict={
    #        network.x1: batch_x1,
    #        network.x2: batch_x2,
    #        network.y_: batch_y})
    #    writer.add_summary(loss_sum, step)
    if step == 1000:
        train_step = tf.train.GradientDescentOptimizer(0.000005).minimize(network.loss, var_list=vars)
writer.close()
saver.save(sess, 'model/Final')