import tensorflow as tf
from matplotlib import pyplot as plt
import math
import model
import numpy as np

image_size = 28
batch_size = 1

def _parse_function(example_proto):
    feature = {
        'img_a': tf.FixedLenFeature([], tf.string),
        'img_b': tf.FixedLenFeature([], tf.string),
        'match': tf.FixedLenFeature([], tf.int64)
    }
    features = tf.parse_single_example(example_proto, feature)
    # Convert the image data from string back to the numbers
    image_a = tf.decode_raw(features['img_a'], tf.int64)
    image_b = tf.decode_raw(features['img_b'], tf.int64)
    match = tf.cast(features['match'], tf.int32)

    # Reshape image data into the original shape
    image_a = tf.reshape(image_a, [image_size, image_size, 1])
    image_b = tf.reshape(image_b, [image_size, image_size, 1])

    return image_a, image_b, match

def load_data():
    # prepare data and tf.session
    data_path = 'datasets/gray.tfrecords'
    dataset = tf.data.TFRecordDataset(data_path)
    dataset = dataset.map(_parse_function)  # Parse the record into tensors.
    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.repeat()  # Repeat the input indefinitely.
    dataset = dataset.batch(batch_size)
    return dataset.make_initializable_iterator()
    #[x1, x2, y] = iterator.get_next()

def load_model(sess, saver):
    if tf.train.checkpoint_exists("./model/Final"):
        print("Model exists")
        saver.restore(sess, './model/Final')  # Sloppy and dangerous
        return True
    else:
        print("Model not found")
        return False

def getActivations(sess, layer):
    print("Test")
    input, units = sess.run([x1, layer])
    plotNNFilter(input, units)

def plotNNFilter(input, units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20, 20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i + 1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0, :, :, i], interpolation="nearest", cmap="gray")
    # Show the input too
    plt.subplot(n_rows, n_columns, i + 2)
    plt.title("Original")
    input = input.reshape([1, image_size, image_size, 1])
    plt.imshow(input[0, :, :,0], interpolation="nearest", cmap="gray")
    plt.show()
    print("Fin")
def print_inputs(sess):
    n = 20
    n_columns = 8
    n_rows = 5

    plt.figure(1, figsize=(20, 20))
    for i in range(n):
        [ex1, ex2, match_bool ] = sess.run([x1, x2, match])
        ex1 = ex1.reshape([image_size, image_size])
        plt.subplot(n_rows, n_columns, 2*i+1)
        plt.title('Fig 1 - ' + str(match_bool))
        plt.imshow(ex1, interpolation="nearest", cmap="gray")
        plt.axis('off')
        ex2 = ex2.reshape([image_size, image_size])
        plt.subplot(n_rows, n_columns, 2*i+2)
        plt.title('Fig 2 - ' + str(match_bool))
        plt.imshow(ex2, interpolation="nearest", cmap="gray")
        plt.axis('off')
    plt.show()

def print_output_vecs(sess, net):
    with open('output.csv', 'w') as the_file:
        for i in range(1000):
            [mb, out1, out2] = sess.run([match, net.o1, net.o2])
            if (batch_size > 1):
                out1 = out1.reshape([batch_size, net.num_labels])
                out2 = out2.reshape([batch_size, net.num_labels])
            else:
                out1 = out1.reshape([net.num_labels])
                out2 = out2.reshape([net.num_labels])
            mbs = str(mb)
            mbs = mbs.strip('[')
            mbs = mbs.strip(']')
            str1 = ", ".join(str(x) for x in out1)
            str2 = ", ".join(str(x) for x in out2)
            the_file.write(mbs + ', ' + str1 + ', ' + str2 + '\n')

def output_test(sess, net):
    [mb, out1, out2, score] = sess.run([match, net.o1, net.o2, net.loss])
    out1 = out1.reshape([3, net.num_labels])
    out2 = out2.reshape([3, net.num_labels])
    print("Pause")

def get_distances(sess, net):
    n_bins = 40
    bin_max = 5
    dist_diff=[]
    dist_same=[]
    for i in range(1000):
        [mb, out1, out2] = sess.run([match, net.o1, net.o2])
        out1 = out1.reshape([net.num_labels])
        out2 = out2.reshape([net.num_labels])
        dist = np.linalg.norm(out1 - out2)
        if (mb):
            dist_same.append(min(dist, bin_max-0.001))
        else:
            dist_diff.append(min(dist, bin_max-0.001))
    #print("Same: " + str(dist_same/1000.) + " Diff: " + str(dist_diff/1000.))


    fig, axs = plt.subplots(1, 2, sharey=True, tight_layout=True)
    # We can set the number of bins with the `bins` kwarg
    #axs[0].hist(dist_same, bins=n_bins)
    #axs[1].hist(dist_diff, bins=n_bins)
    #plt.show()
    # Fix the range
    axs[0].hist(dist_same, bins=n_bins, range=[0., bin_max])
    axs[0].set_title("Same Image Dist")
    axs[1].hist(dist_diff, bins=n_bins, range=[0., bin_max])
    axs[1].set_title("Diff Image Dist")
    plt.show()

def assess_activation(sess, net):
    [conv1, conv2, conv3] = sess.run([net.out_1, net.out_2, net.out_3])
    m1 = np.mean(conv1, axis=(0, 1, 2),  dtype=np.float64)
    m2 = np.mean(conv2, axis=(0, 1, 2),  dtype=np.float64)
    m3 = np.mean(conv3, axis=(0, 1, 2),  dtype=np.float64)
    s1 = np.std(conv1, axis=(0, 1, 2),  dtype=np.float64)
    s2 = np.std(conv2, axis=(0, 1, 2),  dtype=np.float64)
    s3 = np.std(conv3, axis=(0, 1, 2),  dtype=np.float64)
    low = m3-s3
    high = m3+s3
    plt.plot(m3)
    plt.plot(low)
    plt.plot(high)
    plt.show()
    print("fin")


# Main body:
iterator = load_data()
[x1, x2, match] = iterator.get_next(name="Iterator")
sess = tf.InteractiveSession()
sess.run(iterator.initializer)


network = model.siamese(x1, x2, match, [1024, 1024, 2])
mod = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
saver = tf.train.Saver(mod, max_to_keep=15)

if load_model(sess, saver):
    print("Loaded")
    #getActivations(sess, network.out_1)
    #print_inputs(sess)
    #print_output_vecs(sess, network)
    #output_test(sess, network)
    get_distances(sess, network)
    #assess_activation(sess, network)