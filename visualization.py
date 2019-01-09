import tensorflow as tf
from matplotlib import pyplot as plt
import math
import model

image_size = 28

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
    dataset = dataset.batch(1)
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
    getActivations(sess, network.out_1)
