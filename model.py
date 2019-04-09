import tensorflow as tf

class siamese:

    # Create model
    def __init__(self,x1, x2, y, sizes):
        self.margin = 25.00
        self.keep_prob = 1.0 #tf.placeholder(tf.float32, name='dropout_prob')
        self.num_labels = 128
        self.x1 = tf.scalar_mul(0.003922, tf.cast(x1, dtype=tf.float32))#tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        self.x2 = tf.scalar_mul(0.003922, tf.cast(x2, dtype=tf.float32))#tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        self.layers = []
        with tf.variable_scope("siamese") as scope:
            self.o1 = self.network(self.x1,sizes)
            scope.reuse_variables()
            self.o2 = self.network(self.x2,sizes)

        # Create loss
        self.y_ = y#tf.placeholder(tf.bool, [None])
        self.loss = self.custom_loss()
        self.acc = self.acc_summary()

    def network(self, input_layer, sizes):
        #i = 0
        l1_filters = 16
        l2_filters = 32#64
        l3_filters = 32
        l4_filters = 32
        l5_filters = 64
        l6_filters = 128
        input_layer_local = input_layer
        with tf.variable_scope("conv1"):
            self.out_1 = self.conv_layer(input_layer_local, [7,7,3, l1_filters],'layer1', padding='VALID', stride = 1)
        with tf.variable_scope("conv2"):
            self.out_2 = self.conv_layer(self.out_1, [5, 5, l1_filters, l2_filters],'layer2', padding='VALID', stride = 1, pooling=False)
        with tf.variable_scope("conv3"):
            self.out_3 = self.conv_layer(self.out_2, [5, 5, l2_filters, l3_filters], 'layer3', padding='SAME', stride = 1, pooling=False)
        with tf.variable_scope("conv4"):
            self.out_4 = self.conv_layer(self.out_3, [3, 3, l3_filters, l4_filters], 'layer4', padding='VALID',stride = 1, pooling=False)
        with tf.variable_scope("conv5"):
            self.out_5 = self.conv_layer(self.out_4, [3, 3, l4_filters, l5_filters], 'layer5', padding='VALID',stride = 1, pooling=False)
        with tf.variable_scope("conv6"):
            self.out_6 = self.conv_layer(self.out_5, [3, 3, l5_filters, l6_filters], 'layer6', padding='VALID',stride = 1, pooling=False)
        self.final_out = tf.reshape(self.out_6, [-1, l6_filters])
        return self.final_out
    """
        for x in sizes:
            self.layers.append(self.layer_generation(input_layer_local, x, "layer" + str(i)))
            i = i + 1
            input_layer_local = tf.nn.relu(self.layers[-1], name='out_'+str(i))
        return self.layers[-1]
    """
    def layer_generation(self, input, layer_size, name):
        input_len = input.get_shape()[1]
        seed = tf.truncated_normal_initializer(stddev=0.01)
        w = tf.get_variable(name+'_W', dtype=tf.float32, shape=[input_len, layer_size], initializer=seed)
        b = tf.get_variable(name+'_b', dtype=tf.float32, initializer=tf.constant(0.01, shape=[layer_size], dtype=tf.float32) )
        #out = tf.nn.relu(tf.nn.bias_add(tf.matmul(input, w, name=name+'_mul'), b,name=name+'_add'), name=name+'_out')
        out = tf.nn.bias_add(tf.matmul(input, w, name=name + '_mul'), b, name=name + '_add')
        return out

    def conv2d(self, input_layer, W, pad, stride=1):
        return tf.nn.conv2d(input=input_layer,
                            filter=W,
                            strides=[1, stride, stride, 1],
                            padding=pad)

    def create_max_pool_layer(self, input):
        return  tf.nn.max_pool(value=input,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    def activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        return tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def conv_layer(self, input_layer, weights, name, padding, stride=1, pooling=True):
#        with tf.variable_scope(name) as scope:
        kernel = tf.get_variable(name+"_kernel", shape=weights, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.1, dtype=tf.float32))
        conv = self.conv2d(input_layer, kernel, padding, stride)
        init = tf.constant(1., shape=[weights[-1]], dtype=tf.float32)
        bias = tf.get_variable(name+"_bias",  dtype=tf.float32, initializer=init)
        preactivation = tf.nn.bias_add(conv, bias)
        conv_relu = tf.nn.relu(preactivation, name=name)
        self.activation_summary(conv_relu)
        if pooling:
            out = self.create_max_pool_layer(conv_relu)
        else:
            out = conv_relu
        return out

    def custom_loss(self):
        labels_t = tf.to_float(self.y_)
        labels_f = tf.subtract(1.0, labels_t, name="1-yi")
        distance2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        distance2 = tf.reduce_sum(distance2, 1)
        distance = tf.sqrt(distance2 + 1e-6, name="Distance")
        same = tf.multiply(labels_t, distance2)
        margin_tensor = tf.constant(self.margin, dtype=tf.float32, name="Margin")
        diff = tf.multiply(labels_f, tf.pow(tf.maximum(0.0, tf.subtract(margin_tensor, distance)), 2.))
        loss = tf.reduce_mean(same)+tf.reduce_mean(diff)
        return loss

    def acc_summary(self):
        ##return [tf.summary.scalar("same", 9.0 * tf.reduce_mean(same)), tf.summary.scalar("True", tf.reduce_mean(labels_t)),
        ##        tf.summary.scalar("False", tf.reduce_mean(labels_f))]
        ##fcw1 = tf.Graph.get_tensor_by_name(tf.get_default_graph(), name="siamese/local1/fcw_1").read_value()
        #fcw1 = self.W_fc1.read_value()
        labels_t = tf.to_float(self.y_)
        labels_f = tf.subtract(1.0, labels_t, name="1-yi")
        distance2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        distance2 = tf.reduce_sum(distance2, 1)
        distance = tf.sqrt(distance2 + 1e-6, name="Distance")
        same = tf.multiply(labels_t, distance2)
        margin_tensor = tf.constant(self.margin, dtype=tf.float32, name="Margin")
        diff = tf.multiply(labels_f, tf.pow(tf.maximum(0.0, tf.subtract(margin_tensor, distance)), 2.))
        return tf.summary.scalar("loss", tf.reduce_mean(same) + tf.reduce_mean(diff))
        #return [tf.summary.histogram("fcw", fcw1)]

