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
        fc1 = 128
        fc2 = 64
        fc3 = 64
        """
        mean_tensor = tf.constant(0., dtype=tf.float64)
        variance_tensor = tf.constant(1., dtype=tf.float64)
        normalized = tf.nn.batch_normalization(input_layer,mean=mean_tensor, variance=variance_tensor, offset=None, scale=None, variance_epsilon=0.0000001)
        """
        #batch_mean1, batch_var1 = tf.nn.moments(input_layer, [0])
        #normalized = tf.nn.batch_normalization(input_layer, mean=batch_mean1, variance=batch_var1,
        #                                       offset=None,
        #                                       scale=None, variance_epsilon=0.0000001)
        #input_layer_local = normalized
        input_layer_local = input_layer
        self.out_1 = self.conv_layer(input_layer_local, [7,7,1, l1_filters],'layer1', stride = 1)
        self.out_2 = self.conv_layer(self.out_1, [5, 5, l1_filters, l2_filters],'layer2', stride = 1, pooling=False)
        self.out_3 = self.conv_layer(self.out_2, [5, 5, l2_filters, l3_filters], 'layer3', stride = 1, pooling=True)
        reshape = tf.reshape(self.out_3, [-1, 7 * 7 * l3_filters])
        self.out_4 = self.layer_generation(reshape, fc1, "fc1")
        self.out_5 = self.layer_generation(self.out_4, fc2, "fc2")
        self.out_6 = self.layer_generation(self.out_5, fc3, "fc3")
        self.final_out = self.layer_generation(self.out_6, self.num_labels, "output")
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

    def conv2d(self, input_layer, W, stride=1):
        return tf.nn.conv2d(input=input_layer,
                            filter=W,
                            strides=[1, stride, stride, 1],
                            padding='SAME')

    def create_max_pool_layer(self, input):
        return  tf.nn.max_pool(value=input,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')
    def activation_summary(self, x):
        tensor_name = x.op.name
        tf.summary.histogram(tensor_name + '/activations', x)
        return tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

    def conv_layer(self, input_layer, weights, name, stride=1, pooling=True):
        with tf.variable_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal(shape=weights, stddev=0.1, dtype=tf.float32))
            conv = self.conv2d(input_layer, kernel, stride)
            bias = tf.Variable(tf.constant(1., shape=[weights[-1]], dtype=tf.float32))
            preactivation = tf.nn.bias_add(conv, bias)
            conv_relu = tf.nn.relu(preactivation, name=scope.name)
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
"""
    def custom_loss(self):
        margin = 5.0
        labels_t = tf.to_float(self.y_)
        labels_f = tf.subtract(1.0, labels_t, name="1-yi")          # labels_ = !labels;
        distance2 = tf.pow(tf.subtract(self.o1, self.o2), 2)
        distance2 = tf.reduce_sum(distance2, 1)
        distance = tf.sqrt(distance2+1e-6, name="Distance")
        same_class_losses = tf.multiply(labels_t, distance2)
        margin_tensor = tf.constant(margin,dtype=tf.float32, name="Margin")
        diff_class_losses = tf.multiply(labels_f, tf.pow(tf.maximum(0.0, tf.subtract(margin_tensor, distance)), 2.))
        #losses = tf.add(same_class_losses, diff_class_losses)
        #loss = tf.reduce_sum(losses, name="loss")#losses, name="loss")
        loss = tf.add(tf.reduce_mean(same_class_losses), tf.reduce_mean(diff_class_losses))
        return loss
"""
