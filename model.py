import tensorflow as tf

class siamese:

    # Create model
    def __init__(self,x1, x2, y, sizes):
        self.margin = 25.0
        self.keep_prob = 0.5 #tf.placeholder(tf.float32, name='dropout_prob')
        self.num_labels = 4
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
        l1_filters = 16#32
        l2_filters = 32#64
        fc1 = 256
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
        self.out_1 = self.conv_layer(input_layer_local, [5,5,1,l1_filters],'layer1')
        out_2 = self.conv_layer(self.out_1, [5, 5, l1_filters, l2_filters],'layer2')
        with tf.variable_scope('local1') as scope:
            reshape = tf.reshape(out_2, [-1, 7 * 7 * l2_filters])
            #W_fc1 = self._create_weights([7 * 7 * 64, 1024])
            #b_fc1 = self._create_bias([1024])

            self.W_fc1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * l2_filters, fc1], stddev=0.1, dtype=tf.float32), name="fcw_1")
            self.b_fc1 = tf.Variable(tf.constant(1., shape=[fc1], dtype=tf.float32))
            local1 = tf.nn.relu(tf.matmul(reshape, self.W_fc1) + self.b_fc1, name=scope.name)
            #self._activation_summary(local1)

        with tf.variable_scope('local2_linear') as scope:
            W_fc2 = tf.Variable(tf.truncated_normal(shape=[fc1, self.num_labels], stddev=0.1, dtype=tf.float32))
            b_fc2 = tf.Variable(tf.constant(1., shape=[self.num_labels], dtype=tf.float32))
            local1_drop = tf.nn.dropout(local1, self.keep_prob)
            local2 = tf.nn.bias_add(tf.matmul(local1_drop, W_fc2), b_fc2, name=scope.name)
            #self._activation_summary(local2)
        return local2
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

    def conv2d(self, input_layer, W):
        return tf.nn.conv2d(input=input_layer,
                            filter=W,
                            strides=[1, 1, 1, 1],
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

    def conv_layer(self, input_layer, weights, name):
        with tf.variable_scope(name) as scope:
            kernel = tf.Variable(tf.truncated_normal(shape=weights, stddev=0.1, dtype=tf.float32))
            conv = self.conv2d(input_layer, kernel)
            bias = tf.Variable(tf.constant(1., shape=[weights[-1]], dtype=tf.float32))
            preactivation = tf.nn.bias_add(conv, bias)
            conv_relu = tf.nn.relu(preactivation, name=scope.name)
            self.activation_summary(conv_relu)
            h_pool = self.create_max_pool_layer(conv_relu)
        return h_pool

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
