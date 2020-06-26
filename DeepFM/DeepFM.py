#encoding=utf-8
import tensorflow as tf
#from tf.contrib.layers import xavier_initializer
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm


class DeepFM(object):
    def __init__(self, feature_size, field_size,
                 embedding_size=8, dropout_fm=[1.0, 1.0],
                 deep_layers=[32, 32], dropout_deep=[0.5, 0.5, 0.5],
                 deep_layers_activation=tf.nn.relu,
                 epoch=10, batch_size=256,
                 learning_rate=0.001, optimizer_type="adam",
                 batch_norm=0, batch_norm_decay=0.995,
                 verbose=False, random_seed=2016,
                 use_fm=True, use_deep=True,
                 loss_type="logloss", eval_metric=roc_auc_score,
                 l2_reg=0.0, greater_is_better=True):

        self.feature_size = feature_size        # denote as M, size of the feature dictionary
        self.field_size = field_size            # denote as F, size of the feature fields
        self.embedding_size = embedding_size    # denote as K, size of the feature embedding

        self.dropout_fm = dropout_fm
        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.deep_layers_activation = deep_layers_activation
        self.use_fm = use_fm
        self.use_deep = use_deep
        self.l2_reg = l2_reg

        self.epoch = epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer_type = optimizer_type

        self.batch_norm = batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.verbose = verbose
        self.random_seed = random_seed
        self.loss_type = loss_type
        self.eval_metric = eval_metric
        self.greater_is_better = greater_is_better
        self.train_result, self.valid_result = [], []

        self._init_graph()

    def add_placeholder(self):
        self.feat_index = tf.placeholder(tf.int32, shape=[None, None], name='feat_index')
        self.feat_value = tf.placeholder(tf.float32, shape=[None, None], name='feat_value')
        self.label = tf.placeholder(tf.float32, shape=[None, 1], name='label')
        self.dropout_keep_fm = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_fm")
        self.dropout_keep_deep = tf.placeholder(tf.float32, shape=[None], name="dropout_keep_deep")
        self.train_phase = tf.placeholder(tf.bool, name="train_phase")

    def inference(self):
        self.weights = dict()
        # M * K
        self.weights['feature_embeddings'] = tf.get_variable('feature_embeddings', shape=[self.feature_size, self.embedding_size],\
            dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), trainable=True)
        # M * 1
        self.weights['feature_bias'] = tf.get_variable('feature_bias', shape=[self.feature_size, 1],\
            dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=1.0), trainable=True)

        self.embeddings = tf.nn.embedding_lookup(self.weights["feature_embeddings"],\
                                                             self.feat_index)  # None * F * K

        feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, 1]) #None * F * K

        self.embeddings = tf.multiply(self.embeddings, feat_value) #None * F * K
        
        self.y_first_order = tf.nn.embedding_lookup(self.weights["feature_bias"], self.feat_index) # None * F * 1
        self.y_first_order = tf.reduce_sum(tf.multiply(self.y_first_order, feat_value), 2)  # None * F
        self.y_first_order = tf.nn.dropout(self.y_first_order, self.dropout_keep_fm[0]) # None * F

        # sum_square part
        self.vx_square = tf.reduce_sum(self.embeddings, axis=1)
        self.vx_square = tf.square(self.vx_square)
        # square_sum part
        self.v_square_x_square = tf.square(self.embeddings)
        self.v_square_x_square = tf.reduce_sum(self.v_square_x_square, axis=1)

        # second order, FM计算公式
        self.y_second_order = 0.5 * tf.subtract(self.vx_square, self.v_square_x_square)
        self.y_second_order = tf.nn.dropout(self.y_second_order, self.dropout_keep_fm[1])  # None * K

        # ---------- Deep component ----------
        #init deep weight
        num_layer = len(self.deep_layers)
        input_size = self.field_size * self.embedding_size
        self.weights['layer_0'] = tf.get_variable('layer_0', shape=[input_size, self.deep_layers[0]], dtype=tf.float32,\
            initializer=tf.contrib.layers.xavier_initializer())
        self.weights['bias_0'] = tf.get_variable('bias_0', shape=[self.deep_layers[0]], dtype=tf.float32,\
            initializer=tf.contrib.layers.xavier_initializer())
        for i in range(1, num_layer):
            self.weights['layer_%d' % i] = tf.get_variable('layer_%d' % i, shape=[self.deep_layers[i - 1], self.deep_layers[i]],\
                dtype=tf.float32, initializer=tf.contrib.layers.xavier_initializer())
            self.weights['bias_%d' % i] = tf.get_variable('bias_%d' % i, shape=[1, self.deep_layers[i]], dtype=tf.float32,\
                initializer=tf.contrib.layers.xavier_initializer())

        #deep model
        self.y_deep = tf.reshape(self.embeddings, shape=[-1, self.field_size * self.embedding_size]) # None * (F*K)
        self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[0])
        for i in range(0, len(self.deep_layers)):
            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights["layer_%d" % i]), self.weights["bias_%d" % i]) # None * layer[i] * 1
            if self.batch_norm:
                self.y_deep = self.batch_norm_layer(self.y_deep, train_phase=self.train_phase, scope_bn="bn_%d" %i) # None * layer[i] * 1
            self.y_deep = self.deep_layers_activation(self.y_deep)
            self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_keep_deep[1+i]) # dropout at each Deep layer
        
        # final concat projection layer
        if self.use_fm and self.use_deep:
            input_size = self.field_size + self.embedding_size + self.deep_layers[-1]
            concat_input = tf.concat([self.y_first_order, self.y_second_order, self.y_deep], axis=1)

        self.weights['concat_projection'] = tf.get_variable('concat_projection', shape=[input_size, 1], dtype=tf.float32,\
            initializer=tf.contrib.layers.xavier_initializer())
        self.weights['concat_bias'] = tf.get_variable('concat_bias', shape=[1], initializer=tf.constant_initializer(0.01))
        
        self.out = tf.add(tf.matmul(concat_input, self.weights["concat_projection"]), self.weights["concat_bias"])

    def add_loss(self):
        self.out = tf.nn.sigmoid(self.out)
        self.loss = tf.losses.log_loss(self.label, self.out)
        if self.l2_reg > 0:
            self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["concat_projection"])
            if self.use_deep:
                for i in range(len(self.deep_layers)):
                    self.loss += tf.contrib.layers.l2_regularizer(self.l2_reg)(self.weights["layer_%d"%i])

    def train(self):
        # optimizer
        if self.optimizer_type == "adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.9, beta2=0.999,
                                                    epsilon=1e-8).minimize(self.loss)

    def _init_graph(self):
        self.add_placeholder()
        self.inference()
        self.add_loss()
        self.train()

    def batch_norm_layer(self, x, train_phase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_inference = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                                  is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
        return z
