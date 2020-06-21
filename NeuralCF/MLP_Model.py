#encoding=utf-8

import tensorflow as tf

class MLP_Model(object):
    def __init__(self, num_users, num_items, lr, learner, layers):
        self.num_users = num_users
        self.num_items = num_items
        #self.num_factors = num_factors
        #self.regs = regs
        self.lr = lr
        self.learner = learner
        self.layers = layers
        self.num_layers = len(layers)
        #self.users_embeddings = tf.keras.layers.Embedding(input_dim=self.num_users, output_dim=self.num_factors,\
        #    input_length=1, name='user_embeddings')
        #self.items_embeddings = tf.keras.layers.Embedding(input_dim=self.num_items, output_dim=self.num_factors,\
        #    input_length=1, name='item_embeddings')
        self.users_embeddings = tf.get_variable("users_embeddings", shape=[self.num_users, self.layers[0] / 2], dtype=tf.float32,\
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), trainable=True)
        self.items_embeddings = tf.get_variable("items_embeddings", shape=[self.num_items, self.layers[0] / 2], dtype=tf.float32,\
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), trainable=True)

    def add_placeholder(self):
        self.user_inputs = tf.placeholder(dtype=tf.int32, shape=[None], name='user_inputs')
        self.item_inputs = tf.placeholder(dtype=tf.int32, shape=[None], name='item_inputs')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name='true_label')
    
    def inference(self):
        self.user_latent = tf.nn.embedding_lookup(self.users_embeddings, self.user_inputs)
        self.item_latent = tf.nn.embedding_lookup(self.items_embeddings, self.item_inputs)
    
        vector = tf.concat([self.user_latent, self.item_latent], axis=1, name='concat')

        #MLP layers
        for idx in range(1, self.num_layers):
            vector = tf.layers.dense(vector, units=self.layers[idx], kernel_initializer=tf.contrib.layers.xavier_initializer(),\
                activation=tf.nn.relu, name='layer%d' % idx)
        #Final prediction
        self.prediction = tf.layers.dense(vector, units=1, activation=None, name='prediction')
        #logits
        self.logits = tf.squeeze(self.prediction, name='logits')

    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))
        
    def train(self):
        if self.learner == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        
        self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())

    def add_performance(self):
        """
        """
        #with tf.name_scope('performance'):
        self.tf_loss_ph = tf.placeholder(tf.float32, shape=None, name='loss_summary')
        self.tf_hr_ph = tf.placeholder(tf.float32, shape=None, name='hr_summary')
        self.tf_ndcg_ph = tf.placeholder(tf.float32, shape=None, name='ndcg_summary')

        tf_loss_summary = tf.summary.scalar('loss', self.tf_loss_ph)
        tf_hr_summary = tf.summary.scalar('hr', self.tf_hr_ph)
        tf_ndcg_summary = tf.summary.scalar('ndcg', self.tf_ndcg_ph)
        
        self.performance_summaries = tf.summary.merge([tf_loss_summary, tf_hr_summary, tf_ndcg_summary])

            
    def build_graph(self):
        self.add_placeholder()
        self.inference()
        self.add_loss()
        self.train()
        self.add_performance()
