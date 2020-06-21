#encoding=utf-8
import tensorflow as tf
from MLP_Model import MLP_Model
from GMF_Model import GMF_Model

class Neural_Model(object):
    def __init__(self, num_users, num_items, num_factors, layers, lr, learner):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.layers = layers
        self.num_layers = len(layers)
        self.lr = lr
        self.learner = learner

        self.mf_users_embeddings = tf.get_variable("mf_users_embeddings", shape=[self.num_users, self.num_factors], dtype=tf.float32,\
            trainable=True)
        self.mf_items_embeddings = tf.get_variable("mf_items_embeddings", shape=[self.num_items, self.num_factors], dtype=tf.float32,\
            trainable=True)

        self.mlp_users_embeddings = tf.get_variable("mlp_users_embeddings", shape=[self.num_users, self.layers[0] / 2], dtype=tf.float32,\
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), trainable=True)
        self.mlp_items_embeddings = tf.get_variable("mlp_items_embeddings", shape=[self.num_items, self.layers[0] / 2], dtype=tf.float32,\
            initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01), trainable=True)

    def add_placeholder(self):
        self.user_inputs = tf.placeholder(dtype=tf.int32, shape=[None], name='user_inputs')
        self.item_inputs = tf.placeholder(dtype=tf.int32, shape=[None], name='item_inputs')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name='true_label')

    def inference(self):
        with tf.name_scope("MF_part"):
            self.mf_user_latent = tf.nn.embedding_lookup(self.mf_users_embeddings, self.user_inputs, name='mf_user_latent')
            self.mf_item_latent = tf.nn.embedding_lookup(self.mf_items_embeddings, self.item_inputs, name='mf_item_latent')

            self.mf_vector = tf.multiply(self.mf_user_latent, self.mf_item_latent, name='multiply')

        with tf.name_scope("MLP_part"):
            self.mlp_user_latent = tf.nn.embedding_lookup(self.mlp_users_embeddings, self.user_inputs, name='mlp_user_latent')
            self.mlp_item_latent = tf.nn.embedding_lookup(self.mlp_items_embeddings, self.item_inputs, name='mlp_item_latent')
    
            self.mlp_vector = tf.concat([self.mlp_user_latent, self.mlp_item_latent], axis=1, name='concat')

            #MLP layers
            for idx in range(1, self.num_layers):
                self.mlp_vector = tf.layers.dense(self.mlp_vector, units=self.layers[idx], activation=tf.nn.relu, name='layer%d' % idx)

        with tf.name_scope("MF_MLP_final"):
            # Concatenate MF and MLP parts
            predict_vector = tf.concat([self.mf_vector, self.mlp_vector], axis=1, name='MF_MLP_vector')
            # Final prediction layer
            self.prediction = tf.layers.dense(predict_vector, units=1, activation=None, name='final_prediction')
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
