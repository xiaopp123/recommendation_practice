#encoding=utf-8
import tensorflow as tf

class GMF_Model(object):
    def __init__(self, num_users, num_items, num_factors, regs, lr, learner):
        self.num_users = num_users
        self.num_items = num_items
        self.num_factors = num_factors
        self.regs = regs
        self.lr = lr
        self.learner = learner
        #self.users_embeddings = tf.keras.layers.Embedding(input_dim=self.num_users, output_dim=self.num_factors,\
        #    input_length=1, name='user_embeddings')
        #self.items_embeddings = tf.keras.layers.Embedding(input_dim=self.num_items, output_dim=self.num_factors,\
        #    input_length=1, name='item_embeddings')
        self.users_embeddings = tf.get_variable("users_embeddings", shape=[self.num_users, self.num_factors], dtype=tf.float32,\
            initializer=tf.random_uniform(), trainable=True)
        self.items_embeddings = tf.get_variable("items_embeddings", shape=[self.num_items, self.num_factors], dtype=tf.float32,\
            initializer=tf.random_uniform(), trainable=True)


    def add_placeholder(self):
        self.user_inputs = tf.placeholder(dtype=tf.int32, shape=[None], name='user_inputs')
        self.item_inputs = tf.placeholder(dtype=tf.int32, shape=[None], name='item_inputs')
        self.y = tf.placeholder(dtype=tf.float32, shape=[None], name='true_label')

    def inference(self):
        #self.user_latent = self.users_embeddings(self.user_inputs)
        #self.item_latent = self.items_embeddings(self.item_inputs)
        self.user_latent = tf.nn.embedding_lookup(self.users_embeddings, self.user_inputs)
        self.item_latent = tf.nn.embedding_lookup(self.items_embeddings, self.item_inputs)

        predict_vector = tf.multiply(self.user_latent, self.item_latent, name='multiply')
        
        # Final prediction layer
        self.prediction = tf.layers.dense(inputs=predict_vector, units=1, activation=tf.nn.relu, name='prediction')
        # 去掉没用的维度
        self.logits = tf.squeeze(self.prediction, name='logits')

    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.y))
    
    def train(self):
        if self.learner == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_global_step())


    def build_graph(self):
        self.add_placeholder()
        self.inference()
        self.add_loss()
        self.train()
