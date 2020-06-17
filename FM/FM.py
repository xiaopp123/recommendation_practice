#encoding=utf-8
import os
import sys
import logging
import numpy as np
import argparse
import pickle
import tensorflow as tf
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',level=logging.INFO)

tf.compat.v1.disable_eager_execution()

class FM(object):
    def __init__(self, config):
        """
        :param config: configuration of hyperparameters
        type of dict
        """
        # number of latent factors
        #self.k = config['k']
        self.lr = config['lr']
        self.batch_size = config['batch_size']
        self.reg_l1 = config['reg_l1']
        self.reg_l2 = config['reg_l2']
        # num of features
        self.p = feature_length

    def add_placeholders(self):
        self.X = tf.compat.v1.sparse_placeholder(tf.float32, [None, self.p])
        self.y = tf.compat.v1.placeholder(tf.int64, [None,])
        self.keep_prob = tf.compat.v1.placeholder(tf.float32)

    def inference(self):
        with tf.compat.v1.variable_scope('linear_layer'):
            b = tf.compat.v1.get_variable('bias', shape=[2],\
	                        initializer=tf.zeros_initializer())
            w1 = tf.compat.v1.get_variable('w1', shape=[self.p, 2],\
	            initializer=tf.compat.v1.truncated_normal_initializer(mean=0, stddev=1e-2))
            self.mult = tf.sparse.sparse_dense_matmul(self.X, w1)
            self.linear_terms = tf.add(tf.sparse.sparse_dense_matmul(self.X, w1), b)

        with tf.compat.v1.variable_scope('interaction_layer'):
            v = tf.compat.v1.get_variable('v', shape=[self.p, self.k],\
	            initializer=tf.compat.v1.truncated_normal_initializer(mean=0, stddev=1e-2))
            self.interaction_terms = tf.multiply(0.5, \
	            tf.reduce_mean(
		        tf.subtract(
			    tf.pow(tf.sparse.sparse_dense_matmul(self.X, v), 2),
			    tf.sparse.sparse_dense_matmul(self.X, tf.pow(v, 2))
			), 1, keepdims=True
		    ))

        self.y_out = tf.add(self.linear_terms, self.interaction_terms)

    def add_loss(self):
        #todo, sparse tensor
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_out)
        #cross_entropy = tf.compat.v1.losses.sparse_softmax_cross_entropy(labels=tf.sparse.to_dense(self.y), logits=self.y_out)
        mean_loss = tf.reduce_mean(cross_entropy)
        self.loss = mean_loss
        tf.summary.scalar('loss', self.loss)

    def add_accuracy(self):
        self.correct_prediction = tf.equal(tf.cast(tf.argmax(self.y_out, 1), tf.int64), self.y)
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', self.accuracy)

    def train(self):
        #学习率指数衰减
        self.global_step = tf.Variable(0, trainable=False)
	#优化器
        optimizer = tf.compat.v1.train.FtrlOptimizer(self.lr,\
                         l1_regularization_strength=self.reg_l1,\
	                 l2_regularization_strength=self.reg_l2)

        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)
        extra_update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def build_graph(self):
        """build graph for model"""
        self.add_placeholders()
        self.inference()
        self.add_loss()
        self.add_accuracy()
        self.train()

def check_restore_parameters(sess, saver):
    """ Restore the previously trained parameters if there are any. """
    ckpt = tf.train.get_checkpoint_state("checkpoints")
    if ckpt and ckpt.model_checkpoint_path:
        logging.info("Loading parameters for the my CNN architectures...")
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        logging.info("Initializing fresh parameters for the my Factorization Machine")

def train_model(sess, model, epochs=10, print_every=50):
    """train model"""
    # Merge all the summaries and write them out to train_logs
    merged = tf.compat.v1.summary.merge_all()
    train_writer = tf.compat.v1.summary.FileWriter('train_logs', sess.graph)
    
    with open('train_sparse_data_frac_0.01.pkl', 'rb') as f:
        sparse_data_fraction = pickle.load(f)
    # get number of batches
    num_batches = len(sparse_data_fraction)

    for e in range(epochs):
        for batch in range(num_batches):
            '''
            if batch != 134:
                continue
            '''j
            #batch size data
            batch_y = sparse_data_fraction[batch]['labels']
            batch_y = np.array(batch_y)
            actual_batch_size = len(batch_y)
            batch_index = np.array(sparse_data_fraction[batch]['indexes'], dtype=np.int64)
   
            batch_shape = np.array([actual_batch_size, model.p], dtype=np.int64)
            batch_value = np.ones(len(batch_index), dtype=np.float32)
            feed_dict = {
                model.X: (batch_index, batch_value, batch_shape),
                model.y: batch_y,
                model.keep_prob: 1.0
            }
            loss, acc, _ = sess.run([model.loss, model.accuracy, model.global_step], feed_dict=feed_dict)
            x = sess.run([model.X], feed_dict=feed_dict)

            print('batch %d finished' % batch)

if __name__ == '__main__':
    '''launching TensorBoard: tensorboard --logdir=path/to/log-directory'''
    # get mode (train or test)
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', help='train or test', type=str)
    args = parser.parse_args()
    mode = args.mode
    # original fields
    fields = ['hour', 'C1', 'C14', 'C15', 'C16', 'C17', 'C18', 'C19', 'C20', 'C21',
                    'banner_pos', 'site_id' ,'site_domain', 'site_category', 'app_domain',
                    'app_id', 'app_category', 'device_model', 'device_type', 'device_id',
                    'device_conn_type','click']
    # loading dicts
    fields_dict = {}
    for field in fields:
        with open('dicts/'+field+'.pkl','rb') as f:
            fields_dict[field] = pickle.load(f)

    # length of representation
    train_array_length = max(fields_dict['click'].values()) + 1
    test_array_length = train_array_length - 2 #?

    # initialize the model
    config = {}
    config['lr'] = 0.01
    config['batch_size'] = 512
    config['reg_l1'] = 2e-2
    config['reg_l2'] = 0
    config['k'] = 40

    # get feature length
    feature_length = test_array_length

    # initialize FM model
    model = FM(config)

    # build graph for model
    model.build_graph()

    saver = tf.compat.v1.train.Saver()

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
	#restore trained parameter
        check_restore_parameters(sess, saver)
        if mode == 'train':
            train_model(sess, model, epochs=20, print_every=500)
