#encoding=utf-8
import tensorflow as tf
import math

class AutoRec(object):
    def __init__(self, num_items, num_users, args):
        self.num_items = num_items
        self.num_users = num_users

        self.hidden_neuron = args.hidden_neuron
        self.lambda_v = args.lambda_value
        self.optimizer_method = args.optimizer_method
        self.batch_size = args.batch_size

        self.num_batch = int(math.ceil(self.num_users / float(self.batch_size)))

        self.base_lr = args.base_lr
        self.global_step = tf.Variable(0, trainable=False)
        self.grad_clip = args.grad_clip
        self.decay_epoch_step = args.decay_epoch_step
        self.decay_step = self.decay_epoch_step * self.num_batch
        self.lr = tf.train.exponential_decay(self.base_lr, self.global_step,\
                      self.decay_step, 0.96, staircase=True) 

    def add_placeholder(self):
        self.input_rating = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name='input_rating')
        self.input_mask_rating = tf.placeholder(dtype=tf.float32, shape=[None, self.num_items], name='input_rating')

    def inference(self):
        self.encoder_v = tf.get_variable(name='encoder_v', shape=[self.num_items, self.hidden_neuron], dtype=tf.float32,\
                        initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
        self.encoder_w = tf.get_variable(name='encoder_w', shape=[self.hidden_neuron, self.num_items], dtype=tf.float32,\
                        initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))

        mu = tf.get_variable(name='mu', shape=[self.hidden_neuron], dtype=tf.float32,\
                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
        
        b = tf.get_variable(name='b', shape=[self.num_items], dtype=tf.float32,\
                 initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))

        pre_encoder = tf.matmul(self.input_rating, self.encoder_v) + mu
        self.encoder = tf.nn.sigmoid(pre_encoder)

        pre_decoder = tf.matmul(self.encoder, self.encoder_w) + b
        self.decoder = tf.identity(pre_decoder)

    def add_loss(self):
         #[batch, num_items]
         pre_rec_cost = tf.multiply((self.input_rating - self.decoder), self.input_mask_rating)
         rec_cost = tf.reduce_sum(tf.square(pre_rec_cost))
         pre_reg_cost = tf.square(self.l2_norm(self.encoder_v)) + tf.square(self.l2_norm(self.encoder_w))
         reg_cost = 0.5 * self.lambda_v * pre_reg_cost
         self.cost = rec_cost + reg_cost

    def train_model(self):
         if self.optimizer_method == 'Adam':
             optimizer = tf.train.AdamOptimizer(self.lr)
         elif self.optimizer_method == 'RMSProp':
             optimizer = tf.train.RMSPropOptimizer(self.lr)
         else:
             raise ValueError("Optimizer Key ERROR")
         if self.grad_clip:
             gvs = optimizer.compute_gradients(self.cost)
             capped_gvs = [(tf.clip_by_value(grad, -5, 5), var) for grad, var in gvs]
             self.optimizer = optimizer.apply_gradients(capped_gvs, global_step=self.global_step)
         else:
             self.optimizer = optimizer.minimize(self.cost, global_step=self.global_step)

    def l2_norm(self, tensor):
        return tf.sqrt(tf.reduce_sum(tf.square(tensor)))

    def build_graph(self):
        self.add_placeholder()
        self.inference()
        self.add_loss()
        self.train_model()
