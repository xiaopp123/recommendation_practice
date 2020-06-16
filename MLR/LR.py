#encoding=utf-8
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from data_process import get_data

class LR(object):
    def __init__(self):
        self.learning_rate = 0.3
        self.m = 1
        self.feature_length = 108

    def add_placeholder(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.feature_length], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None], name='y')

    def inference(self):
        self.w = tf.get_variable('w', shape=[self.feature_length, self.m], \
                     dtype=tf.float32,\
                     initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
        self.y_logits = tf.reduce_sum(tf.matmul(self.x, self.w), axis=1)
        self.y_hat = tf.nn.sigmoid(self.y_logits)

    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(\
                                       logits=self.y_logits, labels=self.y))
    
    def train(self):
        self.train_op = tf.train.FtrlOptimizer(self.learning_rate).minimize(self.loss)
        

    def build_graph(self):
        #build tf graph
        self.add_placeholder()
        self.inference()
        self.add_loss()
        self.train()

def train_model(sess, model, epochs, print_every):
    #获取数据
    train_x, train_y, test_x, test_y = get_data()
    batch_size = 64
    for e in range(epochs):
        result = []
        for start in range(0, len(train_y), batch_size):
            end = min(start + batch_size, len(train_y))
            batch_x = train_x[start: end]
            batch_y = train_y[start: end]
            feed_dict = {
                model.x: batch_x,
                model.y: batch_y
            }
            _, loss, y_logits = sess.run([model.train_op, model.loss, model.y_hat], feed_dict=feed_dict)
            result.extend(y_logits)
        auc = roc_auc_score(train_y, result)
        print(auc)
        if e % print_every == 0 and e > 0:
            print(auc)

if __name__ == '__main__':
    lr = LR()
    lr.build_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_model(sess, lr, epochs=10, print_every=50)
