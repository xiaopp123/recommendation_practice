#encoding=utf-8
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from data_process import get_data

class MLR(object):
    def __init__(self):
        self.learning_rate = 0.3
        self.m = 5
        self.feature_length = 108
        
    def add_placeholder(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.feature_length], name='x')
        self.y = tf.placeholder(tf.float32, shape=[None], name='y')
 
    def inference(self):
        u = tf.get_variable('u', shape=[self.feature_length, self.m], dtype=tf.float32,\
                     initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
        w = tf.get_variable('w', shape=[self.feature_length, self.m], dtype=tf.float32,\
                     initializer=tf.truncated_normal_initializer(mean=0,stddev=1e-2))
        U = tf.matmul(self.x, u)
        p1 = tf.nn.softmax(U)
        
        W = tf.matmul(self.x, w)
        p2 = tf.nn.sigmoid(W)

        self.y_logits = tf.reduce_sum(tf.multiply(p1, p2), 1)

    def add_loss(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.y_logits, labels=self.y))

    def train(self):
        self.train_op = tf.train.FtrlOptimizer(self.learning_rate).minimize(self.loss)


    def build_graph(self):
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
            _, loss, y_logits = sess.run([model.train_op, model.loss, model.y_logits], feed_dict=feed_dict)
            result.extend(y_logits)
        auc = roc_auc_score(train_y, result)
        print(auc)
        if e % print_every == 0 and e > 0:
            print(auc)

if __name__ == '__main__':
    mlr = MLR()
    mlr.build_graph()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_model(sess, mlr, epochs=10, print_every=50)
