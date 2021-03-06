import tensorflow as tf
import numpy as np
try:
    from tensorflow.contrib import keras
except:
    import keras

class SimpleModel:
    def one_hot(self, data, num=10):
        rows = data.shape[0]
        out = []
        for i in range(rows):
            index = data[i][0]
            temp = np.zeros(num)
            temp[index] = 1
            out.append(temp)
        return np.array(out)

    def __init__(self):
        data = keras.datasets.cifar10.load_data()
        (self.x_train, self.y_train), (self.x_test, self.y_test) = data
        self.y_train = self.one_hot(self.y_train)
        self.y_test = self.one_hot(self.y_test)

        self.X = tf.placeholder("float", [None, 32, 32, 3])
        self.Y = tf.placeholder("float", [None, 10])
        # conv layers
        conv1 = self.init_weights([3, 3, 3, 32])
        # FC layers
        fc1 = self.init_weights([16*16*32, 635])
        fc2 = self.init_weights([635, 10])
        self.p_keep_conv = tf.placeholder("float")
        self.p_keep_hidden = tf.placeholder("float")
        self.train_model = self.get_model(self.X, conv1, fc1, fc2, self.p_keep_conv, self.p_keep_hidden)

    def get_model(self, X, conv1, fc1, fc2, p_keep_conv, p_keep_hidden):
        layer1a = tf.nn.relu(tf.nn.conv2d(X, conv1, strides=[1, 1, 1, 1], padding='SAME'))
        layer1 = tf.nn.max_pool(layer1a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
        layer1 = tf.nn.dropout(layer1, p_keep_conv)
        layer1 = tf.reshape(layer1, [-1, fc1.get_shape().as_list()[0]])

        x_fc = tf.nn.relu(tf.matmul(layer1, fc1))
        x_fc = tf.nn.dropout(x_fc, p_keep_hidden)

        out = tf.matmul(x_fc, fc2)
        return out

    def init_weights(self, shape):
        return tf.Variable(tf.random_normal(shape, stddev=.01))

    def get_cost(self):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.train_model, labels=self.Y))

    def train(self, batch_size=128, epoch=5, test_size=256):
        train_op = tf.train.RMSPropOptimizer(.001, .9).minimize(self.get_cost())
        predict_op = tf.argmax(self.train_model, 1)
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            for i in range(epoch):
                train_batch = zip(range(0, len(self.x_train), batch_size),
                                  range(batch_size, len(self.x_train)+1, batch_size))
                for start, end in train_batch:
                    sess.run(train_op, feed_dict={self.X: self.x_train[start:end], 
                                        self.Y: self.y_train[start:end],
                                        self.p_keep_conv: .8,
                                        self.p_keep_hidden: .5})
                test_index = np.arange(len(self.x_test))
                np.random.shuffle(test_index)
                test_index = test_index[:test_size]
                print i, np.mean(np.argmax(self.y_test[test_index], axis=1) ==
                            sess.run(predict_op, feed_dict={self.X: self.x_test[test_index],
                                        self.p_keep_conv: 1.,
                                        self.p_keep_hidden: 1.}))


        
if __name__ == "__main__":
    model = SimpleModel()
    model.train()
