class Model:
    def __init__(self, sess, name):
        self.sess = sess
        self.name = name
        self.build_net()
    def build_net(self):
        with tf.Variable_scope(self.name):
            self.x = tf.placeholder(tf.float32, [None,784])
            x_img = tf.reshape(self.x, [-1, 28 ,28 ,1])
            self.y = tf.placeholder(tf.float32, [None, 10])

            # 定义一些网格参数
            w1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev = 0.01))
            w2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev = 0.01))
            w3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev = 0.01))
            wd1 = tf.Variable('wd1', shape = [128*4*4, 625],initializer = tf.contrib.layers.xavier_initializer())
            wd2 = tf.Variable('wd2', shape = [625, 10], initializer = tf.contrib.layers.xavier_initializer())
            b1 = tf.Variable(tf.Constant(0.05, shape = [32]))
            b2 = tf.Variable(tf,Constant(0.05, shape = [64]))
            b3 = tf.Variable(tf,Constant(0.05, shape = [128]))
            bd1 = tf.Variable(tf,Constant(0.05, shape = [625]))
            bd2 = tf.Variable(tf,Constant(0.05, shape = [10])) 
            strides_con = [1, 1, ,1, 1]
            strides_pool = [1, 2, 2, 1]
            ksize = [1, 2, 2, 1]
            #定义卷积函数和全连接函数
            def conv2d(self, input, filter_con, strides_con, b, strides_pool, ksize, padding, keepprop):
                layer = tf.nn.conv2d(input = input, filter = filter_con, strides = strides_con, padding = padding)
                layer += b1
                layer = tf.nn.relu(layer)
                layer = tf.nn.max_pool(input = layer, filter = filter_pool, ksize = ksize, strides = strides_pool, padding = padding)
                layer = tf.dropout(layer, keepprop = keepprop)
                return layer
            def dense1(self, layer, w, b, keepprop):
                num_features = layer_shape.get_shape()[1:4].num_elements()
                layer_flat = tf.reshape(layer, [-1, num_features])
                layer = tf.matmul(layer_flat, w) + b
                layer = tf.nn.relu(layer)
                layer = tf.dropout(layer,keepprop)
                return layer
            def dense2(self, layer, w, b):
                hypothesis = tf.matmul(layer, w) + b
                return hypothesis


            #构建一个cnn
            l1 = conv2d(input = x_img, filter_con = w1, strides_con = strides_con, b =b1, strides_pool =strides_pool, ksize = ksize,padding = padding,keepprop = 0.8)
            l2 = conv2d(input = l1, filter_con = w2, strides_con = strides_con, b =b2, strides_pool =strides_pool, ksize = ksize,padding = padding,keepprop = 0.8)
            l3 = conv2d(input = l2, filter_con = w3, strides_con = strides_con, b =b3, strides_pool =strides_pool, ksize = ksize,padding = padding,keepprop = 0.8)
            d1 = dense1(layer = l3, wd1, bd1, keepprop = 0.8)
            self.logits = dense2(layer = d1, wd2, bd2, keepprop = 0.8)

            #代价函数与优化方法
            self.cost = tf.reduce_mean(tf.softmax_cross_entropy_with_logits(logits = self.logits, labels = self.y))
            self.optimizer = tf.train.AdamOptimizer(learning_rate = 1e-4).minimize(self.cost)
           
            #预测类别
            y_pred = tf.nn.softmax(self.logits)
            self.predict = tf.argmax(y_pred, axis = 1)
            #性能度量
            correct_prediction = tf.equal(self.predict, tf.argmax(self.y, axis = 1))
            self.accuracy = tf.reduce_maen(tf.cast(correct_prediction,tf.float32)) 


    total_iteration = 0
    #定义训练
    def train(self, num_iteration, x_train, y_train):
        global total_iteration
        start_time = time.time()
        for i in range(total_iteration, total_iteration + num_iteration):
            
        cost =  sess.run([cost, optimizer], feed_dict = {self.x : x_train, self.y : y_train})

    #定义测试
    def predict(self, x_test, y_test):
        return sess.run(self.predict, feed_dict = {self.x : x_test, self.y : y_test})

    #定义精度
    def get_accuracy(self, x_data, y_data):
        return sess.run(self.accuracy, feed_dict = {self.x : x_data,self.y : y_data})


def optimize(num_iteration):
    global total_iterations
    start_time=time.time()
    for i in range(total_iterations,total_iterations+num_iteration):
        x_batch,y_batch=mnist.train.next_batch(train_batch_size)
        feed_dict_train={x:x_batch,y_true:y_batch,drop_conv:0.8,drop_fc:0.7}
        session.run(optimizer,feed_dict=feed_dict_train)
        if i%200==0:
            acc=session.run(accuracy,feed_dict=feed_dict_train)
            print('迭代次数：{}\n'.format(i+1))
            print('训练准确率：{0:.1%}'.format(acc))
    total_iterations+=num_iteration
    end_time=time.time()
    time_dif=end_time-start_time
    print('用时：'+str(timedelta(seconds=int(round(time_dif)))))


#initialize
sess = tf.Session()
m1 = Model(sess,'m1')
sess.run(tf.global_Variables_initializer())
print('Learing started!')
#train my model
for epoch in range()









avg_cost = 0
   total_batch = int(mnist.train.num_examples / batch_size)

   for i in range(total_batch):
       batch_xs, batch_ys = mnist.train.next_batch(batch_size)
       c, _ = m1.train(batch_xs, batch_ys)
       avg_cost += c / total_batc


       train_batch_size=64
total_iterations=0



    #将测试集分为更小的批次
test_batch_size=256
def print_test_accuracy(show_example_errors=False,show_confusion_matrix=False):
    num_test=len(mnist.test.images)
    cls_pred=np.zeros(shape=num_test,dtype=np.int)
    i=0
    while i<num_test:
        j=min(i+test_batch_size,num_test)
        images=mnist.test.images[i:j,:]
        labels=mnist.test.labels[i:j,:]
        feed_dict={x:images,y_true:labels,drop_conv:1.0,drop_fc:1.0}
        cls_pred[i:j]=session.run(y_pred_cls,feed_dict=feed_dict)
        i=j
        cls_true=mnist.test.cls
        correct=(cls_true==cls_pred)
        correct_sum=correct.sum()
        acc=float(correct_sum)/num_test
    msg='测试集准确率：{0:.1%} ({1},{2})'
    print(msg.format(acc,correct_sum,num_test))







