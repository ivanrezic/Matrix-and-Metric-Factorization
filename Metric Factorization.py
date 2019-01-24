import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time


# read data
df = pd.read_csv('u.data', sep='\t', names=['user', 'item', 'rate', 'time'])
df.head()


num_users = len(pd.unique(df.user))
num_items = len(pd.unique(df.item))

df_train, df_test = train_test_split(df, test_size=0.1)

user_indices_train = [x-1 for x in df_train.user.values]
item_indices_train = [x-1 for x in df_train.item.values]

user_indices_test = [x-1 for x in df_test.user.values]
item_indices_test = [x-1 for x in df_test.item.values]

y = tf.dtypes.cast(df_train.rate.values, tf.float32)
y_test = tf.dtypes.cast(df_test.rate.values, tf.float32)


feature_len = 10

U = tf.Variable(tf.random_normal([num_users, feature_len], mean=0.08, stddev=0.03), dtype=tf.float32)
V = tf.Variable(tf.random_normal([num_items, feature_len], mean=0.08, stddev=0.03), dtype=tf.float32)
B_u = tf.Variable(tf.random_normal([num_users],  stddev=0.001))
B_v = tf.Variable(tf.random_normal([num_items],  stddev=0.001))

#for train set
users_train = tf.nn.embedding_lookup(U ,user_indices_train)
items_train = tf.nn.embedding_lookup(V, item_indices_train)
bias_u_train = tf.nn.embedding_lookup(B_u ,user_indices_train)
bias_v_train = tf.nn.embedding_lookup(B_v ,item_indices_train)

#for test set
users_test = tf.nn.embedding_lookup(U ,user_indices_test)
items_test = tf.nn.embedding_lookup(V, item_indices_test)
bias_u_test = tf.nn.embedding_lookup(B_u ,user_indices_test)
bias_v_test = tf.nn.embedding_lookup(B_v ,item_indices_test)


min_rating, max_rating = 1, 5
global_mean = np.mean(df.rate)

#as we train model on droput distances
#we need these two for accuracy mesurments
distances_test = tf.clip_by_value( tf.reduce_sum( tf.square(users_test - items_test)  ,1) + bias_u_test + bias_v_test +   (max_rating - global_mean), min_rating, max_rating)
distances_train = tf.clip_by_value( tf.reduce_sum( tf.square(users_train - items_train)  ,1) + bias_u_train + bias_v_train +   (max_rating - global_mean), min_rating, max_rating)

dropout_distances = tf.clip_by_value(tf.reduce_sum( tf.nn.dropout(tf.square(users_train - items_train), 0.95) ,1)  + bias_u_train + bias_v_train +   (max_rating - global_mean)  , min_rating, max_rating)
loss = tf.reduce_sum( (1+0.2*tf.abs(y-(max_rating )/2)) * tf.square((max_rating - y) - dropout_distances) ) + 0.01 * (tf.norm(B_u) + tf.norm(B_v) )
optimizer = tf.train.AdagradOptimizer(0.05).minimize(loss)


#training accuracy
diff_op_train = tf.subtract((max_rating - distances_train), y, name='train_diff')
good_val_train = tf.less(tf.abs(diff_op_train), 0.5)
training_accuracy = tf.reduce_sum(tf.cast(good_val_train, tf.float32)) / tf.size(good_val_train, out_type = tf.float32)

#test accuracy
diff_op_test = tf.subtract((max_rating - distances_test), y_test, name='test_diff')
good_val_test = tf.less(tf.abs(diff_op_test), 0.5)
test_accuracy = tf.reduce_sum(tf.cast(good_val_test, tf.float32)) / tf.size(good_val_test, out_type = tf.float32)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

iterations = 100000
for i in range(iterations + 1):
    sess.run((optimizer, loss))
    
    if i % 10000 == 0:
        print("Iter:", i, " Train_acc:", sess.run(training_accuracy), " Test_acc:", sess.run(test_accuracy))