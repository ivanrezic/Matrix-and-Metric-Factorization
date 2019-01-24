import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# read data
df = pd.read_csv('u.data', sep='\t', names=['user', 'item', 'rate', 'time'])
df.head()


global_mean = np.mean(df.rate)
unique_users = pd.unique(df.user)
unique_items = pd.unique(df.item)

user_biases = np.zeros((len(unique_users), 1))
for i, user in enumerate(unique_users):
    user_biases[i] = np.mean(df.loc[df['user'] == user].rate) - global_mean

item_biases = np.zeros((1, len(unique_items)))
for i, item in enumerate(unique_items):
    item_biases[0:i] = np.mean(df.loc[df['item'] == item].rate) - global_mean
    
bias_users = tf.Variable(initial_value=user_biases, dtype=tf.float32)
bias_items = tf.Variable(initial_value=item_biases, dtype=tf.float32)


# variables
feature_len = tf.placeholder(tf.int32)
U = tf.Variable(initial_value=tf.truncated_normal([len(unique_users), feature_len]), name='users', validate_shape=False)
P = tf.Variable(initial_value=tf.truncated_normal([feature_len, len(unique_items)]), name='items', validate_shape=False)

# To the user matrix we add a bias column holding the bias of each user,
# and another column of 1s to multiply the item bias by.
U_plus_bias = tf.concat(axis=1, values = [U, bias_users, tf.ones((len(unique_users),1), dtype=tf.float32)])

# To the item matrix we add a row of 1s to multiply the user bias by, and
# a bias row holding the bias of each item.
P_plus_bias = tf.concat(axis=0, values = [P, tf.ones((1, len(unique_items)), dtype=tf.float32), bias_items])

result = tf.matmul(U, P)
result_flatten = tf.reshape(result, [-1])


df_train, df_test = train_test_split(df, test_size=0.3)

user_indices_train = [x-1 for x in df_train.user.values]
item_indices_train = [x-1 for x in df_train.item.values]

user_indices_test = [x-1 for x in df_test.user.values]
item_indices_test = [x-1 for x in df_test.item.values]

result_flatten_train = tf.gather(result_flatten, user_indices_train * tf.shape(result)[1] + item_indices_train)
result_flatten_test = tf.gather(result_flatten, user_indices_test * tf.shape(result)[1] + item_indices_test)


diff_squared = tf.square(tf.subtract(result_flatten_train, df_train.rate.values))
base_cost = tf.reduce_sum(diff_squared, name="sum_squared_error")

lambda_norms = tf.constant(.001, dtype=tf.float32)
norm_sums = tf.add(tf.reduce_sum(tf.square(U_plus_bias)), tf.reduce_sum(tf.square(P_plus_bias)))
norms_regularized = tf.multiply(lambda_norms, norm_sums)

lambda_biases = tf.constant(.01, dtype=tf.float32)
biases_sums = tf.add(tf.reduce_sum(tf.square(bias_users)), tf.reduce_sum(tf.square(bias_items)))
biases_regularized = tf.multiply(lambda_biases, biases_sums)

norm_biases_sums = tf.add(norms_regularized, biases_regularized)
base_cost_regularized = tf.add(base_cost, norm_biases_sums)


global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(.001, global_step, 10000, 0.96, staircase=True)
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
training_step = optimizer.minimize(base_cost_regularized, global_step=global_step)


#training accuracy
diff_op_train = tf.subtract(result_flatten_train, df_train.rate.values, name='train_diff')
good_val_train = tf.less(tf.abs(diff_op_train), 0.5)
training_accuracy = tf.reduce_sum(tf.cast(good_val_train, tf.float32)) / tf.size(good_val_train, out_type = tf.float32)

#test accuracy
diff_op_test = tf.subtract(result_flatten_test, df_test.rate.values, name='test_diff')
good_val_test = tf.less(tf.abs(diff_op_test), 0.5)
test_accuracy = tf.reduce_sum(tf.cast(good_val_test, tf.float32)) / tf.size(good_val_test, out_type = tf.float32)


sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init, feed_dict={feature_len:10})

iterations = 100000
for i in range(iterations + 1):
    if i % 10000 == 0:
        print("Iter:", i, " Train_acc:", sess.run(training_accuracy), " Test_acc:", sess.run(test_accuracy))

    sess.run(training_step)