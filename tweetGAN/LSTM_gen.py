import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxint)
import tensorflow as tf
import random
import math
import datetime


# load and prepare data 
text = open("preprocessed_final.csv").read()

chars = sorted(list(set(text)))
char_size = len(chars)

char2id = dict((w, i) for i, w in enumerate(chars))
id2char = dict((i, w) for i, w in enumerate(chars))


len_per_section = 140
skip = 140
sections = []
next_chars = []

print "Creating sections..."
for i in range(0, len(text) - len_per_section, skip):
    sections.append(text[i: i + len_per_section])
    next_chars.append(text[i + len_per_section])
print "complete!"

print "Vectorizing..."
X = np.zeros((len(sections), len_per_section, char_size))
y = np.zeros((len(sections), char_size))


for i, section in enumerate(sections):
    for j, char in enumerate(section):
        if char in char2id:
            X[i, j, char2id[char]] = 1
            continue
    if next_chars[i] in char2id:
        y[i, char2id[next_chars[i]]] = 1
        continue

# print(y)
print "complete!"


def sample(prediction):
    r = random.uniform(0,1)
    s = 0
    char_id = len(prediction) - 1
    for i in range(len(prediction)):
        s += prediction[i]
        if s >= r:
            char_id = i
            break
    char_one_hot = np.zeros(shape=[char_size])
    char_one_hot[char_id] = 1.0
    return char_one_hot


def lrelu(x, alpha=0.01):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)


mini_batch = 1024

graph = tf.Graph()
with graph.as_default():

    # Z = tf.placeholder(tf.float32, shape=[None, char_size])
    data = tf.placeholder(tf.float32, [mini_batch, len_per_section, char_size])
    labels = tf.placeholder(tf.float32, [mini_batch, char_size])

    # Generator parameters
    w_ii = tf.get_variable('w_ii', [char_size, char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    w_io = tf.get_variable('w_io', [char_size, char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_i = tf.get_variable('b_i', [1, char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))

    w_fi = tf.get_variable('w_fi', [char_size, char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    w_fo = tf.get_variable('w_fo', [char_size, char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_f = tf.get_variable('b_f', [1, char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))

    w_oi = tf.get_variable('w_oi', [char_size, char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    w_oo = tf.get_variable('w_oo', [char_size, char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_o = tf.get_variable('b_o', [1, char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    
    w_ci = tf.get_variable('w_ci', [char_size, char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    w_co = tf.get_variable('w_co', [char_size, char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    b_c = tf.get_variable('b_c', [1, char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))
    
    theta_G = [w_ii, w_io, b_i, w_fi, w_fo, b_f, w_oi, w_oo, b_o, w_ci, w_co, b_c]


    def lstm_generator(i, o, state):
        with tf.name_scope('Input_gate'):
            input_gate = tf.sigmoid(tf.matmul(i, w_ii) + tf.matmul(o, w_io) + b_i)
            tf.summary.histogram('i', i)
            tf.summary.histogram('o', o)
            tf.summary.histogram("w_ii", w_ii)
            tf.summary.histogram("w_io", w_io)
            tf.summary.histogram('b_i', b_i)
        with tf.name_scope('Forget_gate'):
            forget_gate = tf.sigmoid(tf.matmul(i, w_fi) + tf.matmul(o, w_fo) + b_f)
            tf.summary.histogram("w_fi", w_fi)
            tf.summary.histogram("w_fo", w_fo)
            tf.summary.histogram('b_f', b_f)
        with tf.name_scope('Output_gate'):
            output_gate = tf.sigmoid(tf.matmul(i, w_oi) + tf.matmul(o, w_oo) + b_o)
            tf.summary.histogram("w_oi", w_oi)
            tf.summary.histogram("w_oo", w_oo)
            tf.summary.histogram('b_o', b_o)
        with tf.name_scope('Memory_cell'):
            memory_cell = tf.sigmoid(tf.matmul(i, w_ci) + tf.matmul(o, w_co) + b_c)
            tf.summary.histogram("w_ci", w_ci)
            tf.summary.histogram("w_co", w_co)
            tf.summary.histogram('b_c', b_c)
        with tf.name_scope('Output'):
            state = forget_gate * state + input_gate * memory_cell
            output = output_gate * tf.tanh(state)
            tf.summary.histogram("state", state)
            tf.summary.histogram("output", output)
            return output, state



    # Discriminator parameters
    D_X = tf.placeholder(tf.float32, shape=[None, char_size])

    D_W0 = tf.get_variable('D_W0', [char_size], initializer=tf.truncated_normal_initializer(stddev=0.02))

    D_W1 = tf.get_variable('D_W1', [char_size, mini_batch], initializer=tf.truncated_normal_initializer(stddev=0.02))
    D_b1 = tf.get_variable('D_b1', [mini_batch], initializer=tf.constant_initializer(0))

    D_W2 = tf.get_variable('D_W2', [mini_batch, 1], initializer=tf.truncated_normal_initializer(stddev=0.02))
    D_b2 = tf.get_variable('D_b2', [1], initializer=tf.constant_initializer(0))


    theta_D = [D_W1, D_b1, D_W2, D_b2, D_W0]


    def discriminator(x):
        with tf.name_scope("Dis_Layer"):
            D0 = tf.nn.conv1d(value=x, filters=D_W0, stride=140, padding='SAME')
            D_h1 = lrelu(tf.matmul(D0, D_W1) + D_b1)
            D_logit = tf.matmul(D_h1, D_W2) + D_b2
            D_prob = tf.tanh(D_logit)
            return D_prob, D_logit



    print "Prepare for training...\n"
    output = tf.zeros([mini_batch, char_size])
    state = tf.zeros([mini_batch, char_size])


    for i in range(len_per_section):

        output, state = lstm_generator(data[:, i, :], output, state)

        if i == 0:
            outputs_all_i = output
            labels_all_i = data[:, i+1, :]

        elif i != len_per_section - 1:
            outputs_all_i = tf.concat([outputs_all_i, output], 0)
            labels_all_i = tf.concat([labels_all_i, data[:, i+1, :]], 0)
        
        else:
            outputs_all_i = tf.concat([outputs_all_i, output], 0)
            labels_all_i = tf.concat([labels_all_i, labels], 0)

    w = tf.Variable(tf.truncated_normal([char_size, char_size], -0.1, 0.1))
    b = tf.Variable(tf.zeros([char_size]))

    logits = tf.matmul(outputs_all_i, w) + b

    G_sample, _ = lstm_generator(data[:, 0, :], output, state)
    D_real, D_logit_real = discriminator(D_X)
    D_fake, D_logit_fake = discriminator(G_sample)

    with tf.name_scope("G_loss_labels"):
        G_loss_labels = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels_all_i))
    with tf.name_scope("G_loss_by_D"):
        G_loss_by_D = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    with tf.name_scope("G_loss"):
        G_loss = G_loss_labels + G_loss_by_D


    with tf.name_scope("D_loss_real"):
        D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    with tf.name_scope("D_loss_fake"):
        D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
    with tf.name_scope("D_loss"):
        D_loss = D_loss_real + D_loss_fake

    learning_rate = 1e-4
    with tf.name_scope("D_solver"):
        D_solver = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=theta_D)
    with tf.name_scope("G_solver"):
        G_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=theta_G)
    
    # Tensorboard
    merged_summary = tf.summary.merge_all()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    log_dir = './gan_test/1'
    writer = tf.summary.FileWriter(log_dir, sess.graph)

    # Testing
    test_data = tf.placeholder(tf.float32, shape=[1, char_size])
    test_output = tf.Variable(tf.zeros([1, char_size]))
    test_state = tf.Variable(tf.zeros([1, char_size]))
    
    # Reset at the beginning of each test
    reset_test_state = tf.group(test_output.assign(tf.zeros([1, char_size])), 
                                test_state.assign(tf.zeros([1, char_size])))

    # LSTM
    test_output, test_state = lstm_generator(test_data, test_output, test_state)
    test_prediction = tf.nn.softmax(tf.matmul(test_output, w) + b)

    test_start = 'fake news '

    
num_iter = 100001
save_every = 500
checkpoint_directory = 'ckpt_lstm_test'
# new_ckpt = 'ckpt_gan_with_tensorboard'
with tf.Session(graph=graph) as sess:
    print "Training..."
    tf.global_variables_initializer().run()

    saver = tf.train.import_meta_graph(checkpoint_directory + '/model-70000.meta')
    saver.restore(sess,tf.train.latest_checkpoint('./' + checkpoint_directory))
    # new_saver = tf.train.Saver(var_list=theta_G)

    offset = 0
    print "# Iterations: ", num_iter
    for step in range(num_iter):
        
        offset = offset % len(X)
        
        if offset <= (len(X) - mini_batch):
            batch_data = X[offset: offset + mini_batch]
            batch_labels = y[offset: offset + mini_batch]
            offset += mini_batch
        else:
            to_add = mini_batch - (len(X) - offset)
            batch_data = np.concatenate((X[offset: len(X)], X[0: to_add]))
            batch_labels = np.concatenate((y[offset: len(X)], y[0: to_add]))
            offset = to_add
        
        if step % 5 == 0:
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={D_X: batch_data[:, 0, :], data: batch_data})
        else:
            D_loss_curr = D_loss_curr
        # _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={D_X: batch_data[:, 0, :], data: batch_data})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={data: batch_data, labels: batch_labels})

        
        if step % 100 == 0:
            print('\nIter: {} / {}'.format(step, num_iter))
            print('D_loss: {:.5}'.format(D_loss_curr))
            print('G_loss: {:.5}\n'.format(G_loss_curr))
            print(datetime.datetime.now())

            # if step % save_every == 0:
            #     new_saver.save(sess, new_ckpt + '/model', global_step=step)

            for i in range(len(test_start) - 1):

                test_X = np.zeros((1, char_size))

                test_X[0, char2id[test_start[i]]] = 1.

                _ = sess.run(test_prediction, feed_dict={test_data: test_X})

            test_generated = test_start

            test_X = np.zeros((1, char_size))
            test_X[0, char2id[test_start[-1]]] = 1.

            for i in range(140):

                prediction = test_prediction.eval({test_data: test_X})[0]
                next_char_one_hot = sample(prediction)
                next_char = id2char[np.argmax(next_char_one_hot)]
                test_generated += next_char
                test_X = next_char_one_hot.reshape((1, char_size))

            print(test_generated)
            # with open("LSTM_GAN_tweets.txt", "a") as filehandler:
            #     filehandler.write(str(step))
            #     filehandler.write('\n')
            #     filehandler.write(str(D_loss_curr))
            #     filehandler.write('\n')
            #     filehandler.write(str(G_loss_curr))
            #     filehandler.write('\n')
            #     filehandler.write(test_generated)
            #     filehandler.write('\n')
            #     filehandler.write('\n')


with tf.Session(graph=graph) as sess:
    print "\nTesting..."

    tf.global_variables_initializer().run()
    # model = tf.train.latest_checkpoint(new_ckpt)
    # saver = tf.train.Saver(var_list=theta_G)
    # new_saver.restore(sess, model)

    reset_test_state.run() 
    test_generated = []
    test_generated.append(test_start)

    for i in range(len(test_start) - 1):

        test_X = np.zeros((1, char_size))

        test_X[0, char2id[test_start[i]]] = 1.

        _ = sess.run(test_prediction, feed_dict={test_data: test_X})
    

    test_X = np.zeros((1, char_size))
    test_X[0, char2id[test_start[-1]]] = 1.

    sentence = ""
    for i in range(140):
        prediction = test_prediction.eval(feed_dict={test_data: test_X})[0]
        next_char_one_hot = sample(prediction)
        next_char = id2char[np.argmax(next_char_one_hot)]
        test_generated.append(next_char)
        test_X = next_char_one_hot.reshape((1, char_size))
        sentence += next_char

    print(test_start, sentence)














