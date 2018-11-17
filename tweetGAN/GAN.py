import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxint)
import tensorflow as tf
from random import randint


# load and prepare data 
text = np.loadtxt("trump_tweet_reuse.csv", dtype=str, delimiter=",")
pretrain_text = text[:len(text)/2]
posttrain_text = text[len(text)/2:]

pre_num_tweets = len(pretrain_text)
print "Number of pretrain tweets: ", pre_num_tweets

post_num_tweets = len(posttrain_text)
print "Number of posttrain tweets: ", post_num_tweets


pre_max_tweet_len, pre_longest_tweet = max([(len(x.split()),x) for x in pretrain_text])
print "Longest pre train tweet consists of {} words: tweet: {}".format(pre_max_tweet_len, pre_longest_tweet)
post_max_tweet_len, post_longest_tweet = max([(len(x.split()),x) for x in posttrain_text])
print "\nLongest post train tweet consists of {} words: tweet: {}".format(post_max_tweet_len, post_longest_tweet)

words = []
count = 0
for tweet in text:
    for word in tweet.split(" "):
        words.append(word)

unique_words = sorted(list(set(words)))
unique_words_len = len(unique_words)


print "number of unique words in text: ", unique_words_len

print "\nBuilding dictionaries... "
word2id = dict((w, i) for i, w in enumerate(unique_words))
id2word = dict((i, w) for i, w in enumerate(unique_words))
print "complete!\n"


def vectorize_tweets(text, pre=True):
    if pre:
        num_tweets = pre_num_tweets
        print "Vectorizing pretrain tweets..."
    else:
        num_tweets = post_num_tweets
        print "Vectorizing post tweets..."

    global tweet_vectors
    tweet_vectors = np.zeros([num_tweets, unique_words_len], dtype=np.float32)

    for i, tweet in enumerate(text):
        vector = np.zeros([unique_words_len])
        for word in tweet.split(" "):
            index = word2id[word]
            vector[index] = 1.

        tweet_vectors[i] = vector
    print "complete!\n"


vectorize_tweets(pretrain_text)


def vec_to_text(vec, pre=True, final=False):
    if final:
        vec = vec[0]

    tweet = ""

    if pre:
        max_tweet_len = pre_max_tweet_len
    else:
        max_tweet_len = post_max_tweet_len

    for num in range(max_tweet_len):
        digit_val = np.argmax(vec)
        vec[digit_val] = 0.
        tweet += id2word[digit_val]
        tweet += " "
    tweet = ' '.join(reversed(tweet.split()))

    return tweet


# params
mini_batch = 32


# Generator parameters
Z = tf.placeholder(tf.float32, shape=[None, unique_words_len])

G_W1 = tf.get_variable('G_W1', [unique_words_len, mini_batch], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
G_b1 = tf.get_variable('G_b1', [mini_batch], initializer=tf.constant_initializer(0))

G_W2 = tf.get_variable('G_W2', [mini_batch, unique_words_len], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
G_b2 = tf.get_variable('G_b2', [unique_words_len], initializer=tf.constant_initializer(0))

G_W3 = tf.get_variable('G_W3', [unique_words_len, mini_batch], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
G_b3 = tf.get_variable('G_b3', [mini_batch], initializer=tf.constant_initializer(0))

G_W4 = tf.get_variable('G_W4', [mini_batch, mini_batch], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
G_b4 = tf.get_variable('G_b4', [mini_batch], initializer=tf.constant_initializer(0))

G_W5 = tf.get_variable('G_W5', [mini_batch, unique_words_len], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
G_b5 = tf.get_variable('G_b5', [unique_words_len], initializer=tf.constant_initializer(0))

theta_G = [G_W1, G_W2, G_W3, G_W4, G_W5, G_b1, G_b2, G_b3, G_b4, G_b5]


# Discriminator parameters
X = tf.placeholder(tf.float32, shape=[None, unique_words_len])

D_W1 = tf.get_variable('D_W1', [unique_words_len, mini_batch], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
D_b1 = tf.get_variable('D_b1', [mini_batch], initializer=tf.constant_initializer(0))

D_W2 = tf.get_variable('D_W2', [mini_batch, mini_batch], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
D_b2 = tf.get_variable('D_b2', [mini_batch], initializer=tf.constant_initializer(0))

D_W3 = tf.get_variable('D_W3', [mini_batch, mini_batch], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
D_b3 = tf.get_variable('D_b3', [mini_batch], initializer=tf.constant_initializer(0))

D_W4 = tf.get_variable('D_W4', [mini_batch, mini_batch], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
D_b4 = tf.get_variable('D_b4', [mini_batch], initializer=tf.constant_initializer(0))

D_W5 = tf.get_variable('D_W5', [mini_batch, 1], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
D_b5 = tf.get_variable('D_b5', [1], initializer=tf.constant_initializer(0))


theta_D = [D_W1, D_W2, D_W3, D_W4, D_W5, D_b1, D_b2, D_b3, D_b4, D_b5]

def leaky_relu(x, alpha=0.01):
    # return tf.nn.relu(x) - alpha * tf.nn.relu(-x)
    return tf.maximum(x, alpha * x)

def generator(z):
    with tf.name_scope('GEN_HL1'):
        G_h1 = leaky_relu(tf.matmul(z, G_W1) + G_b1)
        tf.summary.histogram('G_W1', G_W1)
        tf.summary.histogram("G_b1", G_b1)
        tf.summary.histogram("G_h1", G_h1)
    with tf.name_scope('GEN_HL2'):
        G_h2 = leaky_relu(tf.matmul(G_h1, G_W2) + G_b2)
        tf.summary.histogram('G_W2', G_W2)
        tf.summary.histogram("G_b2", G_b2)
        tf.summary.histogram("G_h2", G_h2)
    with tf.name_scope('GEN_HL3'):
        G_h3 = leaky_relu(tf.matmul(G_h2, G_W3) + G_b3)
        tf.summary.histogram('G_W3', G_W3)
        tf.summary.histogram("G_b3", G_b3)
        tf.summary.histogram("G_h3", G_h3)
    with tf.name_scope('GEN_HL4'):
        G_h4 = leaky_relu(tf.matmul(G_h3, G_W4) + G_b4)
        tf.summary.histogram('G_W4', G_W4)
        tf.summary.histogram("G_b4", G_b4)
        tf.summary.histogram("G_h4", G_h4)
    with tf.name_scope('GEN_HL5'):
        G_prob = tf.nn.sigmoid(tf.matmul(G_h4, G_W5) + G_b5)
        tf.summary.histogram('G_W5', G_W5)
        tf.summary.histogram("G_b5", G_b5)
        tf.summary.histogram("G_prob", G_prob)
        
        return G_prob

def discriminator(x):
    with tf.name_scope('DIS_HL1'):
        D_h1 = leaky_relu(tf.matmul(x, D_W1) + D_b1)
        tf.summary.histogram('D_W1', D_W1)
        tf.summary.histogram("D_b1", D_b1)
        tf.summary.histogram("D_h1", D_h1)
    with tf.name_scope('DIS_HL2'):
        D_h2 = leaky_relu(tf.matmul(D_h1, D_W2) + D_b2)
        tf.summary.histogram('D_W2', D_W2)
        tf.summary.histogram("D_b2", D_b2)
        tf.summary.histogram("D_h2", D_h2)
    with tf.name_scope('DIS_HL3'):
        D_h3 = leaky_relu(tf.matmul(D_h2, D_W3) + D_b3)
        tf.summary.histogram('D_W3', D_W3)
        tf.summary.histogram("D_b3", D_b3)
        tf.summary.histogram("D_h3", D_h3)
    with tf.name_scope('DIS_HL4'):
        D_h4 = leaky_relu(tf.matmul(D_h3, D_W4) + D_b4)
        tf.summary.histogram('D_W4', D_W4)
        tf.summary.histogram("D_b4", D_b4)
        tf.summary.histogram("D_h4", D_h4)
    with tf.name_scope('DIS_HL5'):
        D_logit = tf.matmul(D_h4, D_W5) + D_b5
        D_prob = tf.nn.sigmoid(D_logit)
        tf.summary.histogram('D_W5', D_W5)
        tf.summary.histogram("D_b5", D_b5)
        tf.summary.histogram("D_prob", D_prob)

        return D_prob, D_logit


# Training
print "\tStart training..."
G_sample = generator(Z)
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_sample)

# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1.0 - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))
with tf.name_scope("D_loss_real"):
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
    tf.summary.scalar("D_loss_real", D_loss_real)
with tf.name_scope("D_loss_fake"):
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    tf.summary.scalar("D_loss_fake", D_loss_fake)
with tf.name_scope("D_loss"):
    D_loss = D_loss_real + D_loss_fake
    tf.summary.scalar("D_loss", D_loss)
with tf.name_scope("G_loss"):
    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
    tf.summary.scalar("G_loss", G_loss)


learning_rate = 1e-4
with tf.name_scope("D_solver"):
    D_solver = tf.train.AdamOptimizer(learning_rate).minimize(D_loss, var_list=theta_D)
with tf.name_scope("G_solver"):
    G_solver = tf.train.AdamOptimizer(learning_rate).minimize(G_loss, var_list=theta_G)

# -----------------------------------------------------------------------------------------------------------------------------
merged_summary = tf.summary.merge_all()

sess = tf.Session()
sess.run(tf.global_variables_initializer())
loc_dir = '/Users/hyunjaecho/Desktop/emory/DT_Tweets/gan_bow/1'

writer = tf.summary.FileWriter(loc_dir, sess.graph)

# -----------------------------------------------------------------------------------------------------------------------------

G_W1_ = np.zeros([unique_words_len, mini_batch])
G_b1_ = np.zeros([mini_batch])
G_W2_ = np.zeros([mini_batch, unique_words_len])
G_b2_ = np.zeros([unique_words_len])
G_W3_ = np.zeros([unique_words_len, mini_batch])
G_b3_ = np.zeros([mini_batch])
G_W4_ = np.zeros([mini_batch, mini_batch])
G_b4_ = np.zeros([mini_batch])
G_W5_ = np.zeros([mini_batch, unique_words_len])
G_b5_ = np.zeros([unique_words_len])

D_W1_ = np.zeros([unique_words_len, mini_batch])
D_b1_ = np.zeros([mini_batch])
D_W2_ = np.zeros([mini_batch, mini_batch])
D_b2_ = np.zeros([mini_batch])
D_W3_ = np.zeros([mini_batch, mini_batch])
D_b3_ = np.zeros([mini_batch])
D_W4_ = np.zeros([mini_batch, mini_batch])
D_b4_ = np.zeros([mini_batch])
D_W5_ = np.zeros([mini_batch, 1])
D_b5_ = np.zeros([1])

def update_parameters():
    G_W1, G_b1, G_W2, G_b2, G_W3, G_b3, G_W4, G_b4, G_W5, G_b5 = G_W1_, G_b1_, G_W2_, G_b2_, G_W3_, G_b3_, G_W4_, G_b4_, G_W5_, G_b5_
    D_W1, D_b1, D_W2, D_b2, D_W3, D_b3, D_W4, D_b4, D_W5, D_b5 = D_W1_, D_b1_, D_W2_, D_b2_, D_W3_, D_b3_, D_W4_, D_b4_, D_W5_, D_b5_


def assign_updated_vals_to_params():
    G_W1_ = G_W1.eval(session=sess)
    G_b1_ = G_b1.eval(session=sess)
    G_W2_ = G_W2.eval(session=sess)
    G_b2_ = G_b2.eval(session=sess)
    G_W3_ = G_W3.eval(session=sess)
    G_b3_ = G_b3.eval(session=sess)
    G_W4_ = G_W4.eval(session=sess)
    G_b4_ = G_b4.eval(session=sess)
    G_W5_ = G_W5.eval(session=sess)
    G_b5_ = G_b5.eval(session=sess)

    D_W1_ = D_W1.eval(session=sess)
    D_b1_ = D_b1.eval(session=sess)
    D_W2_ = D_W2.eval(session=sess)
    D_b2_ = D_b2.eval(session=sess)
    D_W3_ = D_W3.eval(session=sess)
    D_b3_ = D_b3.eval(session=sess)
    D_W4_ = D_W4.eval(session=sess)
    D_b4_ = D_b4.eval(session=sess)
    D_W5_ = D_W5.eval(session=sess)
    D_b5_ = D_b5.eval(session=sess)

    # save_parameters_to_file()


def save_parameters_to_file():
    with open("gan_saved_parameters.txt", 'w') as file_handler:

        file_handler.write("{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}".\
                    format(G_W1_, G_b1_, G_W2_, G_b2_, G_W3_, G_b3_, G_W4_, G_b4_, G_W5_, G_b5_))

        file_handler.write("{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}\n\n{}".\
                    format(D_W1_, D_b1_, D_W2_, D_b2_, D_W3_, D_b3_, D_W4_, D_b4_, D_W5_, D_b5_))


loss_track = []

num_iter = 201
start = 0
discriminator_update_count = 0
for i in range(num_iter):
    # Pre train
    if i < num_iter/2:

        # get tweet vectors
        sample_tweets = tweet_vectors[start:start + mini_batch]

        for st in sample_tweets:
            # add noise
            noise = np.random.normal(-1., 1., size=[unique_words_len])
            st += noise
            # normalize
            norm = np.linalg.norm(st)
            st = st / norm

        if i == (num_iter/2)-1:
            start = 0
        else:
            start += mini_batch

        if i == 0:
            print "\n\t\tBegin pre-training... "
            discriminator_update_count +=1
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: sample_tweets, Z: np.random.normal(-1., 1., size=[mini_batch, unique_words_len])})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: np.random.normal(-1., 1., size=[mini_batch, unique_words_len])})

        else:
            if G_loss_curr < 0.8 and D_loss_curr > 0.1:
                discriminator_update_count +=1
                _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: sample_tweets, Z: np.random.normal(-1., 1., size=[mini_batch, unique_words_len])})
            else:
                D_loss_curr = D_loss_curr
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: np.random.normal(-1., 1., size=[mini_batch, unique_words_len])})
            
                        

        # Save results from pre-train for post-train
        if i == num_iter/2 - 1:
            print "\n\t\tEnding pre-training...\n"

            assign_updated_vals_to_params()

    # Post train
    else:
        
        update_parameters()

        if i == num_iter/2:
            vectorize_tweets(posttrain_text, pre=False)

        sample_tweets = tweet_vectors[start:start + mini_batch]

        for st in sample_tweets:
            noise = np.random.normal(-1., 1., size=[unique_words_len])
            st += noise
            norm = np.linalg.norm(st)
            st = st / norm

        if start + mini_batch == post_num_tweets:
            print i
            start = 0
        else:
            start += mini_batch

        if i == num_iter/2:
            print "\n\t\tBegin post-training..."
            discriminator_update_count +=1
            _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: sample_tweets, Z: np.random.normal(-1., 1., size=[mini_batch, unique_words_len])})
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: np.random.normal(-1., 1., size=[mini_batch, unique_words_len])})

        else:
            if G_loss_curr < 0.5 and D_loss_curr > 0.1:
                discriminator_update_count +=1
                _, D_loss_curr = sess.run([D_solver, D_loss], feed_dict={X: sample_tweets, Z: np.random.normal(-1., 1., size=[mini_batch, unique_words_len])})
            else:
                D_loss_curr = D_loss_curr
            _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: np.random.normal(-1., 1., size=[mini_batch, unique_words_len])})


        # Save results from pre-train for post-train
        if i == num_iter - 1:
            print "\t\tEnding post-training..."

            assign_updated_vals_to_params()

    if i % 10 == 0:
        s = sess.run(merged_summary, feed_dict={X: sample_tweets, Z: np.random.normal(-1., 1., size=[mini_batch, unique_words_len])})
        writer.add_summary(s, i)


    if i % 100 == 0:
       # print "generated_tweets:"

        numbering = 0
        tweet_track = ""

        gen_out = generator(tf.cast(np.random.normal(-1., 1., size=[mini_batch, unique_words_len]), tf.float32)).eval(session=sess)

        for _ in range(gen_out.shape[0]):
            current_row = gen_out[_]
            if i < num_iter/2:
                generated_tweet = vec_to_text(current_row)
            else:
                generated_tweet = vec_to_text(current_row, pre=False)

            if _ == gen_out.shape[0] - 1:
                tweet_track = generated_tweet
            numbering += 1

        # print "\n"
        # loss_track.append((i, D_loss_curr, G_loss_curr, tweet_track))
        # print "Loss track:\n", loss_track
        print "\n"
        print('Iter: {} / {}'.format(i, num_iter))
        print('D_loss: {:.4}'.format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        # print



update_parameters()

given_text = "america".lower()
cleaned_text = ""
for word in given_text.split(" "):
    if word in unique_words:
        cleaned_text += word
        cleaned_text += " "

cleaned_text.strip()
        
print "cleaned given text: ", cleaned_text


given_text_to_vec = np.zeros([1, unique_words_len], dtype=np.float32)
inc = 1
for word in cleaned_text.split(" "):
    if word != "":
        ind = word2id[word]
        given_text_to_vec[0][ind] += inc
        inc += 1


generated_vec = generator(given_text_to_vec).eval(session=sess)
generated_tweet = vec_to_text(generated_vec, final=True)

print "\nGenerated tweet: {}\n".format(cleaned_text + generated_tweet)









