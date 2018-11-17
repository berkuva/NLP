import numpy as np
from nltk.tokenize import RegexpTokenizer

# Load files and labels
dir_path = "/Users/hyunjaecho/Desktop/proj02/"
train_pos = dir_path + "trn.pos"
dev_pos = dir_path + "dev.pos"
tst_file = dir_path + "tst.word"

labels = ["A", "C", "D", "M", "N", "O", "P", "R", "V", "W", "<start>", "<end>"]
true_labels = labels[:-2]
num_labels = len(labels)

train_file = open(train_pos, "r").readlines()
dev_file = open(dev_pos, "r").readlines()
tst_file = open(tst_file, "r").readlines()

write_to_file = False

######################## 1.2.1 - Preprocessing, K=5 ########################

threshold_K = 2
tokenizer = RegexpTokenizer(r'\w+')

def hasNumbers(inputString):
    return any(char.isdigit() for char in inputString)

train_word_dict = {}
for line in train_file:
    for word in line.lower().split():
        w, _ = word.split("/")
        if hasNumbers(w):
            continue
        else:
            # tokens = tokenizer.tokenize(w)
            # for tok in tokens:
            if w in train_word_dict:
                train_word_dict[w] += 1
            else:
                train_word_dict[w] = 1

train_set = set()
for k, v in train_word_dict.items():
    if v >= threshold_K:
        train_set.add(k)

train_set.add("Unk")
num_vocab = len(train_set)
print("Vocabulary size = ", num_vocab)

train_vocab = sorted(list(train_set))

word2idx = dict((w, i) for i, w in enumerate(train_vocab))
idx2word = dict((i, w) for i, w in enumerate(train_vocab))


######################## 1.2.2 - Transition Probability ########################

def label_transition_prob():

    label_transitions = np.zeros((num_labels, num_labels))

    # Fill label_transitions matrix
    num_sentences = 0
    for line in train_file:
        num_sentences += 1
        previous_tag_index = labels.index("<start>")
        for word in line.lower().split():
            _, tag = word.split("/")
            cur_tag = tag.upper()
            cur_tag_ind = labels.index(cur_tag)

            label_transitions[previous_tag_index][cur_tag_ind] += 1
            previous_tag_index = cur_tag_ind

        previous_tag_index = cur_tag_ind
        cur_tag_ind = labels.index("<end>")
        label_transitions[previous_tag_index][cur_tag_ind] += 1

    # Calculate the total for each row for normalization
    total = []
    for k in range(num_labels):
        total.append(np.sum(label_transitions[k,:]))

    # Normalize to make the sum of Pr(yt|yt-1) = 1 for all yt
    normalized_label_transitions = np.zeros((num_labels, num_labels))
    for i in range(num_labels):
        cur_total = total[i]
        if cur_total == 0:
            continue
        for j in range(num_labels):
            normalized_label_transitions[i][j] += label_transitions[i][j]/cur_total

    # Write to file
    if write_to_file:
        out_path = dir_path + "hc2kc-tprob.txt"
        open(out_path, "w")
        tups = []
        for i in range(num_labels):
            for j in range(num_labels):
                result = labels[i], labels[j], normalized_label_transitions[i][j]
                tups.append(result)
        np.savetxt(out_path, tups, fmt="%s", delimiter=", ")

    return label_transitions, normalized_label_transitions


label_transitions, normalized_label_transitions = label_transition_prob()


######################## 1.2.3 - Emission Probability ########################

def word_transition_prob():
    word_label_matrix = np.zeros((num_labels, num_vocab))

    # Fill word_label_matrix
    num_sentences = 0
    for line in train_file:
        num_sentences += 1
        for word in line.lower().split():
            w, tag = word.split("/")
            if w not in train_set:
                w = "Unk"
            cur_tag = tag.upper()
            cur_tag_ind = labels.index(cur_tag)
            wordid = word2idx[w]

            word_label_matrix[cur_tag_ind][wordid] += 1

    # Count total for each row for normalization
    total_per_label = []
    for i in range(num_labels):
        total_per_label.append(np.sum(word_label_matrix[i,:]))

    # Normalize to make the sum of Pr(xt|yt) = 1 for all xt
    normalized_word_label_matrix = np.zeros((num_labels, num_vocab))
    for i in range(num_labels):
        cur_total = total_per_label[i]
        if cur_total == 0:
            continue
        for j in range(num_vocab):
            normalized_word_label_matrix[i][j] += word_label_matrix[i][j]/cur_total

    # Write to file
    if write_to_file:
        out_path = dir_path + "hc2kc-eprob.txt"
        open(out_path, "w")
        tups = []
        for la in range(num_labels):
            for v in range(num_vocab):
                result = labels[la], idx2word[v],normalized_word_label_matrix[la][v]
                tups.append(result)
        np.savetxt(out_path, tups, fmt="%s", delimiter=", ")

    return total_per_label, word_label_matrix, normalized_word_label_matrix


total_per_label, word_label_matrix, normalized_word_label_matrix = word_transition_prob()


######################## 1.2.4 - Smoothing ########################

alpha = 1
V = num_vocab
beta = 0.01
N = num_labels

def smoothed_emission():
    smoothed_emission_matrix = normalized_word_label_matrix.copy()
    for i in range(num_labels):
        for j in range(num_vocab):
            numerator = smoothed_emission_matrix[i][j] + alpha
            denominator = total_per_label[i] + V*alpha
            probability = numerator / denominator
            smoothed_emission_matrix[i][j] += probability

    # Write to file
    if write_to_file:
        out_path = dir_path + "hc2kc-eprob-smoothed.txt"
        open(out_path, "w")
        tups = []
        for i in range(num_labels):
            for j in range(num_vocab):
                result = labels[i], idx2word[j], smoothed_emission_matrix[i][j]
                tups.append(result)
        np.savetxt(out_path, tups, fmt="%s", delimiter=", ")

    return smoothed_emission_matrix, np.log(smoothed_emission_matrix)

smoothed_emission_matrix, log_smoothed_emission_matrix = smoothed_emission()


def smoothed_transitions():
    smoothed_transitions_matrix = normalized_label_transitions.copy()
    for i in range(num_labels):
        for j in range(num_labels):
            numerator = smoothed_transitions_matrix[i][j] + beta
            denominator = total_per_label[i] + N * beta
            probability = numerator / denominator
            smoothed_transitions_matrix[i][j] += probability

    # Write to file
    if write_to_file:
        out_path = dir_path + "hc2kc-tprob-smoothed.txt"
        open(out_path, "w")
        tups = []
        for i in range(num_labels):
            for j in range(num_labels):
                result = labels[i], labels[j], smoothed_transitions_matrix[i][j]
                tups.append(result)
        np.savetxt(out_path, tups, fmt="%s", delimiter=", ")

    return smoothed_transitions_matrix, np.log(smoothed_transitions_matrix)

smoothed_transitions_tups, log_smoothed_transitions_matrix = smoothed_transitions()


######################## 1.2.5 - Viterbi, log space ########################

# Helper function
def calculate_log_emission_score(word, label):
    if word not in train_set:
        word = 'Unk'
    wordID = word2idx[word]
    labelID = labels.index(label)
    return log_smoothed_emission_matrix[labelID][wordID]


# Helper function
def calculate_log_transition_score(prev_label, cur_label):
    prev_index = labels.index(prev_label)
    cur_index = labels.index(cur_label)
    return log_smoothed_transitions_matrix[prev_index][cur_index]


def viterbi(sentence, test=False):
    viterbi_matrix = np.zeros(shape=(num_vocab, num_labels))

    # Get the first word and its index
    if not test:
        first_word, _ = sentence.lower().split()[0].split("/")
    else:
        first_word = sentence.lower().split()[0]
    if first_word not in train_set:
        first_word = 'Unk'
    first_word_idx = word2idx[first_word]

    backpointers = []

    # Initialization
    max_score = float("-inf")
    pointer = None
    for k, label in enumerate(true_labels):

        log_transition_score = calculate_log_transition_score('<start>', label)
        log_emission_score = calculate_log_emission_score(first_word, label)

        score = log_transition_score + log_emission_score
        viterbi_matrix[first_word_idx][k] += score

        if score > max_score:
            max_score = score
            pointer = label

    backpointers.append(pointer)

    prev_word = first_word
    for word in sentence.lower().split()[1:]:
        if not test:
            w, _ = word.split("/")
        else:
            w = word
        if w in train_set:
            wordid = word2idx[w]
        else:
            wordid = word2idx['Unk']

        if prev_word not in train_set:
            prev_word = "Unk"
        prev_word_idx = word2idx[prev_word]
        max_score = float('-inf')
        best_label_so_far = None
        for i, cur_label in enumerate(true_labels):
            scores_for_cur_label = []
            for j, prev_label in enumerate(true_labels):
                log_transition_score = calculate_log_transition_score(prev_label, cur_label)
                prev_label_score = log_transition_score + viterbi_matrix[prev_word_idx][j]
                scores_for_cur_label.append(prev_label_score)

            log_emission_score = calculate_log_emission_score(w, cur_label)
            cur_score = log_emission_score + max(scores_for_cur_label)

            viterbi_matrix[wordid][i] += cur_score

            if cur_score > max_score:
                max_score = cur_score
                best_label_so_far = cur_label

        backpointers.append(best_label_so_far)
        prev_word = w

    return backpointers



######################## 1.2.6 - accuracy on dev data ########################

correct = 0
wrong = 0
def update_accuracy(reference, prediction):
    global correct, wrong
    assert len(reference) == len(prediction)
    length = len(reference)
    for i in range(length):
        if reference[i] == prediction[i]:
            correct += 1
        else:
            wrong += 1

count = 0
for line in dev_file:
    reference = [tok.split("/")[1].upper() for tok in line.lower().split()]
    prediction = viterbi(line)
    update_accuracy(reference, prediction)
    # if count % 3000 == 0:
    #     print(reference)
    #     print(prediction)
    #     update_accuracy(reference, prediction)
    #     print("running accuracy at iter #{} is {}\n".format(count, correct / (correct+wrong)))
    # count += 1
print("Final accuracy on dev data: {} | parameters: alpha: {}, beta: {}".format(correct / (correct+wrong), alpha, beta))


######################## 1.2.6 - test data ########################
if write_to_file:
    out_path = dir_path + "hc2kc-viterbi.txt"
    open(out_path, "w")
    out_file = open(out_path, "a")

    for line in tst_file:
        prediction = viterbi(line, test=True)
        # Make sure that the number of tags from viterbi is equal to the number of words
        assert len(prediction) == len(line.split())
        result = ""
        for i, word in enumerate(line.lower().split()):
            tag = prediction[i].upper()
            format = word + "/" + tag + " "
            result += format
        out_file.write(result)
        out_file.write("\n")
    out_file.close()



######################## 1.2.8 - test data (tuned) ########################
if write_to_file:
    out_path = dir_path + "hc2kc-viterbi-tuned.txt"
    open(out_path, "w")
    out_file = open(out_path, "a")

    for line in tst_file:
        prediction = viterbi(line, test=True)
        # Make sure that the number of tags from viterbi is equal to the number of words
        assert len(prediction) == len(line.split())
        result = ""
        for i, word in enumerate(line.lower().split()):
            tag = prediction[i].upper()
            format = word + "/" + tag + " "
            result += format
        out_file.write(result)
        out_file.write("\n")
    out_file.close()


pass
