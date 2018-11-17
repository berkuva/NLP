from util import *

import sklearn_crfsuite as crfsuite
from sklearn_crfsuite import metrics


class CRF(object):
    def __init__(self, trnfile, devfile):
        self.trn_text = load_data(trnfile)
        self.dev_text = load_data(devfile)
        #
        print ("Extracting features on training data ...")
        self.trn_feats, self.trn_tags = self.build_features(self.trn_text)
        print ("Extracting features on dev data ...")
        self.dev_feats, self.dev_tags = self.build_features(self.dev_text)
        #
        self.model, self.labels = None, None

    def build_features(self, text):
        feats, tags = [], []
        for sent in text:
            N = len(sent.tokens)
            sent_feats = []
            for i in range(N):
                word_feats = self.get_word_features(sent, i)
                sent_feats.append(word_feats)
            feats.append(sent_feats)
            tags.append(sent.tags)
        return (feats, tags)

    def train(self):
        print ("Training CRF ...")
        self.model = crfsuite.CRF(
                                  # algorithm="ap",
                                  algorithm='lbfgs',
                                  max_iterations=5)
        self.model.fit(self.trn_feats, self.trn_tags)
        print "------Evaluating train data------"
        trn_tags_pred = self.model.predict(self.trn_feats)
        self.eval(trn_tags_pred, self.trn_tags)
        print "\n------Evaluating dev data------"
        dev_tags_pred = self.model.predict(self.dev_feats)
        self.eval(dev_tags_pred, self.dev_tags)

    def eval(self, pred_tags, gold_tags):
        if self.model is None:
            raise ValueError("No trained model")
        print (self.model.classes_)
        print ("Acc =", metrics.flat_accuracy_score(pred_tags, gold_tags))

    def get_word_features(self, sent, i):
        """ Extract features with respect to time step i
        """

        word_feats = {'tag': sent.tags[i],
                      'token': sent.tokens[i],
                      'first letter': sent.tokens[i][0],
                      'last letter': sent.tokens[i][-1],
                      'first two letters': sent.tokens[i][:2],
                      'last two letters': sent.tokens[i][-2:],
                      'last token': sent.tokens[i-1],
                      }
        if i < len(sent.tokens)-1:
            word_feats.update({'next token': sent.tokens[i+1]})
        return word_feats


if __name__ == '__main__':
    trnfile = "trn-tweet.pos"
    devfile = "dev-tweet.pos"
    crf = CRF(trnfile, devfile)
    crf.train()


