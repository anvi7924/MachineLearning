from csv import DictReader, DictWriter

import numpy as np
from numpy import array

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix

kTARGET_FIELD = 'spoiler'
kTEXT_FIELD = 'sentence'


class Featurizer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.vectorizer = TfidfVectorizer()
        self.kill = TfidfVectorizer(analyzer='word', token_pattern='kill', max_features=500)
        self.death = TfidfVectorizer(analyzer='word', token_pattern='death', max_features=500)
        self.end = TfidfVectorizer(analyzer='word', token_pattern='end', max_features=500)
        self.finale = TfidfVectorizer(analyzer='word', token_pattern='finale', max_features=500)
    def train_feature(self, examples):
        features = FeatureUnion([('tfidf', self.vectorizer), ('kill', self.kill), ('death', self.death), ('end', self.end), ('finale', self.finale)])
        return features.fit_transform(examples)
    def test_feature(self, examples):
        features = FeatureUnion([('tfidf', self.vectorizer), ('kill', self.kill), ('death', self.death), ('end', self.end), ('finale', self.finale)])
        return features.transform(examples)

    def show_top10(self, classifier, categories):
        feature_names = np.asarray(self.vectorizer.get_feature_names())
        if len(categories) == 2:
            top10 = np.argsort(classifier.coef_[0])[-10:]
            bottom10 = np.argsort(classifier.coef_[0])[:10]
            print("Pos: %s" % " ".join(feature_names[top10]))
            print("Neg: %s" % " ".join(feature_names[bottom10]))
        else:
            for i, category in enumerate(categories):
                top10 = np.argsort(classifier.coef_[i])[-10:]
                print("%s: %s" % (category, " ".join(feature_names[top10])))

if __name__ == "__main__":

    # Cast to list to keep it all in memory
    train = list(DictReader(open("../data/spoilers/train.csv", 'r')))
    test = list(DictReader(open("../data/spoilers/test.csv", 'r')))
    ex = list(DictReader(open("../data/spoilers/ex.csv", 'r')))
    #print('test', test)
    feat = Featurizer()

    labels = []
    for line in train:
        if not line[kTARGET_FIELD] in labels:
            labels.append(line[kTARGET_FIELD])

    print("Label set: %s" % str(labels))
    x_train = feat.train_feature([x[kTEXT_FIELD] for x in train])
    x_test = feat.test_feature([x[kTEXT_FIELD] for x in test])

    y_train = array(list(labels.index(x[kTARGET_FIELD])
                         for x in train))

    print(len(train), len(y_train))
    print(set(y_train))
    # Train classifier
    lr = SGDClassifier(loss='log', penalty='l2', shuffle=True)
    lr.fit(x_train, y_train)
    feat.show_top10(lr, labels)
    print('score', cross_val_score(lr, x_train, y_train, cv=5))
    predictions = lr.predict(x_test)
    o = DictWriter(open("predictions.csv", 'w'), ["Id", "spoiler"])
    o.writeheader()
    for ii, pp in zip([x['Id'] for x in test], predictions):
        #print('pp', pp)
        d = {'Id': ii, 'spoiler': labels[pp]}
        o.writerow(d)
