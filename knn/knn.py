import argparse
from collections import Counter, defaultdict

import random
import numpy
from numpy import median
from sklearn.neighbors import BallTree

class Numbers:
    """
    Class to store MNIST data
    """

    def __init__(self, location):
        # You shouldn't have to modify this class, but you can if
        # you'd like.

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)

        self.train_x, self.train_y = train_set
        self.test_x, self.test_y = valid_set
        f.close()


class Knearest:
    """
    kNN classifier
    """

    def __init__(self, x, y, k=5):
        """
        Creates a kNN instance

        :param x: Training data input
        :param y: Training data output
        :param k: The number of nearest points to consider in classification
        """

        # You can modify the constructor, but you shouldn't need to.
        # Do not use another data structure from anywhere else to
        # complete the assignment.

        self._kdtree = BallTree(x)
        self._y = y
        self._k = k

    def majority(self, item_indices):
        """
        Given the indices of training examples, return the majority label.  If
        there's a tie, return the median of the majority labels (as implemented 
        in numpy).

        :param item_indices: The indices of the k nearest neighbors
        """
        med_array = []
        labels = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, -1: 0}
        assert len(item_indices) == self._k, "Did not get k inputs"
        for index in item_indices:  
            if self._y[index] == 0:
                labels[0] = labels[0] + 1
            elif self._y[index] == 1:
                labels[1] = labels[1] + 1
            elif self._y[index] == -1:
                labels[-1] = labels[-1] + 1
            elif self._y[index] == 2:
                labels[2] = labels[2] + 1
            elif self._y[index] == 3:
                labels[3] = labels[3] + 1
            elif self._y[index] == 4:
                labels[4] = labels[4] + 1
            elif self._y[index] == 5:
                labels[5] = labels[5] + 1
            elif self._y[index] == 6:
                labels[6] = labels[6] + 1
            elif self._y[index] == 7:
                labels[7] = labels[7] + 1
            elif self._y[index] == 8:
                labels[8] = labels[8] + 1
            elif self._y[index] == 9:
                labels[9] = labels[9] + 1
            else:
                print('error, not a valid digit')

        major = max(labels.values())
        num_max = labels.values().count(major)
        #if there is more than one major, get the median
        if num_max > 1:
            for key in labels:
                #check label at key, if it is a major, add that key to med_array
                if labels.get(key) == major:
                    med_array.append(key)
            return numpy.median(med_array)

        return labels.keys()[labels.values().index(major)]

    def classify(self, example):
        """
        Given an example, classify the example.

        :param example: A representation of an example in the same
        format as training data
        """

        #get indices of k neareast nieghbors
        dist, ind = self._kdtree.query([example], self._k)

        #query majority function to get majority label of k nearest
        return self.majority(ind[0])

    def confusion_matrix(self, test_x, test_y):
        """
        Given a matrix of test examples and labels, compute the confusion
        matrix for the current classifier.  Should return a dictionary of
        dictionaries where d[ii][jj] is the number of times an example
        with true label ii was labeled as jj.

        :param test_x: Test data representation
        :param test_y: Test data answers
        """
        d = defaultdict(dict)
        data_index = 0
        for xx, yy in zip(test_x, test_y):
            #classify each input xx and compare to true label yy
            label = self.classify(xx)
            d[yy][label] = d.get(yy, {}).get(label, 0) + 1
        return d

    @staticmethod
    def accuracy(confusion_matrix):
        """
        Given a confusion matrix, compute the accuracy of the underlying classifier.
        """

        # You do not need to modify this function

        total = 0
        correct = 0
        for ii in confusion_matrix:
            total += sum(confusion_matrix[ii].values())
            correct += confusion_matrix[ii].get(ii, 0)

        if total:
            return float(correct) / float(total)
        else:
            return 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='KNN classifier options')
    parser.add_argument('--k', type=int, default=3,
                        help="Number of nearest points to use")
    parser.add_argument('--limit', type=int, default=-1,
                        help="Restrict training to this many examples")
    args = parser.parse_args()

    data = Numbers("../data/mnist.pkl.gz")

    # You should not have to modify any of this code

    if args.limit > 0:
        print("Data limit: %i" % args.limit)
        knn = Knearest(data.train_x[:args.limit], data.train_y[:args.limit],
                       args.k)
    else:
        knn = Knearest(data.train_x, data.train_y, args.k)
    print("Done loading data")

    confusion = knn.confusion_matrix(data.test_x, data.test_y)
    print("\t" + "\t".join(str(x) for x in xrange(10)))
    print("".join(["-"] * 90))
    for ii in xrange(10):
        print("%i:\t" % ii + "\t".join(str(confusion[ii].get(x, 0))
                                       for x in xrange(10)))
    print("Accuracy: %f" % knn.accuracy(confusion))
