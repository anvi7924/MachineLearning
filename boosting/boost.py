import argparse
import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Perceptron
from sklearn.base import clone 
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

np.random.seed(1234)

class FoursAndNines:
    """
    Class to store MNIST data
    """

    def __init__(self, location):

        import cPickle, gzip

        # Load the dataset
        f = gzip.open(location, 'rb')

        # Split the data set 
        train_set, valid_set, test_set = cPickle.load(f)

        # Extract only 4's and 9's for training set 
        self.x_train = train_set[0][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0],:]
        self.y_train = train_set[1][np.where(np.logical_or( train_set[1]==4, train_set[1] == 9))[0]]
        self.y_train = np.array([1 if y == 9 else -1 for y in self.y_train])
        
        # Shuffle the training data 
        shuff = np.arange(self.x_train.shape[0])
        np.random.shuffle(shuff)
        self.x_train = self.x_train[shuff,:]
        self.y_train = self.y_train[shuff]

        # Extract only 4's and 9's for validation set 
        self.x_valid = valid_set[0][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0],:]
        self.y_valid = valid_set[1][np.where(np.logical_or( valid_set[1]==4, valid_set[1] == 9))[0]]
        self.y_valid = np.array([1 if y == 9 else -1 for y in self.y_valid])
        
        # Extract only 4's and 9's for test set 
        self.x_test  = test_set[0][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0],:]
        self.y_test  = test_set[1][np.where(np.logical_or( test_set[1]==4, test_set[1] == 9))[0]]
        self.y_test = np.array([1 if y == 9 else -1 for y in self.y_test])
        
        f.close()

class AdaBoost:
    def __init__(self, n_learners=20, base=DecisionTreeClassifier(max_depth=2)):
        """
        Create a new adaboost classifier.
        
        Args:
            n_learners (int, optional): Number of weak learners in classifier.
            base (BaseEstimator, optional): Your general weak learner 

        Attributes:
            base (estimator): Your general weak learner 
            n_learners (int): Number of weak learners in classifier.
            alpha (ndarray): Coefficients on weak learners. 
            learners (list): List of weak learner instances. 
        """
        
        self.n_learners = n_learners 
        self.base = base
        self.alpha = np.zeros(self.n_learners)
        self.learners = []
        
    def fit(self, X_train, y_train):
        """
        Train AdaBoost classifier on data. Sets alphas and learners. 
        
        Args:
            X_train (ndarray): [n_samples x n_features] ndarray of training data   
            y_train (ndarray): [n_samples] ndarray of data 
        """

        w = np.ones(len(y_train))
        size = X_train.size
        #initialize all weights to 1/m (m is size of X_train)
        count = 0
        for item in w:
            item = item/size
            w[count] = item
            count += 1
        err = np.zeros(self.n_learners)
        #for n_learners create a weak learner, generate alpha, and update w
        #print(self.base.max_depth)

        w_sum = 1
        for n in range(self.n_learners):
            self.learners.append(clone(self.base))
            self.learners[n].fit(X_train, y_train, sample_weight=w)
            counter1 = 0
            for x in X_train:
                x = x.reshape(1, -1)
                if self.learners[n].predict(x)[0] == y_train[counter1]:
                    indicator = 0
                else:
                    indicator = 1
                    
                err[n] += w[counter1] * indicator
                counter1 += 1
            if err[n] == 0:
                err[n] = .0001
            err[n] = err[n]/sum(w)
            self.alpha[n] = 0.5 * np.log((1-err[n])/(err[n]))
            counter2 = 0
            for x in X_train:
                x = x.reshape(1, -1)
                w[counter2] = w[counter2] * np.exp((-self.alpha[n]) * y_train[counter2] * self.learners[n].predict(x)[0])
                w[counter2] = w[counter2]/w_sum
                counter2 += 1
            w_sum = sum(w)

       
                          
    def predict(self, X):
        """
        Adaboost prediction for new data X.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            
        Returns: 
            [n_samples] ndarray of predicted labels {-1,1}
        """

        h = 0

        for n in range(self.n_learners):
            h += self.alpha[n] * self.learners[n].predict(X)
        
        prediction = np.sign(h)
        return prediction
    
    def score(self, X, y):
        """
        Computes prediction accuracy of classifier.  
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            Prediction accuracy (between 0.0 and 1.0).
        """

        h = self.predict(X)
        score = accuracy_score(h,y)
        return score
    
    def staged_score(self, X, y):
        """
        Computes the ensemble score after each iteration of boosting 
        for monitoring purposes, such as to determine the score on a 
        test set after each boost.
        
        Args:
            X (ndarray): [n_samples x n_features] ndarray of data 
            y (ndarray): [n_samples] ndarray of true labels  
            
        Returns: 
            [n_learners] ndarray of scores 
        """

        h = 0
        stag_score = []

        for n in range(self.n_learners):
            h += self.alpha[n] * self.learners[n].predict(X)
            prediction = np.sign(h)
            stag_score.append(accuracy_score(prediction, y))

        return stag_score


def mnist_digit_show(flatimage, outname=None):

	import matplotlib.pyplot as plt

	image = np.reshape(flatimage, (-1,28))

	plt.matshow(image, cmap=plt.cm.binary)
	plt.xticks([])
	plt.yticks([])
	if outname: 
	    plt.savefig(outname)
	else:
	    plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='AdaBoost classifier options')
    parser.add_argument('--limit', type=int, default=-1,
                    help="Restrict training to this many examples")
    parser.add_argument('--n_learners', type=int, default=50,
                    help="Number of weak learners to use in boosting")
    args = parser.parse_args()

    data = FoursAndNines("../data/mnist.pkl.gz")

    clf = AdaBoost(n_learners=700, base=DecisionTreeClassifier(max_depth=2, criterion="entropy"))
    if (args.limit > 0):
        clf.fit(data.x_train[:args.limit], data.y_train[:args.limit])
        train_error = clf.staged_score(data.x_train[:args.limit], data.y_train[:args.limit])
        test_error = clf.staged_score(data.x_test[:args.limit], data.y_test[:args.limit])
        iterations = np.zeros(clf.n_learners)
        for n in range(clf.n_learners):
            iterations[n] = n+1
        plt.plot(train_error, iterations)
        plt.plot(test_error, iterations)
        plt.ylim([0,700])
        plt.xlim([0.80, 1.01])
        plt.show()
