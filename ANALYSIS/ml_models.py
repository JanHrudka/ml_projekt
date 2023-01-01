from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier


class MLModels:

    def svc(self, model_kwargs={}):

        classifier = SVC(**model_kwargs)

        return classifier

    def logistic_regression(self, model_kwargs={}):

        classifier = LogisticRegression(**model_kwargs)

        return classifier

    def decision_tree(self, model_kwargs={}):

        classifier = DecisionTreeClassifier(**model_kwargs)

        return classifier

    def random_forest(self, model_kwargs={}):

        classifier = RandomForestClassifier(**model_kwargs)

        return classifier

    def knn(self, model_kwargs={}):

        classifier = KNeighborsClassifier(**model_kwargs)

        return classifier

    def mlp(self, model_kwargs={}):

        classifier = MLPClassifier(**model_kwargs)

        return classifier

    def gnb(self, model_kwargs={}):

        classifier = GaussianNB(**model_kwargs)

        return classifier

    def ada(self, model_kwargs={}):

        classifier = AdaBoostClassifier(**model_kwargs)

        return classifier
