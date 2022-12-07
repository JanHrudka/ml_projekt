from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

class MLModels:

    def __init__(self, X_train, X_test, y_train, y_test):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test

    def svc(self, model_kwargs):

        classifier = make_pipeline(
            StandardScaler(),
            SVC(**model_kwargs)
        )
        classifier.fit(self.X_train, self.y_train)
        y_predicted = classifier.predict(self.X_test)

        return y_predicted

    def logistic_regression(self, model_kwargs):

        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(**model_kwargs)
        )
        classifier.fit(self.X_train, self.y_train)
        y_predicted = classifier.predict(self.X_test)

        return y_predicted

    def decision_tree(self, model_kwargs):
        classifier = make_pipeline(
            StandardScaler(),
            clf = RandomForestClassifier(**model_kwargs)
        )
        classifier.fit(self.X_train, self.y_train)
        y_predicted = classifier.predict(self.X_test)

        return y_predicted

    def random_forest(self, model_kwargs):
        classifier = make_pipeline(
            StandardScaler(),
            clf = RandomForestClassifier(**model_kwargs)
        )
        classifier.fit(self.X_train, self.y_train)
        y_predicted = classifier.predict(self.X_test)

        return y_predicted
