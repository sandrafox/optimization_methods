import matplotlib
import numpy as np
import sklearn.model_selection

from gradient_descent import gradient_descent, backtracking, save
from newton_method import newton
from scipy.special import expit
import pandas as pd
import matplotlib.pyplot as plt


class NumberOfSteps:
    def __init__(self, errors, steps):
        self.errors = errors
        self.steps = steps


class Logistic_regression:
    def __init__(self, alpha, solver, max_errors=100):
        assert solver in {'gradient', 'newton'}
        self.alpha = alpha
        self.w = None
        self.solver = solver
        self.max_errors = max_errors

    @staticmethod
    def __add_features(X):
        objects_count, _ = X.shape
        ones = np.ones((objects_count, 1))
        return np.hstack((X, ones))

    def fit(self, X, y, debug_iters=None, eps=1e-5):
        objects_count, features_count = X.shape
        assert y.shape == (objects_count,)
        X_r = Logistic_regression.__add_features(X)

        start_w = np.random.normal(loc=0., scale=1., size=features_count + 1)

        def Q(weights):
            predictions = np.matmul(X_r, weights)
            margins = predictions * y
            losses = np.logaddexp(0, -margins)
            return (np.sum(losses) / objects_count) + np.linalg.norm(weights) ** 2 * self.alpha / 2

        A = np.transpose(X_r * y.reshape((objects_count, 1)))

        def Q_grad(weights):
            predictions = np.matmul(X_r, weights)
            margins = predictions * y
            b = expit(-margins)
            grad = -np.matmul(A, b) / objects_count
            return grad + self.alpha * weights

        def Q_hess(weights):
            predictions = np.matmul(X_r, weights)
            margins = predictions * y
            C = np.transpose(X_r * expit(-margins).reshape((objects_count, 1)))
            D = X_r * expit(margins).reshape((objects_count, 1))
            hess = np.matmul(C, D) / objects_count
            return hess + self.alpha * np.eye(features_count + 1)

        if self.solver == 'gradient':
            trace = gradient_descent(Q, Q_grad, start_w, backtracking, 'grad', eps=eps)
            self.w = trace[-1]
            return NumberOfSteps(0, len(trace))
        else:
            errors = 0
            while True:
                try:
                    if errors >= self.max_errors:
                        self.w = start_w
                        return NumberOfSteps(errors, -1)
                    else:
                        trace = newton(Q, Q_grad, Q_hess, start_w, 'delta', eps=eps, cho=True)
                        self.w = trace[-1]
                        return NumberOfSteps(errors, len(trace))
                except ArithmeticError:
                    errors += 1
                    start_w = np.random.normal(loc=0., scale=1., size=features_count + 1)

    def predict(self, X):
        X_r = Logistic_regression.__add_features(X)
        return np.sign(np.matmul(X_r, self.w)).astype(int)


def read_dataset(path):
    data = pd.read_csv(path)
    X = data.iloc[:,:-1].values
    y = data.iloc[:, -1].apply(lambda c: 1 if c == 'P' else -1).values
    return X, y


def calc_f_score(X, y, alpha, solver):
    n_splits = 5
    cv = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True)
    mean_f_score = 0.0
    for train_indexes, test_indexes in cv.split(X):
        X_train = X[train_indexes]
        X_test = X[test_indexes]
        y_train = y[train_indexes]
        y_test = y[test_indexes]

        classifier = Logistic_regression(alpha, solver)
        classifier.fit(X_train, y_train)
        y_pred = classifier.predict(X_test)

        tp = np.sum((y_pred == 1) & (y_test == 1))
        fp = np.sum((y_pred == 1) & (y_test != 1))
        fn = np.sum((y_pred != 1) & (y_test == 1))

        if tp != 0:
            precision = tp / (tp + fp)
            recall = tp / (tp + fn)
            f_score = 2 * precision * recall / (precision + recall)
            mean_f_score += f_score
    return mean_f_score / n_splits


def get_best_param(X, y, solver):
    best_alpha = None
    max_f_score = -1
    for alpha in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.]:
        cur_f_score = calc_f_score(X, y, alpha, solver)
        print('alpha =', alpha, 'f-score =', cur_f_score)
        if cur_f_score > max_f_score:
            max_f_score = cur_f_score
            best_alpha = alpha
    return best_alpha, max_f_score


def draw(clf, X, ans, step_x, step_y, name):
    x_min, y_min = np.amin(X, axis=0)
    x_max, y_max = np.amax(X, axis=0)
    x_min -= step_x
    x_max += step_x
    y_min -= step_y
    y_max += step_y

    xx, yy = np.meshgrid(np.arange(x_min, x_max, step_x), np.arange(y_min, y_max, step_y))

    zz = clf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    plt.figure(figsize=(12, 12))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)

    x0, y0 = X[ans != 1].T
    x1, y1 = X[ans == 1].T

    plt.pcolormesh(xx, yy, zz, cmap=matplotlib.colors.ListedColormap(['#FFAAAA', '#AAAAFF']))
    plt.scatter(x0, y0, color='red', s=100)
    plt.scatter(x1, y1, color='blue', s=100)
    plt.xlabel('x')
    plt.ylabel('y')
    save(name, "png")
    plt.show()


def process_with_solver(X, y, solver, step_x, step_y):
    best_alpha, max_f_score = get_best_param(X, y, solver)
    print('Best params:', best_alpha, max_f_score)
    best_classifier = Logistic_regression(best_alpha, solver)
    number_of_steps = best_classifier.fit(X, y)
    if solver == 'newton':
        print('errors =', number_of_steps.errors, 'steps =', number_of_steps.steps)
    else:
        print('steps =', number_of_steps.steps)
    draw(best_classifier, X, y, step_x, step_y, 'classification_' + solver)


