"""
Contains all the necessary functions to find the best parameters for each
algorithm, fit the model and save the grid search results and best estimator.

"""
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
from helpers import get_abspath, save_pickled_model
from clf_knn import KNN
from clf_decision_tree import DT
from clf_boosting import ADA
from clf_neural_network import MLP
from clf_svm import SVM_RBF, SVM_PLY
import pandas as pd
import timeit
import argparse


def split_data(df, test_size=0.3, seed=42):
    """Prepares a data frame for model training and testing by converting data
    to Numpy arrays and splitting into train and test sets.

    Args:
        df (pandas.DataFrame): Source data frame.
        test_size (float): Size of test set as a percentage of total samples.
        seed (int): Seed for random state in split.
    Returns:
        X_train (numpy.Array): Training features.
        X_test (numpy.Array): Test features.
        y_train (numpy.Array): Training labels.
        y_test (numpy.Array): Test labels.

    """
    # convert data frame to Numpy array and split X and y
    X_data = df.drop(columns='class').as_matrix()
    y_data = df['class'].as_matrix()

    # split into train and test sets, ensuring that composition of classes in
    # original dataset is maintained in the splits
    X_train, X_test, y_train, y_test = train_test_split(
        X_data, y_data, test_size=test_size, random_state=seed,
        stratify=y_data
    )

    return X_train, X_test, y_train, y_test


def balanced_f1(labels, predictions):
    """Modifies the standard F1 scoring function to account for potential
    imbalances in class distributions.

    Args:
        labels (numpy.array): Actual class labels.
        predictions (numpy.array): Predicted class labels.
    Returns:
        Modified F1 scoring function

    """
    return f1_score(labels, predictions, average='weighted')


def balanced_accuracy(labels, predictions):
    """Modifies the standard accuracy scoring function to account for
    potential imbalances in class distributions.

    Args:
        labels (numpy.array): Actual class labels.
        predictions (numpy.array): Predicted class labels.
    Returns:
        Modified accuracy scoring function.

    """
    weights = compute_sample_weight('balanced', labels)
    return accuracy_score(labels, predictions, sample_weight=weights)


def train_model(X_train, y_train, clf, scorer, n_jobs=2, cv=5):
    """Trains model using GridSearchCV and returns object containing results.

    Args:
        X_train (numpy.Array): Training features.
        y_train (numpy.Array): Training labels.
        clf (object): Classifier object with pipeline and params attributes.
        scorer (object): Scoring function.
        n_jobs (int): Number of jobs to run in parallel.
        cv (int): Number of folds in cross-validation splitting strategy.
    Returns:
        grid (GridSearchCV object)

    """
    # get pipeline and params from classifier object
    pipeline = clf.pipeline
    params = clf.params

    # generate grid search object
    grid = GridSearchCV(
            estimator=pipeline,
            n_jobs=n_jobs, param_grid=params,
            cv=cv, scoring=scorer, refit=True,
            return_train_score=False)

    # initiate grid search
    grid.fit(X_train, y_train)

    return grid


def save_train_results(grid, data_name, clf_name):
    """Saves grid search cross-validation results and pickles the entire
    pipeline and best estimator.

    Args:
        grid (GridSearchCV object): Trained grid search object.
        data_name (str): Name of data set algorithm was trained on.
        clf_name (str): Type of algorithm.
        path (str): Target path for results/pickled model.

    """
    # get cross-validation results and best estimator
    results = pd.DataFrame(grid.cv_results_)
    best_clf = grid.best_estimator_

    # save cross-validation results as CSV
    parentdir = 'models'
    target = '{}/{}'.format(parentdir, clf_name)
    resfile = get_abspath('{}_cv_results.csv'.format(data_name), target)
    results.to_csv(resfile, index=False)

    # save grid search object and best estimator as pickled model files
    gridpath = get_abspath('{}_grid.pkl'.format(data_name), target)
    bestpath = get_abspath('{}_best_estimator.pkl'.format(data_name), target)
    save_pickled_model(grid, gridpath)
    save_pickled_model(best_clf, bestpath)


if __name__ == '__main__':
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--dataset', nargs='?',
                    help='An optional integer positional argument')
    args = parser.parse_args()
    print("Argument values:")
    print(args.dataset)
    # set scoring function
    scorer = make_scorer(balanced_accuracy)

    contraceptive_path = get_abspath('contraceptive.csv', 'data')
    contraceptive_df = pd.read_csv(contraceptive_path)
    abalone_path = get_abspath('abalone.csv', 'data')
    abalone_df = pd.read_csv(abalone_path)

    dfs = {
        'contraceptive': contraceptive_df,
        'abalone': abalone_df,
    }

    if args.dataset:
        dnames = [args.dataset]
    else:
        dnames = ['contraceptive', 'opts_digit', 'abalone']

    # instantiate estimators
    estimators = {
                  'DT': DT,
                  'Boosting': ADA,
                  'ANN': MLP,
                  'SVM_RBF': SVM_RBF,
                  'KNN': KNN,
                  'SVM_PLY': SVM_PLY,
                  }

    # begin training loop
    for df in dnames:
        if df == 'opts_digit':
            train_df = pd.read_csv('../data/optdigits_train.csv', header=None)
            y_train = train_df.iloc[:, -1:].as_matrix().flatten()
            X_train = train_df.iloc[:, :-1].as_matrix()

            test_df = pd.read_csv('../data/optdigits_test.csv', header=None)
            y_test = test_df.iloc[:, -1:].as_matrix().flatten()
            X_test = test_df.iloc[:, :-1].as_matrix()

        else:
            X_train, X_test, y_train, y_test = split_data(dfs[df])
        for name, estimator in estimators.iteritems():
            clf_name = name
            clf = estimator()

            print('Training: {}\nData: {}'.format(clf_name, df))
            # start timing
            start_time = timeit.default_timer()

            # run grid search
            grid = train_model(X_train, y_train, clf=clf, scorer=scorer, cv=4)

            # end timing
            end_time = timeit.default_timer()
            elapsed = end_time - start_time

            # save training results
            save_train_results(grid, df, clf_name)
            print('{} trained in {:f} seconds\n'.format(clf_name, elapsed))
