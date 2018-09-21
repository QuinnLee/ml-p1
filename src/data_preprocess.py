import pandas as pd

import os
from helpers import save_dataset
from sklearn.datasets import load_breast_cancer, load_digits, fetch_mldata


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


if __name__ == '__main__':
    print('Processing \n')
    tdir = 'data'
    mldata_dir = os.path.join(os.getcwd(), os.pardir, tdir)
    mnist = fetch_mldata('MNIST original', data_home=mldata_dir)
    y = pd.Series(mnist.target).astype('int')
    X = pd.DataFrame(mnist.data)
    X.loc[:, 'class'] = y
    X = (X.loc[X['class'].isin([1, 3, 5])]).\
        groupby('class', group_keys=False).\
        apply(lambda x: x.sample(min(len(x), 1000)))

    save_dataset(X, 'digits.csv', sep=',', subdir=tdir)

    df = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/cmc/cmc.data',  # noqa
        header=None
    )

    df.rename(columns={9: 'class'}, inplace=True)
    save_dataset(df, 'contraceptive.csv', sep=',', subdir=tdir)

    df = pd.read_csv(
        'http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data',  # noqa
        header=None
    )

    def map_values(value):
        if value < 9:
            return 0
        if value is 9 or value is 10:
            return 1
        return 2

    df[9] = df[8].map(map_values)
    df = pd.get_dummies(df, prefix=['0'])
    df.drop(columns=8, inplace=True)
    df.rename(columns={9: 'class'}, inplace=True)
    df['class'] = pd.Categorical(pd.factorize(df['class'])[0]+1)

    save_dataset(df, 'abalone.csv', sep=',', subdir=tdir)
