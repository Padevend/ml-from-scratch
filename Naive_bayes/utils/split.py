from numpy import random, arange

def train_test_split(X, y, test_size=0.2, random_state=None, shuffle=True):
    n = len(X)
    indices = arange(n)

    if random_state is not None:
        random.seed(random_state)

    # mellange des donnees
    if shuffle:
        random.shuffle(indices)

    # separateur
    split = int(n * (1 - test_size))

    train_idx = indices[:split]
    test_idx = indices[split:]

    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]
