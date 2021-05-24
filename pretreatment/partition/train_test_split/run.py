from sklearn.model_selection import train_test_split


def run(array, test_size, train_size, random_state, shuffle):
    train, test = train_test_split(array, test_size=test_size, train_size=train_size,
                                   random_state=random_state,
                                   shuffle=shuffle)
    return {"train": train, "test": test}
