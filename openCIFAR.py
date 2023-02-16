import pickle


def unpickle(file):
    with open(file, 'rb') as fp:
        dicts = pickle.load(fp, encoding="latin1")
    return dicts

