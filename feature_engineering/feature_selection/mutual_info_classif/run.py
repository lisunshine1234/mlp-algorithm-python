import numpy as np
from sklearn.feature_selection import mutual_info_classif

def run(x, y, discrete_features, n_neighbors, copy, random_state):
    if type(discrete_features) is str:
        discrete_features = discrete_features.lower()
        if discrete_features == 'true':
            discrete_features = True
        elif discrete_features == 'false':
            discrete_features = False
        elif discrete_features.__contains__('[') or discrete_features.__contains__('(') or discrete_features.__contains__(','):
            discrete_features = discrete_features.replace('true', 'True').replace('false', 'False')
            discrete_features = eval(discrete_features)

    elif type(discrete_features) is list:
        discrete_features = np.array(discrete_features)

    sk = mutual_info_classif(x, y,
                             discrete_features=discrete_features,
                             n_neighbors=n_neighbors,
                             copy=copy,
                             random_state=random_state)

    return {"mi": sk.tolist()}
