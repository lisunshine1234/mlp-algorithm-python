from sklearn.tree import DecisionTreeClassifier


def run(x_train, y_train, x_test, y_test, criterion, splitter, max_depth, min_samples_split, min_samples_leaf,
        min_weight_fraction_leaf, max_features, random_state, max_leaf_nodes,
        min_impurity_decrease, min_impurity_split, class_weight, ccp_alpha
        ):
    reg = DecisionTreeClassifier(criterion=criterion,
                                 splitter=splitter,
                                 max_depth=max_depth,
                                 min_samples_split=min_samples_split,
                                 min_samples_leaf=min_samples_leaf,
                                 min_weight_fraction_leaf=min_weight_fraction_leaf,
                                 max_features=max_features,
                                 random_state=random_state,
                                 max_leaf_nodes=max_leaf_nodes,
                                 min_impurity_decrease=min_impurity_decrease,
                                 min_impurity_split=min_impurity_split,
                                 class_weight=class_weight,
                                 ccp_alpha=ccp_alpha).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'classes_': reg.classes_.tolist(),
            'feature_importances_': reg.feature_importances_.tolist(),
            'max_features_': int(reg.max_features_),
            'n_classes_': int(reg.n_classes_),
            'n_features_': int(reg.n_features_),
            'n_outputs_': int(reg.n_outputs_)}
