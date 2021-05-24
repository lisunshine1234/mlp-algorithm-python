from sklearn.tree import DecisionTreeRegressor


def run(x_train, y_train, x_test, y_test, criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features,
        random_state, max_leaf_nodes,
        min_impurity_decrease, min_impurity_split, ccp_alpha
        ):
    reg = DecisionTreeRegressor(criterion=criterion,
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
                                ccp_alpha=ccp_alpha).fit(x_train, y_train)
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'feature_importances_': reg.feature_importances_.tolist(),
            'max_features_': reg.max_features_,
            'n_features_': reg.n_features_,
            'n_outputs_': reg.n_outputs_
            }
