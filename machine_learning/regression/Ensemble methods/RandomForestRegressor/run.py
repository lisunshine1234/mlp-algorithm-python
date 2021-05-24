from sklearn.ensemble import RandomForestRegressor


def run(x_train, y_train, x_test, y_test,
        n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease,
        min_impurity_split, bootstrap, oob_score, n_jobs, random_state, verbose, warm_start, ccp_alpha, max_samples
        ):
    reg = RandomForestRegressor(n_estimators=n_estimators,
                                criterion=criterion,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_features=max_features,
                                max_leaf_nodes=max_leaf_nodes,
                                min_impurity_decrease=min_impurity_decrease,
                                min_impurity_split=min_impurity_split,
                                bootstrap=bootstrap,
                                oob_score=oob_score,
                                n_jobs=n_jobs,
                                random_state=random_state,
                                verbose=verbose,
                                warm_start=warm_start,
                                ccp_alpha=ccp_alpha,
                                max_samples=max_samples).fit(x_train, y_train)
    oob_score_ = None
    oob_prediction_ = None
    if oob_score:
        oob_score_ = reg.oob_score_
        oob_prediction_ = reg.oob_prediction_.tolist()
    return {'train_predict': reg.predict(x_train).tolist(),
            'test_predict': reg.predict(x_test).tolist(),
            'train_score': reg.score(x_train, y_train),
            'test_score': reg.score(x_test, y_test),
            'feature_importances_': reg.feature_importances_.tolist(),
            'n_features_': reg.n_features_,
            'n_outputs_': reg.n_outputs_,
            'oob_score_': oob_score_,
            'oob_prediction_': oob_prediction_
            }
