import optuna

params_to_tune = {
    "num_leaves": optuna.distributions.IntDistribution(40, 60),
    "min_child_samples" : optuna.distributions.IntDistribution(10, 30),
    "max_depth": optuna.distributions.IntDistribution(5, 20),
    "max_bin" : optuna.distributions.IntDistribution(300, 500),
    "learning_rate": optuna.distributions.FloatDistribution(0.01, 0.1),
    "n_estimators": optuna.distributions.IntDistribution(50, 250),
    "subsample": optuna.distributions.FloatDistribution(0, 1),
    "colsample_bytree": optuna.distributions.FloatDistribution(0, 1),
    "reg_alpha": optuna.distributions.FloatDistribution(1e-9, 1.0),
    "reg_lambda": optuna.distributions.FloatDistribution(1e-9, 5.0),
}

def params_tuning(training_data, features, model, label_name, kfold, scoring_func):
    optuna_search = optuna.integration.OptunaSearchCV(
        model, params_to_tune, 
        n_trials=None, 
        timeout=20*60,
        cv=list(kfold.split(training_data, training_data[label_name])),
        scoring=scoring_func,
        verbose=-1
    )
    optuna_search.fit(training_data[features], training_data[label_name])
    return optuna_search.best_params_