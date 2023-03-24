import numpy as np
import pandas as pd
def cv_inference(training_set, testing_set, lgb_model, features, label_name, kfold, num_fold, tbrain_metric_func):
    test_prediction = np.zeros(len(testing_set))
    val_overall_score = np.zeros(num_fold)
    for fold, (train_index, val_index) in enumerate(kfold.split(training_set, training_set[label_name])):
        x_train, x_val = training_set[features].iloc[train_index], training_set[features].iloc[val_index]
        y_train, y_val = training_set[label_name].iloc[train_index], training_set[['alert_key', label_name]].iloc[val_index]
        model = lgb_model.fit(x_train, y_train)
        val_pred = model.predict(x_val)
        test_pred = model.predict(testing_set[features])
        test_prediction += test_pred / num_fold
        val_score = tbrain_metric_func(y_val, val_pred)
        val_overall_score[fold] = val_score
    print(f'overall_score: {np.average(val_overall_score)}')
    test_df = pd.DataFrame({'alert_key': testing_set.alert_key, 'probability': test_prediction})
    test_df.sort_values(by=['probability'], ascending=False, inplace=True)
    test_df.to_csv('test_prediction.csv', index=False)