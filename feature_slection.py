import pandas as pd
from sklearn.metrics import make_scorer
from mlxtend.feature_selection import SequentialFeatureSelector as SFS

def feature_selection(training_set, label_name, model, Kfold, scoring_func):
    training_set.loc[:,['AGE', 'occupation_code', 'risk_rank']] = training_set[['AGE', 'occupation_code', 'risk_rank']].astype('category')
    features = training_set.columns.values.tolist()
    features.remove('cust_id')
    features.remove('date')
    features.remove('is_test')
    features.remove('sar_flag')
    features.remove('alert_key')

    sfs = SFS(
            model, 
            k_features=50, 
            forward=True,
            floating=True, 
            scoring = scoring_func,
            cv=list(Kfold.split(training_set, training_set[label_name])),
            n_jobs=-1,
            verbose=2
           )
    sfs.fit(training_set[features], training_set[label_name])
    print('CV Score:')
    sfs_data = pd.DataFrame(sfs.get_metric_dict()).T
    sfs_data['scores'] = sfs_data['cv_scores'].apply(lambda x:x.mean())
    best_feature_set = list(sfs_data[sfs_data['scores']==sfs_data['scores'].max()]['feature_names'].values[0])
    return best_feature_set

