import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import make_scorer
from PreprocessingFeature_eng import PreprocessingFeature_eng
from concat_data import concat_all_data
from feature_slection import feature_selection
from metric import tbrain_metric
from params_tuning import params_tuning
from inference import cv_inference

class CFG:
    seed = 42
    n_folds = 5
    label = 'sar_flag'
    boosting_type = 'dart'

params = {
        'objective': 'binary',
        'boosting': CFG.boosting_type,
        'seed': CFG.seed,
        'n_jobs': 2,
        'verbosity': -1,
        }

def main():
    #讀取資料
    cust_label_data = pd.read_csv('cust_label_data.csv')
    data_cdtx = pd.read_csv('public_train_x_cdtx0001_full_hashed.csv')
    data_remit = pd.read_csv('public_train_x_remit1_full_hashed.csv')
    data_dp = pd.read_csv('public_train_x_dp_full_hashed.csv')
    data_dp.rename(columns = {'tx_date': 'date'}, inplace=True)
    data_remit.rename(columns = {'trans_date': 'date'}, inplace=True)

    #preprocessing&feature engineering
    data_cdtx = PreprocessingFeature_eng(data_cdtx, 'cdtx')
    data_dp = PreprocessingFeature_eng(data_dp, 'dp')
    data_remit = PreprocessingFeature_eng(data_remit, 'remit')

    #concat all data
    concat_data = concat_all_data(cust_label_data, data_cdtx, data_dp, data_remit, far_day=7, near_day=5, topk=4)
    training_set = concat_data[concat_data.is_test==0]
    testing_set = concat_data[concat_data.is_test==1]

    #kfold cross validation
    kfold = StratifiedKFold(n_splits = CFG.n_folds, shuffle = True, random_state = CFG.seed)

    #making metric function
    tbrain_metric_func = make_scorer(tbrain_metric, greater_is_better=True, needs_proba=True)

    #Building lightGBM model 
    lgb_model = lgb.LGBMClassifier(**params)

    #feature selection
    best_features = feature_selection(training_set, CFG.label, lgb_model, kfold, tbrain_metric_func)

    #parameter tuning
    best_params = params_tuning(training_set, best_features, lgb_model, kfold, tbrain_metric_func)
    lgb_model = lgb.LGBMClassifier(**params, **best_params)

    #cross validation inference
    cv_inference(training_set, testing_set, lgb_model, best_features, CFG.label)
if __name__ == '__main__':
    main()




