import pandas as pd
import numpy as np
from tqdm import tqdm

#取得類別特徵
def get_near_N_day_topk_cat_feature(data, concat_data, cat_feature, near_day, topk):
    c_id = data.cust_id
    c_date = data.date
    tmp_cat_near_day_data = concat_data[(concat_data.cust_id==c_id) & \
                    (concat_data.date<=c_date) & (concat_data.date >= c_date-near_day)]
    if not tmp_cat_near_day_data.empty:
        for c_feature in cat_feature:
            c_feature_counts = tmp_cat_near_day_data[c_feature].value_counts().index.values
            for k in range(0, topk):
                col_name = 'top' + str(k+1)+ '_' + c_feature #ex.top1_country
                try:
                    data[col_name] = c_feature_counts[k]
                except:
                    data[col_name] = 0
    return data

#取得數值特徵
def get_near_N_day_numerical_feature(data, concat_data, num_feature, far_day, near_day):
    c_id = data.cust_id
    c_date = data.date
    tmp_num_near_day_data = concat_data[(concat_data.cust_id==c_id) & \
                    (concat_data.date<=c_date) & (concat_data.date >= c_date-near_day)]
    tmp_far_d_data = concat_data[(concat_data.cust_id==c_id) & \
                    (concat_data.date<c_date-near_day)&(concat_data.date >= c_date-far_day)]
    
    if not tmp_num_near_day_data.empty:
        near_dayata = tmp_num_near_day_data. \
            groupby('cust_id')[num_feature].agg(['sum', 'mean', 'std', 'min', 'max', 'count']) 
        far_data = tmp_far_d_data. \
            groupby('cust_id')[num_feature].agg(['sum', 'mean', 'std', 'min', 'max', 'count'])
        near_dayata.columns = ['_'.join(x) for x in near_dayata.columns]
        far_data.columns = ['_'.join(x) for x in far_data.columns]
        divide_data = near_dayata / far_data
        divide_data.columns = ['divide_' + divide_col for divide_col in divide_data.columns]
        data = pd.concat([data, divide_data.squeeze(), near_dayata.squeeze()])
    return data

#將多個資料以label data為基準合併
def get_concat_data(cust_label_data, cdtx_data, dp_CR_data, dp_DB_data, remit_data, cdtx_cat_col, cdtx_num_col, dp_CR_cat_col, dp_CR_num_col, dp_DB_cat_col, dp_DB_num_col, remit_cat_col, remit_num_col, far_day, near_day, topk):
    label_data = cust_label_data.copy()
    cdtx = cdtx_data.copy()
    dp_CR = dp_CR_data.copy()
    dp_DB = dp_DB_data.copy()
    remit = remit_data.copy()
    #num_feature
    #cdtx
    # label_data = label_data.apply(get_near_N_day_num_feature, args=(cdtx, cdtx_num_col, near_day), axis=1)
    label_data = label_data.apply(get_near_N_day_numerical_feature, args=(cdtx, cdtx_num_col, far_day, near_day), axis=1)        
    print('cdtx num done')
    #dp_CR
    # label_data = label_data.apply(get_near_N_day_num_feature, args=(dp_CR, dp_CR_num_col, near_day), axis=1)
    label_data = label_data.apply(get_near_N_day_numerical_feature, args=(dp_CR, dp_CR_num_col, far_day, near_day), axis=1) 
    print('dp_CR num done')
    #dp_DB
    # label_data = label_data.apply(get_near_N_day_num_feature, args=(dp_DB, dp_DB_num_col, near_day), axis=1)
    label_data = label_data.apply(get_near_N_day_numerical_feature, args=(dp_DB, dp_DB_num_col, far_day, near_day), axis=1)
    print('dp_DB num done')
    #remit
    # label_data = label_data.apply(get_near_N_day_num_feature, args=(remit, remit_num_col, near_day), axis=1)
    label_data = label_data.apply(get_near_N_day_numerical_feature, args=(remit, remit_num_col, far_day, near_day), axis=1)   
    print('num_feature done')     

    #cat_feature
    label_data = label_data.apply(get_near_N_day_topk_cat_feature, \
                                  args=(cdtx, cdtx_cat_col, near_day, topk), axis=1)
    label_data = label_data.apply(get_near_N_day_topk_cat_feature, \
                                  args=(dp_CR, dp_CR_cat_col, near_day, topk), axis=1)
    label_data = label_data.apply(get_near_N_day_topk_cat_feature, \
                                  args=(dp_DB, dp_DB_cat_col, near_day, topk), axis=1)
    label_data = label_data.apply(get_near_N_day_topk_cat_feature, \
                                  args=(remit, remit_cat_col, near_day, topk), axis=1)
    print('cat_feature done')

    label_data.fillna(0, inplace=True)
    float64_col = label_data.select_dtypes(include=['float64']).columns.values.tolist()
    int64_col = label_data.select_dtypes(include=['int64']).columns.values.tolist()
    label_data.loc[:, float64_col] = label_data.loc[:, float64_col].astype('float16')
    label_data.loc[:, int64_col] = label_data.loc[:, int64_col].astype('int16')
    label_data.replace([np.inf, -np.inf], 0, inplace=True)
    return label_data



def concat_all_data(cust_label_data, data_cdtx, data_dp, data_remit, far_day, near_day, topk):
    #將dp依debit_cridit分開
    data_dp_CR = data_dp[data_dp.debit_credit==0]
    data_dp_DB = data_dp[data_dp.debit_credit==1]

    # Rename columns in data_dp2_CR
    data_dp_CR.columns = data_dp_CR.add_prefix('CR_').columns
    # Rename columns in data_dp2_DB
    data_dp_DB.columns = data_dp_DB.add_prefix('DB_').columns
    # Rename 'CR_cust_id' and 'CR_tx_date' columns in data_dp2_CR
    data_dp_CR.rename(columns={'CR_cust_id': 'cust_id', 'CR_date': 'date'}, inplace=True)
    # Rename 'DB_cust_id' and 'DB_tx_date' columns in data_dp2_DB
    data_dp_DB.rename(columns={'DB_cust_id': 'cust_id', 'DB_date': 'date'}, inplace=True)

    #catgorical_col
    cdtx_cat_col = ['country', 'cur_type', 'country_And_cur_type']
    # 使用列表推導的方式，只保留 data_dp2_CR 的列名，除了第一列和 'CR_debit_credit' 列
    remove_CR_col = ['date', 'CR_tx_amt','CR_cross_bank',\
                'CR_ATM','CR_exchg_rate_is_one']
    remove_DB_col = ['date', 'DB_tx_amt','DB_cross_bank',\
                'DB_ATM','DB_exchg_rate_is_one']
    dp_CR_cat_col = [col for col in data_dp_CR.columns[2:] if col not in remove_CR_col]
    # 使用列表推導的方式，只保留 data_dp2_DB 的列名，除了第一列和 'DB_debit_credit' 列
    dp_DB_cat_col = [col for col in data_dp_DB.columns[2:] if col not in remove_DB_col]
    remit_cat_col = ['trans_no']

    #num_col
    cdtx_num_col = ['amt']
    dp_CR_num_col = ['CR_tx_amt']
    dp_DB_num_col = ['DB_tx_amt']
    remit_num_col = ['trade_amount_usd']

    data = get_concat_data \
        (cust_label_data, data_cdtx, data_dp_CR, data_dp_DB, data_remit,  cdtx_cat_col, cdtx_num_col, dp_CR_cat_col, \
        dp_CR_num_col, dp_DB_cat_col, dp_DB_num_col, remit_cat_col, remit_num_col, far_day=far_day, near_day=near_day, topk=topk)
    data.alert_key = cust_label_data.alert_key
    data.to_csv(f'concat_data_{far_day}_{near_day}_{topk}.csv',index=False)
    return data
