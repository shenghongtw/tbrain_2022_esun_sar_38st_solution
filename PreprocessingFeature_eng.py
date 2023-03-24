import numpy as np
def feature_eng(data, data_name):
    data = data.copy()
    if data_name == 'cdtx':
        data.loc[:, 'country_And_cur_type'] = data.country.astype('str') + data.cur_type.astype('str')
        cat_col = ['country', 'cur_type', 'country_And_cur_type']
    if data_name == 'dp':
        #將fiscTxid的nan補新的類別值29
        data.fiscTxId.fillna(-1)
        #將txbranch的nan補新的類別值371
        data.txbranch.fillna(-1)
        #將類別轉換成數值
        data.loc[:, 'debit_credit'] = data.loc[:, 'debit_credit'].map({'CR':0, 'DB':1})
        #data.drop(columns=['exchg_rate'], inplace=True)
        #將tx_type與info_asset_code的組合轉成新特徵
        data.loc[:, 'tx_type_and_info_asset_code'] = \
        data.tx_type.astype('int').astype('str') + data.info_asset_code.astype('int').astype('str')
        #將fiscTxId與txbranch的組合轉成新特徵
        data.loc[:, 'fiscTxId_and_txbranch'] = \
        data.fiscTxId.astype('int').astype('str') + data.txbranch.astype('int').astype('str')
        #tx_type、info_asset_code、fiscTxId、txbranch的組合
        data.loc[:, 'tx_type_info_asset_code_fiscTxId_and_txbranch'] = \
        data.fiscTxId.astype('int').astype('str') + data.txbranch.astype('int').astype('str') \
        + data.fiscTxId.astype('int').astype('str') + data.txbranch.astype('int').astype('str')
        #fiscTxId,txbranch,crossBank,ATM組合
        data.loc[:, 'fiscTxId_txbranch_crossBank_ATM'] = \
        data.fiscTxId.astype('int').astype('str') + data.txbranch.astype('int').astype('str')\
        + data.cross_bank.astype('int').astype('str') + data.ATM.astype('int').astype('str')
        #tx_type, info_asset_code, crossBank, ATM組合
        data.loc[:, 'tx_type_&_info_asset_code_&_crossBank_&_ATM'] = \
        data.fiscTxId.astype('int').astype('str') + data.txbranch.astype('int').astype('str') \
        + data.cross_bank.astype('int').astype('str') + data.ATM.astype('int').astype('str')
        
    data.fillna(0, inplace=True)
    float64_col = data.select_dtypes(include=['float64']).columns.values.tolist()
    int64_col = data.select_dtypes(include=['int64']).columns.values.tolist()
    data.loc[:, float64_col] = data.loc[:, float64_col].astype('float16')
    data.loc[:, int64_col] = data.loc[:, int64_col].astype('int16')
    data.replace([np.inf, -np.inf], 0, inplace=True)
    return data