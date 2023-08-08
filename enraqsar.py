def enraqsar(df, x_train, x_test, y_train, y_test, n_top, weight, name):
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    import enraqsar as en
    loo = LeaveOneOut()
    scaler = StandardScaler()
    x_train_scale = scaler.fit_transform(x_train)
    x_test_scale  = scaler.transform(x_test)
    from joblib import load
    lasso_cv_reduce = load('qsar.joblib')
    qsar_train = lasso_cv_reduce.predict(x_train_scale)
    qsar_cv    = cross_val_predict(lasso_cv_reduce, x_train_scale, y_train, cv=loo)
    qsar_test  = lasso_cv_reduce.predict(x_test_scale)
    qsar_train = pd.DataFrame(qsar_train, columns=['y_pred']).set_index(x_train.index)
    qsar_cv    = pd.DataFrame(qsar_cv, columns=['y_pred']).set_index(x_train.index)
    qsar_test  = pd.DataFrame(qsar_test, columns=['y_pred']).set_index(x_test.index)
    from enreadacross import readacross_tanimoto_weight, readacross_tanimoto_weight_apf
    ra_ec_train = readacross_tanimoto_weight(df, x_train, y_train, x_train,     n_top, smiles_col='canonical_smiles',weight_power=weight)
    ra_ec_test  = readacross_tanimoto_weight(df, x_train, y_train, x_test,      n_top, smiles_col='canonical_smiles',weight_power=weight)
    ra_ap_train = readacross_tanimoto_weight_apf(df, x_train, y_train, x_train, n_top, smiles_col='canonical_smiles',weight_power=weight)
    ra_ap_test  = readacross_tanimoto_weight_apf(df, x_train, y_train, x_test,  n_top, smiles_col='canonical_smiles',weight_power=weight)
    #get y_pred ra
    ra_ec_train = ra_ec_train['y_pred']
    ra_ec_test  = ra_ec_test['y_pred']
    ra_ap_train = ra_ap_train['y_pred']
    ra_ap_test  = ra_ap_test['y_pred']
    #combine train
    raqsar_train = pd.concat([ra_ec_train, ra_ap_train, qsar_train], axis=1)
    raqsar_test  = pd.concat([ra_ec_test,  ra_ap_test,  qsar_test], axis=1)
    #ensemble
    rf = load(os.path.join('qraqsar.joblib'))
    qsar_result = {}
    loo = LeaveOneOut()
    y_pred_test  = rf.predict(raqsar_test)
    #calculate metrics
    mae_test  = mean_absolute_error(y_test , y_pred_test)
    #add y values and metrics to dictionary
    qsar_result['raqsar_test']  = raqsar_test
    qsar_result['y_pred_test']  = y_pred_test
    qsar_result['mae_test']     = mae_test
    print('MAE_{Ext}: %.2f'%  mae_test)
    #save results
    qsar_result = pd.DataFrame(qsar_result)
    return qsar_result

#Welcome message
print('Welcome to the Ensemble Read-Across QSAR model for predicting the skin toxicity of chemicals.')
print('This model is based on the following publication:')

#Description
print('Description:')
print('')