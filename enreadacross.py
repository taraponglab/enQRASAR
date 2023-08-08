def calculate_ecfp(df, smiles_col, radius=3, nBits=2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem as Chem   
    def get_ecfp(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
        except:
            return None

    df['ECFP'] = df[smiles_col].apply(get_ecfp)
    
    return df

def calculate_apf(df, smiles_col, nBits=2048):
    from rdkit import Chem
    from rdkit.Chem import AllChem as Chem

    def get_apf(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            return Chem.GetHashedAtomPairFingerprintAsBitVect(mol, nBits)
        except:
            return None

    df['APF'] = df[smiles_col].apply(get_apf)

    return df

# Convert ExplicitBitVect to numpy array
def bitvect_to_array(df, feature_col):
    import numpy as np
    def bitvect_to_numpy(bitvect):
        return np.array(list(map(int, bitvect.ToBitString())))
    # Apply conversion to each element in the series
    numpy_arrays = [bitvect_to_numpy(e) for e in df[feature_col]]
    # If you want all arrays stacked together into a 2D array:
    stack = np.vstack(numpy_arrays)
    return stack

#Tanimoto similarity weight
def readacross_tanimoto_weight(df, x_train, y_train, x_test, n_top, smiles_col,weight_power, nBits=1024):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    import pandas as pd
    import numpy as np
    train = df.loc[x_train.index]
    test  = df.loc[x_test.index]
    y_pred = []
    #smiles to morgan fingerprint
    train_smile= list(train[smiles_col])
    test_smile = list(test[smiles_col])
    for compound_test in test_smile:
        mol1 = Chem.MolFromSmiles(compound_test)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 3, nBits=1024)
        similarity_array = []
        for compound_train in train_smile:
            mol2 = Chem.MolFromSmiles(compound_train)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 3, nBits=1024)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            similarity_array.append(similarity)
        similarity_series = pd.Series(similarity_array, index = train.index)
        adjusted_weights = similarity_series ** weight_power
        indices = similarity_series.nlargest(n_top).index
        y_values = y_train.loc[indices].values.flatten()  # make sure y_values is 1D
        y_pred.append(np.average(y_values, weights=adjusted_weights.loc[indices].values))
    y_pred = pd.DataFrame(y_pred, index=test.index, columns=['y_pred'])
    return y_pred

def readacross_tanimoto_weight_apf(df, x_train, y_train, x_test, n_top, smiles_col,weight_power, nBits=1024):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    import pandas as pd
    import numpy as np
    train = df.loc[x_train.index]
    test  = df.loc[x_test.index]
    y_pred = []
    #smiles to morgan fingerprint
    train_smile= list(train[smiles_col])
    test_smile = list(test[smiles_col])
    for compound_test in test_smile:
        mol1 = Chem.MolFromSmiles(compound_test)
        fp1 = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol1,nBits=1024)
        similarity_array = []
        for compound_train in train_smile:
            mol2 = Chem.MolFromSmiles(compound_train)
            fp2 = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol2,nBits=1024)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
            similarity_array.append(similarity)
        similarity_series = pd.Series(similarity_array, index = train.index)
        adjusted_weights = similarity_series ** weight_power
        indices = similarity_series.nlargest(n_top).index
        y_values = y_train.loc[indices].values.flatten()  # make sure y_values is 1D
        y_pred.append(np.average(y_values, weights=adjusted_weights.loc[indices].values))
    y_pred = pd.DataFrame(y_pred, index=test.index, columns=['y_pred'])
    return y_pred


#check accuracy
def accuracy(y_pred, y_test):
    from sklearn.metrics import mean_absolute_error, r2_score
    import pandas as pd
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return pd.Series([mae, r2], index=['MAE', 'R2'])

#Prediction
def enraqsar(x_test, n_top, weight, name):
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import LeaveOneOut, cross_val_predict
    from sklearn.metrics import mean_absolute_error
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    loo = LeaveOneOut()
    scaler = StandardScaler()
    x_train = pd.read_csv()
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