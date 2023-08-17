import pandas as pd
import os

def canonical_smiles(df, smiles_col):
    #import RDKit
    from rdkit.Chem import AllChem as Chem
    #generate canonical smiles
    df['canonical_smiles'] = df[smiles_col].apply(lambda x: Chem.MolToSmiles(Chem.MolFromSmiles(x), isomericSmiles=True))
    df = df.drop(smiles_col, axis=1)
    return df

def remove_missingdata(df):
    #drop missing data
    df = df.dropna()
    return df

def morded_cal(df, smiles_col):
    #import RDKit
    from rdkit.Chem import AllChem as Chem
    from rdkit import Chem
    from mordred import Calculator, descriptors
    #calculate descriptors
    calc = Calculator(descriptors, ignore_3D=True)
    mols = [Chem.MolFromSmiles(smi) for smi in df[smiles_col]]
    des = calc.pandas(mols)
    des = des.set_index(df.index)
    return des

#Tanimoto similarity weight
def readacross_tanimoto_weight(train, y_train, test, n_top, smiles_col,weight_power, nBits=1024):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    import pandas as pd
    import numpy as np
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

def readacross_tanimoto_weight_apf(train, y_train, test, n_top, smiles_col,weight_power, nBits=1024):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    import pandas as pd
    import numpy as np
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

def similarity_tanimono_ecfp(train, test, smiles_col):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem
    import pandas as pd
    import numpy as np
    similarity_max = []
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
        similarity_max.append(max(similarity_array))
    similarity_score = pd.DataFrame({'Similarity': similarity_max}, index=test.index)
    return similarity_score


#Prediction
def enraqsar(test, n_top=3, weight=3):
    from sklearn.preprocessing import StandardScaler
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    from mordred import Calculator, descriptors
    train = pd.read_csv(os.path.join('models', 'training set.csv'), index_col='LigandID')
    x_train = train[['ATSC4c', 'ATSC7i', 'GATS4d', 'SdCH2', 'SdssC', 'BIC4', 'SMR_VSA4',
       'n7Ring', 'n7HRing', 'nG12FARing']]
    y_train = train['pIC50']
    des = morded_cal(test, 'canonical_smiles')
    x_test = remove_missingdata(des)
    #select columns
    x_test = x_test[['ATSC4c', 'ATSC7i', 'GATS4d', 'SdCH2', 'SdssC', 'BIC4', 'SMR_VSA4',
       'n7Ring', 'n7HRing', 'nG12FARing']]
    scaler = StandardScaler().fit(x_train)
    x_test_scale  = scaler.transform(x_test)
    from joblib import load
    lasso_cv_reduce = load(os.path.join('models','qsar.joblib'))
    qsar_test  = lasso_cv_reduce.predict(x_test_scale)
    qsar_test  = pd.DataFrame(qsar_test, columns=['y_pred']).set_index(x_test.index)
    ra_ec_test  = readacross_tanimoto_weight(train, y_train, test, n_top, smiles_col='canonical_smiles',weight_power=weight)
    ra_ap_test  = readacross_tanimoto_weight_apf(train, y_train, test, n_top, smiles_col='canonical_smiles',weight_power=weight)
    #get y_pred ra
    ra_ec_test  = ra_ec_test['y_pred']
    ra_ap_test  = ra_ap_test['y_pred']
    #combine train
    raqsar_test  = pd.concat([ra_ec_test,  ra_ap_test,  qsar_test], axis=1)
    #ensemble
    rf = load(os.path.join('models','qraqsar.joblib'))
    qsar_result = {}
    y_pred_test  = rf.predict(raqsar_test)
    raqsar_test.columns = ['QSAR', 'ECFP', 'APF']
    #similarities
    similarity_ecfp = similarity_tanimono_ecfp(train, test, smiles_col='canonical_smiles')
    #add y values and metrics to dictionary
    qsar_result['raqsar_test']  = raqsar_test
    qsar_result['y_pred_test']  = y_pred_test
    qsar_result['Similarity']   = similarity_ecfp
    #convert pic50 to ic50
    def pIC50_to_IC50(pIC50_values):
        return (10**(-np.array(pIC50_values)))*1e+9
    qsar_result['Predicted IC50'] = pIC50_to_IC50(y_pred_test)
    #create dataframe
    raqsar_df = qsar_result['raqsar_test']
    y_pred_df = pd.DataFrame(qsar_result['y_pred_test'], columns=['Predicted pIC50']).set_index(test.index)
    similarity_df = qsar_result['Similarity']
    ic50_df = pd.DataFrame(qsar_result['Predicted IC50'], columns=['Predicted IC50 (nM)']).set_index(test.index) 
    #combine
    qsar_result_df = pd.concat([raqsar_df, y_pred_df,ic50_df, similarity_df], axis=1)
    return qsar_result_df


