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
