
#check accuracy
def accuracy(y_pred, y_test):
    from sklearn.metrics import mean_absolute_error, r2_score
    import pandas as pd
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return pd.Series([mae, r2], index=['MAE', 'R2'])