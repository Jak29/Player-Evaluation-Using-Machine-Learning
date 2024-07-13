from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
import numpy as np
import re

def assign_position_value(position):
    position_values = {
        'ST': 15,
        'M': 16,
        'D': 19,
        'GK': 17,
    }
    found_positions = re.findall(r'\b(ST|M|D|GK)\b', position)
    max_value = 0
    for pos in found_positions:
        if pos in position_values:
            max_value = max(max_value, position_values[pos])
    return max_value

df2 = pd.read_csv("2020.csv")
df2 = df2[df2["Division"].isin(["English Premier Division", "Ligue 1 Conforama", "Bundesliga", "Spanish First Division", "Italian Serie A"])]
df2 = df2.drop("Division", axis=1)
df2 = df2[["Age", "TrueHeight", "Acc",	"Wor",	"Vis",	"Thr"	,"Tec",	"Tea",	"Tck",	"Str",	"Sta"	,"TRO"	,"Ref",	"Pun",	"Pos",	"Pen"	,"Pas",	"Pac",	"1v1",	"OtB","Mar", "Lon" ,"Ldr" ,"Kic", "Jum" ,"Hea", "Han" ,"Fre" , "Fir" ,"Fin", "Ecc" ,"Dri" ,"Det" ,"Dec" ,"Cro", "Cor", "Cnt" ,"Cmp" ,"Com" ,"Cmd" , "Bal", "Ant" ,"Agi", "Agg" ,"Aer", "Position", "TrueValue"]]

df2['PositionValue'] = df2['Position'].apply(assign_position_value)
df2 = df2.drop("Position", axis=1)


X = df2.drop("TrueValue", axis=1)
y = df2["TrueValue"]

X = X.reset_index(drop=True)
y = y.reset_index(drop=True)


kf = KFold(n_splits=5, shuffle=True, random_state=42)
rmse_scores = []
mae_scores = []
r_squared_scores = []

for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y[train_index], y[test_index]
    
    model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.15, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r_squared = r2_score(y_test, predictions)
    
    rmse_scores.append(rmse)
    mae_scores.append(mae)
    r_squared_scores.append(r_squared)

mean_rmse = np.mean(rmse_scores)
mean_mae = np.mean(mae_scores)
mean_r_squared = np.mean(r_squared_scores)

print(f"Mean RMSE: {mean_rmse:,.2f}")
print(f"Mean MAE: {mean_mae:,.2f}")
print(f"Mean R-squared: {mean_r_squared:,.4f}")



