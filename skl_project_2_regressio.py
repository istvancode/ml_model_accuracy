# Regresszió: célnak használd a teljesitmeny oszlopot, és dobd ki a kategoria oszlopot. 
# Ugyanazokat a feature-öket (óra, napszak, évszak One-Hot) etesd be a regresszorba, 
# és mérj MAE/RMSE-t.
# Jó baseline: LinearRegression (LinREG), aztán próbáld RandomForestRegressor-ral is.
#______________________________
# A KLASSZIFIKÁCIÓBAN pontosságot mérünk (LogisticRegression, DecisionTreeClassifier)
# A REGRESSZIÓBAN hibaarányt (LinearRegression, RandomForestRegressor)


# 📊 Regresszió vs. Klasszifikáció
# Tulajdonság	Regresszió 🧮	Klasszifikáció 🏷️
# Kimenet típusa	Folytonos (numerikus érték, pl. ár, teljesítmény)	Diszkrét (kategória, pl. gyenge/közepes/erős, 0/1)
# Tipikus feladat	„Mekkora lesz?” (értékbecslés)	„Melyik csoportba tartozik?” (osztályozás)
# Modellek példák	Linear Regression, Random Forest Regressor, SVR	Logistic Regression, Decision Tree Classifier, Random Forest Classifier, SVM
# Metrikák	MAE (átlagos abszolút hiba), RMSE (négyzetes hiba gyök), R²	Accuracy, Precision, Recall, F1-score, Confusion Matrix
# Kimenet példák	234 kW, 19.8 °C, 12.500 Ft	„Erős”, „Közepes”, „Gyenge” vagy „igen/nem”
# Vizualizáció	Predikált érték vs. valós érték (scatter plot, line plot)	Confusion Matrix, ROC görbe, oszlopdiagram az osztályokhoz

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

mm_scaler = MinMaxScaler()
st_scaler = StandardScaler()

# adat beolvasás csv-ből
solar_data = pd.read_csv('napelem_teljesitmeny_bovitett.csv')

# kategóriák dekódolása
df_encoded = pd.get_dummies(solar_data, columns=["napszak", "evszak","idopont"])

# train / test szétválasztása
X = df_encoded.drop(columns=["teljesitmeny", "kategoria"])
y = df_encoded["teljesitmeny"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
print(X_train.dtypes)
#______________________________________MLflow loggolás
with mlflow.start_run():
    # skálázás MinMaxScale-vel
    X_train_scaled = mm_scaler.fit_transform(X_train)
    X_test_scaled = mm_scaler.transform(X_test)

    # skálázás stansardSale-vel
    X_train_st_scaled = st_scaler.fit_transform(X_train)
    X_test_st_scaled = st_scaler.transform(X_test)

    # MODELLEK készítése  !!! LINEÁRIS REGRESSZIÓNÁL NEM KELL MAX ITER? MERT ANALITIKUS MEGOLDÁSSAL DOLGOZIK !!!
    mm_linreg = LinearRegression()  
    mm_linreg.fit(X_train_scaled, y_train)

    st_linreg = LinearRegression()  
    st_linreg.fit(X_train_st_scaled, y_train)

    model = RandomForestRegressor(max_depth=5)
    model.fit(X_train,y_train)

    # mentés fájlba
    with open("solar_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # előrejelzések
    predictions = model.predict(X_test)
    mm_preds = mm_linreg.predict(X_test_scaled)
    st_preds = st_linreg.predict(X_test_st_scaled)

    # mérések
    mm_mae = mean_absolute_error(y_test, mm_preds)
    mm_rmse = np.sqrt(mean_squared_error(y_test, mm_preds))
    st_mae = mean_absolute_error(y_test, st_preds)
    st_rmse = np.sqrt(mean_squared_error(y_test, st_preds))
    rf_mae = mean_absolute_error(y_test, predictions)
    rf_rmse = np.sqrt(mean_squared_error(y_test, predictions))


    # Log paraméterek
    mlflow.log_param("max_depth", 5)

    # Log metrikák
    mlflow.log_metric("Mae_minmax", mm_mae)
    mlflow.log_metric("Rmse_minmax", mm_rmse)
    mlflow.log_metric("Mae_stand", st_mae)
    mlflow.log_metric("Rmse_stand", st_rmse)
    mlflow.log_metric("Mae_rf", rf_mae)
    mlflow.log_metric("Rmse_rf", rf_rmse)

    # Modell mentése
    mlflow.sklearn.log_model(model, "model")
    mlflow.sklearn.log_model(mm_linreg, "linreg_minmax")
    mlflow.sklearn.log_model(st_linreg, "linreg_standard")


scores = {
    "Mae_minmax": mm_mae,
    "Rmse_minmax": mm_rmse,
    "Mae_stand": st_mae,
    "Rmse_stand": st_rmse,
    "Mae_rf": rf_mae,
    "Rmse_rf": rf_rmse
}


# _____________________________GRAFIKONOK
colors = ["skyblue", "orange", "limegreen"]

fig, axs = plt.subplots(1, 2, figsize=(12, 6))
fig.suptitle("Modellek összehasonlítása (MAE és RMSE)", fontsize=16, color="navy")

# MAE diagram
mae_scores = {k:v for k,v in scores.items() if "Mae" in k}
axs[0].bar(mae_scores.keys(), mae_scores.values(), color=colors)
axs[0].set_title("MAE összehasonlítás")
axs[0].set_ylabel("Hiba")

# RMSE diagram
rmse_scores = {k:v for k,v in scores.items() if "Rmse" in k}
axs[1].bar(rmse_scores.keys(), rmse_scores.values(), color=colors)
axs[1].set_title("RMSE összehasonlítás")
axs[1].set_ylabel("Hiba")

plt.tight_layout()
plt.show()
