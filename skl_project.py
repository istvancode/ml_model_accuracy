import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

mm_scaler = MinMaxScaler()
st_scaler = StandardScaler()

# adat beolvasás csv-ből
solar_data = pd.read_csv('napelem_teljesitmeny_bovitett.csv')

# kategóriák dekódolása
df_encoded = pd.get_dummies(solar_data, columns=["napszak", "evszak"])

# train / test szétválasztása
X = df_encoded.drop(columns=["idopont", "kategoria"])
y = df_encoded["kategoria"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#______________________________________MLflow loggolás
with mlflow.start_run():
    # skálázás MinMaxScale-vel
    X_train_scaled = mm_scaler.fit_transform(X_train)
    X_test_scaled = mm_scaler.transform(X_test)

    # skálázás stansardSale-vel
    X_train_st_scaled = st_scaler.fit_transform(X_train)
    X_test_st_scaled = st_scaler.transform(X_test)

    # MODELLEK készítése
    mm_logreg = LogisticRegression(max_iter=1000)  
    mm_logreg.fit(X_train_scaled, y_train)

    st_logreg = LogisticRegression(max_iter=1000)  
    st_logreg.fit(X_train_st_scaled, y_train)

    model = DecisionTreeClassifier(max_depth=5)
    model.fit(X_train,y_train)

    # mentés fájlba
    with open("solar_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # előrejelzések
    predictions = model.predict(X_test)
    mm_preds = mm_logreg.predict(X_test_scaled)
    st_preds = st_logreg.predict(X_test_st_scaled)

    # pontok
    score = accuracy_score(y_test, predictions)
    mm_score = accuracy_score(y_test, mm_preds)
    st_score = accuracy_score(y_test, st_preds)

    # Log paraméterek
    mlflow.log_param("max_depth", 5)

    # Log metrikák
    mlflow.log_metric("DecTree - accuracy", score)
    mlflow.log_metric("Logreg - mm_accuracy", mm_score)
    mlflow.log_metric("Logreg - st_accuracy", st_score)

    # Modell mentése
    mlflow.sklearn.log_model(model, "model")
    mlflow.sklearn.log_model(mm_logreg, "logreg_minmax")
    mlflow.sklearn.log_model(st_logreg, "logreg_standard")

print("DecTree - score:", score)
print("Logreg - mm_score:", mm_score)
print("Logreg - st_score:", st_score)

scores = {
    "DecisionTree": score,
    "LogReg (MinMax)": mm_score,
    "LogReg (Standard)": st_score
}


# _____________________________GRAFIKONOK
colors = ["skyblue", "orange", "limegreen"]

fig, axs = plt.subplots(2, 1, figsize=(10, 8))
fig.suptitle("Modellek összehasonlítása (Accuracy)", fontsize=20, color="navy")
fig.subplots_adjust(hspace=0.8)

axs[0].bar(scores.keys(), scores.values(), color=colors)
axs[0].set_title("Oszlop diagram")
axs[0].set_ylabel("Pontosság")
axs[0].set_ylim(0, 1)

values = [v*100 for v in scores.values()]
axs[1].pie(values, labels=scores.keys(), autopct="%1.1f%%", startangle=90)
axs[1].set_title("Kördiagram")
axs[1].set_ylim(0, 1)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
