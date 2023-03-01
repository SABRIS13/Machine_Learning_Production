#Importar librerías
import argparse
# import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Carga de dataset
titanic_df = pd.read_csv("https://raw.githubusercontent.com/edroga/Datasets_for_projects/main/titanic.csv")

# Preprocesamos el dataset
# Eliminamos columnas que no nos sirven para el modelo
# Eliminamos datos faltantes
titanic_df = (titanic_df
              .drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
              .dropna()
              )

# Crear variables dummy
titanic_df = pd.get_dummies(titanic_df, columns=["Sex", "Embarked", "Pclass"])

# Normalizamos las variables numéricas
scaler = StandardScaler()
titanic_df[["Age", "Fare"]] = scaler.fit_transform(titanic_df[["Age", "Fare"]])

# Parametros de la linea de comando
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--penalty", type=str, default='l2')
args = parser.parse_args()

# Partir los datos en train y test
X_train, X_test, y_train, y_test = train_test_split(titanic_df.drop(columns=["Survived"]),
                                                    titanic_df["Survived"],
                                                    test_size=0.2,
                                                    random_state=42)

# Instanciar el modelo
model = LogisticRegression(penalty=args.penalty, C=args.alpha)

# Ajustar el modelo
model.fit(X_train, y_train)

# Crear el predict
y_pred = model.predict(X_test)

# Calcular las métricas
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Imprimir métricas
print("Accuracy: {}".format(accuracy))
print("Precision: {}".format(precision))
print("Recall: {}".format(recall))
print("Balanced Accuracy: {}".format(balanced_accuracy))
print("F1: {}".format(f1))
print("ROC AUC: {}".format(roc_auc))