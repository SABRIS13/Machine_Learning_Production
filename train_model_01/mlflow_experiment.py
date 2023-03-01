#Librerías
import argparse
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
titanic_df = pd.read_csv("https://raw.githubusercontent.com/edroga/Datasets_for_projects/main/titanic.csv")

# Preprocesar Dataset
titanic_df = (titanic_df
              .drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
              .dropna()
              )

# Crear variables dummy
titanic_df = pd.get_dummies(titanic_df, columns=["Sex", "Embarked", "Pclass"])

# Normalizar las variables numéricas
scaler = StandardScaler()
titanic_df[["Age", "Fare"]] = scaler.fit_transform(titanic_df[["Age", "Fare"]])

# Parametros de la línea de comando
parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, default=0.1)
parser.add_argument("--penalty", type=str, default='l2')
args = parser.parse_args()

# Partir datos en train y test
X_train, X_test, y_train, y_test = train_test_split(titanic_df.drop(columns=["Survived"]),
                                                    titanic_df["Survived"],
                                                    test_size=0.2,
                                                    random_state=42)

#Abrir sesión de mlflow
with mlflow.start_run(run_name=f'alpha={args.alpha}, penalty={args.penalty}'):

    # Empezar el experimento
    mlflow.set_experiment("titanic_experiment")

    # Guardar parametros
    mlflow.log_param("alpha", args.alpha)
    mlflow.log_param("penalty", args.penalty)

    # Instanciar modelo
    model = LogisticRegression(penalty=args.penalty, C=args.alpha)

    # Ajustar modelo
    model.fit(X_train, y_train)

    # Crear predict
    y_pred = model.predict(X_test)

    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    balanced_accuracy = balanced_accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)

    # Guardar modelo
    mlflow.sklearn.log_model(model, "model")

    # Guardar metricas
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("balanced_accuracy", balanced_accuracy)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)

    # Imprimir métricas
    print("Accuracy: {}".format(accuracy))
    print("Precision: {}".format(precision))
    print("Recall: {}".format(recall))
    print("Balanced Accuracy: {}".format(balanced_accuracy))
    print("F1: {}".format(f1))
    print("ROC AUC: {}".format(roc_auc))