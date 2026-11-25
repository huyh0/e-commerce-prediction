from data.data import load_csv, load_training_set, categorical_columns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, average_precision_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os


import mlflow

mlflow.set_experiment("Purchase Prediction")


dataframe = load_csv(r"C:\Users\Sky\Documents\Projects\mlflow\dataset\online_shoppers_intention.csv")

X_train, X_val, X_test, y_train, y_val, y_test = load_training_set(dataframe)

numerical_columns = X_train.select_dtypes(include="number").columns.difference(categorical_columns).tolist()


with mlflow.start_run():

    ct = ColumnTransformer([
        ("catagorical_preprocess", OneHotEncoder(sparse_output=False, handle_unknown='ignore'), categorical_columns),
        ("numerical_preprocess", StandardScaler(), numerical_columns)],
        remainder='passthrough'
        )

    pipe = Pipeline([
        ("Preprocessing", ct),
        ("LR", LogisticRegression())
    ])

    accuracy = pipe.fit(X_train, y_train).score(X_val, y_val)

    y_pred = pipe.predict(X_val)

    cm = confusion_matrix(y_val, y_pred, labels=pipe.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=pipe.classes_)
    disp.plot()

    os.makedirs("artifacts/", exist_ok=True)
    plt.savefig("artifacts/confusion_matrix.png", dpi=300, bbox_inches="tight")

    y_prob = pipe.predict_proba(X_val)[:, 1]

    precision = average_precision_score(y_val, y_prob)

    mlflow.log_params({
        "Dataset": "Online Shoppers Intention",
        "Train/Val/Split": "60/20/20",
        "Random State": 42,
    })

    mlflow.log_artifact(r"artifacts/confusion_matrix.png")

    mlflow.log_metrics({
        "Accuracy": accuracy,
        "Average Precision": precision
    })

    mlflow.sklearn.log_model(pipe)
