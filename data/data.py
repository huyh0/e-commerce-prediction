import pandas as pd
from sklearn.model_selection import train_test_split


def load_csv(csv_path):
    data = pd.read_csv(f"{csv_path}")

    print(f"Shape:\n{data.shape}\n")
    print("Base Rate:", data["Revenue"].value_counts(normalize=True), "\n")
    print(f"NA Data:\n{data.isna().sum()}\n")
    print("Duplicated Data:", data.duplicated().sum())

    return data

def load_training_set(df: pd.DataFrame):

    X = df.drop(columns=["PageValues", "Revenue"])
    y = df["Revenue"]

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6, test_size=0.4, random_state=42, stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size = 0.5, test_size=0.5, random_state=42, stratify=y_temp)

    for name, labels in [("train", y_train), ("val", y_val), ("test", y_test)]:
        print(f"{name}: {len(labels):,} rows, {labels.mean():.2%} purchased")

    categorical_columns = [
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    ]

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":

    dataframe = load_csv(r"C:\Users\Sky\Documents\Projects\mlflow\dataset\online_shoppers_intention.csv")

    X_train, X_val, X_test, y_train, y_val, y_test = load_training_set(dataframe)



