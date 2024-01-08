import os
import pandas as pd
import numpy as np
import argparse
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.ensemble import GradientBoostingRegressor

def init():

    mlflow.start_run()

    parser = argparse.ArgumentParser()

    parser.add_argument("--drop_cols", type=str)
    parser.add_argument("--target_col", type=str)
    parser.add_argument("--date_col", type=str)
    parser.add_argument("--model_folder", type=str)
    parser.add_argument("--partition_keys", type=str)

    args, _ = parser.parse_known_args()

    global target_col
    target_col = args.target_col

    global date_col
    date_col = args.date_col

    global model_folder
    model_folder = args.model_folder

    global drop_cols
    drop_cols = args.drop_cols.split(",")


def run(input_data, mini_batch_context):

    store = mini_batch_context.partition_key_value['Store']
    brand = mini_batch_context.partition_key_value['Brand']
    model_name = f"{store}_{brand}"
    model_description = f"GradientBoostingRegressor for store_brand = f{model_name}"
    print(f"Running train.py for...{model_name}")

    
    with mlflow.start_run(run_name=f"{brand}_{store}_job", nested=True) as train_run:
        
        mlflow.set_tags({"brand": f"{brand}", "store": f"{store}"})
        mlflow.sklearn.autolog()
        print("Mlflow sklearn autologging enabled")

        if not isinstance(input_data, pd.DataFrame):
            raise Exception("Not a valid DataFrame input.")

        if target_col not in input_data.columns:
            raise Exception("No target column found from input tabular data")
        elif date_col not in input_data.columns:
            raise Exception("No date column found from input tabular data")

        print(f"partition_key_value = {mini_batch_context.partition_key_value}")

        # data cleaning
        input_data[date_col] = pd.to_datetime(input_data[date_col])
        input_data = input_data.set_index(date_col).sort_index(ascending=True)
        input_data = input_data.assign(Week_Year=input_data.index.isocalendar().week.values)
        input_data = input_data.drop(columns=drop_cols, errors="ignore")

        # traning & evaluation
        features = input_data.columns.drop(target_col)

        X = input_data[features].values
        y = input_data[target_col].values

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.05, random_state=12, shuffle=False
        )

        reg = GradientBoostingRegressor(random_state=12)
        reg.fit(X_train, y_train)
        reg_pred = reg.predict(X_test)

        print("Calculating metrics...")
        test_rmse = np.sqrt(mean_squared_error(y_test, reg_pred))
        mlflow.log_metric("test_rmse", test_rmse)
        test_mape = np.mean(np.abs((y_test - reg_pred) / y_test) * 100)
        mlflow.log_metric("test_mape", test_mape)


        # Dump model
        print("Dumping model...")
        relative_path = os.path.join(
            model_folder,
            *list(str(i) for i in mini_batch_context.partition_key_value.values()),
        )

        if not os.path.exists(relative_path):
            os.makedirs(relative_path)

        print(f"Saving model to {relative_path}")
        mlflow.sklearn.save_model(reg, relative_path)
    
        print(f"Model saved. Registering {model_name} to AML model registry...")
        # mlflow.set_tag("parent_id", train_run().info.run_id)
        mlflow.sklearn.log_model(sk_model=reg,
                                    registered_model_name=model_name,
                                    artifact_path=relative_path
                                )
        # mlflow.register_model(model_uri=relative_path, name=model_name)
        print(f"Run completed for {model_name}")

    return []