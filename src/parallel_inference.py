import os
import pandas as pd
import argparse
import mlflow
from sklearn.ensemble import GradientBoostingRegressor


def get_best_metric(runs, metric_name):
    best_metric = -1
    best_run = None
    for r in runs:
        if best_metric == -1:
            best_metric = r.data.metrics[metric_name]
            best_run = r.info.run_id
        else:
            if r.data.metrics[metric_name] < best_metric:
                best_metric = r.data.metrics[metric_name]
                best_run = r.info.run_id
    
    return best_run

def init():

    # Parse Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--drop_cols", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--experiment_names", type=str)
    parser.add_argument("--metric_name", type=str)
    parser.add_argument("--date_col", type=str)
    parser.add_argument("--tracking_uri", type=str)

    args, _ = parser.parse_known_args()

    global metric_name
    metric_name = args.metric_name

    global drop_cols
    drop_cols = args.drop_cols.split(",")

    global experiment_names
    experiment_names = args.experiment_names.split(",")

    global output_dir
    output_dir = args.output_dir

    global tracking_uri
    tracking_uri = args.tracking_uri

    global date_col
    date_col = args.date_col

    global mlflow_client
    mlflow_client = mlflow.MlflowClient()

    # Connect to mlflow tracking server
    mlflow.set_tracking_uri(tracking_uri)


def run(input_data, mini_batch_context):

    store = mini_batch_context.partition_key_value['Store']
    brand = mini_batch_context.partition_key_value['Brand']
    model_name = f"{store}_{brand}"

    # Prepare Data
    input_data = input_data.drop(columns=drop_cols, errors="ignore")

    experiment_ids = list(map(lambda x: mlflow_client.get_experiment_by_name(x).experiment_id, experiment_names))

    # Get best model from registry for given partition
    runs = mlflow_client.search_runs(experiment_ids = experiment_ids,
                                     filter_string=f"tags.brand = '{brand}' AND tags.store = '{store}' AND attributes.status != 'FAILED'"
                                     )


    if len(runs) == 0:
        print(f"No runs found for {model_name}. Using latest version.")
        best_model_version = 'latest'
    else:
        print(f"Found {len(runs)} runs for {model_name}.")
        
        best_run = get_best_metric(runs, metric_name) # This can be removed if using mlflow search runs order_by arg.
        print(f"Best run for {model_name}: {best_run}")

        best_model = mlflow_client.search_model_versions(filter_string=f"run_id = '{best_run}'")
        print(f"Best model for {model_name}: {best_model}")

        best_model_version = best_model[0].version
        

    # Load model (sklearn / mlflow)
    print(f"Loading model {model_name}:{best_model_version}...")
    reg = mlflow.sklearn.load_model(model_uri=f"models:/{model_name}/{best_model_version}")

    # Prep Data
    input_data[date_col] = pd.to_datetime(input_data[date_col])
    input_data = input_data.set_index(date_col).sort_index(ascending=True)
    input_data = input_data.assign(Week_Year=input_data.index.isocalendar().week.values)

    X_test = input_data.drop(columns=drop_cols, errors="ignore")

    # Make prediction
    predictions = reg.predict(X_test)

    # Combine prediction with input_data
    output_data = input_data.copy()
    output_data['predictions'] = predictions

    # Save predictions to output dir
    relative_path = os.path.join(output_dir, "predictions/")

    print(f"Relative path: {relative_path}...")
    if not os.path.exists(relative_path):
            os.makedirs(relative_path)

    output_path = f"{relative_path}{model_name}.csv"
    print(f"Saving predictions to {output_path}...")
    
    output_data.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}.")

    return []