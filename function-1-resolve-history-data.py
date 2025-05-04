# read data from csv file
import pandas as pd
import os
import sys

def resolve_history_data():

    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the path of the data file
    data_file_path = os.path.join(current_dir, "history-index-data.csv")
    # Check if the data directory exists
    if not os.path.exists(data_file_path):
        print(f"Data file not found at {data_file_path}.")
        sys.exit(1)
    # Read the data from the CSV file
    data = pd.read_csv(data_file_path)

    # exclude first three rows
    data = data.iloc[3:]
    # split each line by comma,
    # first column is index name,
    # third column is index value, format is float
    data = data.iloc[:, [0, 1, 2]]
    # rename columns
    data.columns = ["index_name", "index_date", "index_value"]
    # convert index_value to float
    data["index_value"] = data["index_value"].astype(float)
    # map index_name
    # 000016 -> SSE Composite Index
    # 000300 -> CSI 300 Index
    # 000905 -> CSI 500 Index
    data["index_name"] = data["index_name"].map(
        {
            "000016": "SSE Composite Index",
            "000300": "CSI 300 Index",
            "000905": "CSI 500 Index",
        }
    )

    # convert to {"index_name": [index_value1, index_value2, ...]}
    data = data.groupby("index_name")["index_value"].apply(list).to_dict()

    import importlib
    write_to_json = importlib.import_module("write-json").write_to_json
    write_to_json(data, "output-1-history-data.json")

    return data
