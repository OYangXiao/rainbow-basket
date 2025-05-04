import numpy
import importlib


def calc_daily_log_returns(history_data):
    daily_log_returns = {}
    # calc daily log returns for each index in history_data
    # history_data is a dict, key is index name, value is a list of index values
    for index_name in history_data:
        # convert list to numpy array
        daily_log_returns[index_name] = numpy.array(history_data[index_name])
        # calc daily log returns
        daily_log_returns[index_name] = numpy.log(
            daily_log_returns[index_name] / numpy.roll(daily_log_returns[index_name], 1)
        )
        # set first value to 0
        daily_log_returns[index_name][0] = 0
        # convert to list
        daily_log_returns[index_name] = daily_log_returns[index_name].tolist()

    write_to_json = importlib.import_module("write-json").write_to_json
    # write daily log returns to file
    write_to_json(daily_log_returns, "output-2-daily-log-returns.json")

    return daily_log_returns
