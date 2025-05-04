import importlib

history_data = importlib.import_module( "function-1-resolve-history-data").resolve_history_data()

daily_log_returns = importlib.import_module( "function-2-calc-daily-log-returns").calc_daily_log_returns(history_data)

importlib.import_module( "function-3-draw-daily-log-return-plot").draw_daily_log_returns_plot(daily_log_returns)

for index_name in daily_log_returns:
    buketized_data = importlib.import_module("function-4-bucketize-returns").bucketize_returns(index_name, daily_log_returns[index_name], 16)
    importlib.import_module("function-5-draw-histogram-returns").draw_histogram_returns(index_name, buketized_data)