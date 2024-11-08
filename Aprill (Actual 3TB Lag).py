import pandas as pd
import numpy as np

df_cli=pd.read_excel('brplpl.xlsx')
df_cli.set_index('date',inplace=True)
df_cli=df_cli.loc['2024-03-01':'2024-04-30']
print(df_cli)

date_range = pd.date_range(start='2024-03-01',end='2024-04-30 23:45:00',freq='15min')

df_cli.set_index(date_range,inplace=True)
df_cli.drop(['block','temp','ws','rh'],axis=1,inplace=True)
print(df_cli)

time_range = pd.date_range(start='2024-04-01 00:00:00',end='2024-04-01 23:45:00',freq='15min').time
date_range = pd.date_range(start='2024-04-01',end='2024-04-30',freq='D')

april_df = pd.DataFrame(index=time_range)
day_start = pd.to_datetime('2024-04-01 00:00:00')
for i in date_range:
    day_end = day_start+pd.Timedelta(minutes=15*95)
    x = df_cli.loc[day_start:day_end]
    x = x['quantum'].values
    april_df[i.date()] = x
    day_start = day_start+pd.Timedelta(days=1)
print(april_df)

from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error

model_run_time = pd.to_datetime('2024-03-31 22:30:00')
april_forecasted_hour_df = pd.DataFrame(index=time_range)
april_forecasted_day_df = pd.DataFrame(index=time_range)
MAPE_df = pd.DataFrame(index=time_range[::6])
MAE_df = pd.DataFrame(index=time_range[::6])
file_path = 'april3TB.xlsx'

for iteration in range(1, 31):
    MAPE_list = []
    MAE_list = []
    day = []
    day_end = model_run_time + pd.Timedelta(days=1)
    while model_run_time != day_end:

        train_end_date = model_run_time - pd.Timedelta(minutes=4 * 15) - pd.Timedelta(minutes=15)
        train_start_date = train_end_date - pd.Timedelta(days=30) + pd.Timedelta(minutes=15)
        test_start_date = model_run_time + pd.Timedelta(minutes=15 * 6)
        test_end_date = test_start_date + pd.Timedelta(minutes=15 * 5)

        train = df_cli.loc[train_start_date:train_end_date]['quantum']
        test = df_cli.loc[test_start_date:test_end_date]['quantum']

        order = (1, 0, 0)
        seasonal_order = (1, 1, 0, 96)

        model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
        results = model.fit()

        forecast_steps = len(df_cli.loc[train_end_date + pd.Timedelta(minutes=15):test_end_date])
        forecast = results.get_forecast(steps=forecast_steps)

        forecast_mean = forecast.predicted_mean
        forecast_ci = forecast.conf_int()

        april_forecasted_hour_df[model_run_time] = np.nan
        april_forecasted_hour_df.loc[test_start_date.time():test_end_date.time(), model_run_time] = forecast_mean[test_start_date:].values
        day = np.append(day, forecast_mean[test_start_date:].values)
        print('forecasted_values for model running at ', model_run_time, 'are:')
        for i in forecast_mean[test_start_date:].values:
            print(i)

        mape = mean_absolute_percentage_error(test, forecast_mean[test_start_date:].values)
        print(f'MAPE: {mape * 100:.2f}% for {test_start_date}')
        MAPE_list.append(mape * 100)

        mae = mean_absolute_error(test, forecast_mean[test_start_date:].values)
        MAE_list.append(mae)
        print(f"Mean Absolute Error: {mae}")

        del model, results, forecast, forecast_mean, forecast_ci

        model_run_time = model_run_time + pd.Timedelta(minutes=15*6)

    april_forecasted_day_df[model_run_time.date()] = day
    MAPE_df[test_end_date.date()] = MAPE_list
    MAE_df[test_end_date.date()] = MAE_list
    print(MAPE_df)

    with pd.ExcelWriter(file_path, engine='xlsxwriter') as writer:
        april_forecasted_hour_df.to_excel(writer, sheet_name='april_forecast_hour', index=True)
        april_forecasted_day_df.to_excel(writer, sheet_name='april_forecast_day', index=True)
        april_df.to_excel(writer, sheet_name='april_test', index=True)
        MAPE_df.to_excel(writer, sheet_name='MAPE', index=True)
        MAE_df.to_excel(writer, sheet_name='MAE', index=True)

