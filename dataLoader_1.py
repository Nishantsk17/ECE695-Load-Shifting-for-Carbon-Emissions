import pandas as pd
import numpy as np

# columns_to_keep = ['datetime', 'lower bound', 'upper bound']  # Replace with your column names

# # Read the original CSV file, selecting only the desired columns
# df = pd.read_csv('data/csvs_spci_hourly/CISO_direct_hourly_CI_forecasts_spci__alpha_0.1.csv', usecols=columns_to_keep)

# # Write the selected columns to a new CSV file
# df.to_csv('loadshift_data/CISO_carbon_interval_alpha_0.1.csv', index=False)

def carbon_data(stage, confidence):
    files = [('data/csvs_carboncast/CISO_direct_24hr_CI_forecasts.csv', f'data/csvs_spci_hourly/CISO_direct_hourly_CI_forecasts_spci__alpha_{confidence}.csv'),
             ('data/csvs_carboncast/ERCO_direct_24hr_CI_forecasts.csv', f'data/csvs_spci_hourly/ERCO_direct_hourly_CI_forecasts_spci__alpha_{confidence}.csv'),
             ('data/csvs_carboncast/ISNE_direct_24hr_CI_forecasts.csv', f'data/csvs_spci_hourly/ISNE_direct_hourly_CI_forecasts_spci__alpha_{confidence}.csv'),]

    carbon_trace = np.empty((0, 24))
    point_pred = np.empty((0, 24))
    interval_trace = np.empty((0, 24, 2))
    for f in files:
        out1, out2, out3 = get_train_data(f)
        carbon_trace = np.concatenate((carbon_trace, out1), axis=0)
        interval_trace = np.concatenate((interval_trace, out2), axis=0)
        point_pred = np.concatenate((point_pred, out3), axis=0)
    
    # print(carbon_trace.shape)
    # print(interval_trace.shape)
    # print(point_pred.shape)
    ind = int(len(carbon_trace)*0.8)
    if stage == "train":
        return carbon_trace[:ind], interval_trace[:ind], point_pred[:ind]
    elif stage == "test":
        return carbon_trace[ind:], interval_trace[ind:], point_pred[ind:]

def power_data(stage):
    files = ['data/google-cella-power/cella_pdu6.csv',
             'data/google-cella-power/cella_pdu7.csv',
             'data/google-cella-power/cella_pdu8.csv',
             'data/google-cella-power/cella_pdu9.csv',
             'data/google-cella-power/cella_pdu10.csv']
    
    power_trace = np.empty((0, 24))
    for f in files:
        power_trace = np.concatenate((power_trace, get_power_data(f)), axis=0)
    
    current_length = power_trace.shape[0]
    target_length = 537

    # Calculate how many times we need to repeat to reach the target length
    repeat_count = (target_length + current_length - 1) // current_length  # Ceiling division

    # Repeat the array along the first axis
    extended_power_trace = np.tile(power_trace, (repeat_count, 1))

    # Trim the array to exactly (537, 24)
    extended_power_trace = extended_power_trace[:target_length, :]
    power_trace = extended_power_trace
    ind = int(len(power_trace)*0.8)
    if stage == "train":
        return power_trace[:ind]
    elif stage == "test":
        return power_trace[ind:]


def get_train_data(files):
    actual_carbon_df = pd.read_csv(files[0])

    # Convert the 'datetime' column to datetime format in both dataframes
    actual_carbon_df['datetime'] = pd.to_datetime(actual_carbon_df['datetime'])

    # Extract the date and hour information
    actual_carbon_df['date'] = actual_carbon_df['datetime'].dt.date
    actual_carbon_df['hour'] = actual_carbon_df['datetime'].dt.hour

    # Reshape the actual carbon intensity data to a days x 24 matrix, removing the datetime column
    actual_carbon_matrix = actual_carbon_df.pivot(index='date', columns='hour', values='carbon_intensity_actual')
    actual_carbon_array = actual_carbon_matrix.to_numpy()
    actual_carbon_array = actual_carbon_array[1:len(actual_carbon_array)-1][:]

    # Reshape the avg carbon intensity forecast data to a days x 24 matrix
    forecast_carbon_matrix = actual_carbon_df.pivot(index='date', columns='hour', values='avg_carbon_intensity_forecast')
    forecast_carbon_array = forecast_carbon_matrix.to_numpy()
    forecast_carbon_array = forecast_carbon_array[1:len(forecast_carbon_array)-1][:]

    carbon_interval_df = pd.read_csv(files[1])

    # Convert the 'datetime' column to datetime format and extract date and hour
    carbon_interval_df['datetime'] = pd.to_datetime(carbon_interval_df['datetime'])
    carbon_interval_df['date'] = carbon_interval_df['datetime'].dt.date
    carbon_interval_df['hour'] = carbon_interval_df['datetime'].dt.hour

    # Reshape the data to create separate day-by-hour matrices for lower and upper bounds
    lower_bound_matrix = carbon_interval_df.pivot(index='date', columns='hour', values='lower bound')
    upper_bound_matrix = carbon_interval_df.pivot(index='date', columns='hour', values='upper bound')

    # Convert both DataFrames to 2D arrays and stack them along a new third axis (depth) to create a 3D array (N days x 24 x 2)
    combined_interval_array = np.stack([lower_bound_matrix.to_numpy(), upper_bound_matrix.to_numpy()], axis=-1)
    #print(len(combined_interval_array))

    #ind = int(len(actual_carbon_array)*0.8)
    return actual_carbon_array, combined_interval_array, forecast_carbon_array

def get_power_data(file):
    power_data_df = pd.read_csv(file)
    power_data = power_data_df['measured_power_util'].tolist()
    ind = len(power_data) - len(power_data)%12
    power_data = power_data[:ind]
    hourly_power_data = np.mean(np.reshape(power_data[:len(power_data) // 12 * 12], (-1, 12)), axis=1)
    daily_power_data = np.reshape(hourly_power_data, (-1,24))
    #split = int(len(daily_power_data)*0.8)
    return daily_power_data


if __name__ == "__main__":
    carbon_data("train")
    power_data("train")
