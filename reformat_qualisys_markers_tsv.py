
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from typing import Tuple

from qualisys.joint_center_calculation.generic_mappings.full_body_generic_marker_mapping import qualisys_marker_mappings
from qualisys.joint_center_calculation.joint_center_weights.full_body_joint_center_weights import joint_center_weights
from calculate_and_process_qualisys_joint_centers import main

from skellyforge.freemocap_utils.postprocessing_widgets.task_worker_thread import TaskWorkerThread
from skellyforge.freemocap_utils.config import default_settings
from skellyforge.freemocap_utils.constants import (
    TASK_INTERPOLATION,
    TASK_FILTERING,
    TASK_FINDING_GOOD_FRAME,
    TASK_SKELETON_ROTATION,
    PARAM_GOOD_FRAME,
    PARAM_AUTO_FIND_GOOD_FRAME
)

from skellymodels.model_info.qualisys_model_info import QualisysModelInfo
from skellymodels.model_info.mediapipe_model_info import MediapipeModelInfo

def create_freemocap_unix_timestamps(csv_path):
    df = pd.read_csv(csv_path)
    df.replace(-1, float('nan'), inplace=True)
    mean_timestamps = df.iloc[:, 2:].mean(axis=1, skipna=True)
    time_diff = np.diff(mean_timestamps)
    framerate = 1 / np.nanmean(time_diff)
    return mean_timestamps, framerate

def strip_qualisys_tsv(tsv_path, header_line_count):
    original_df = pd.read_csv(tsv_path, skiprows=header_line_count, delimiter="\t")
    with open(tsv_path, 'r') as f:
        header = [next(f).strip().split('\t') for _ in range(header_line_count)]
    header_dict = {item[0].lower(): item[1:] for item in header}
    return original_df, header_dict

def synchronize_qualisys_data(qualisys_df, freemocap_timestamps):
    synchronized_rows = {}
    for frame_number, timestamp in enumerate(freemocap_timestamps):
        if frame_number + 1 < len(freemocap_timestamps):
            next_timestamp = freemocap_timestamps[frame_number + 1]
            rows_in_range = qualisys_df.loc[(qualisys_df['unix_timestamps'] >= timestamp) & (qualisys_df['unix_timestamps'] < next_timestamp)]
            mean_row = rows_in_range.mean(axis=0, skipna=True)
        else:
            rows_in_range = qualisys_df.loc[(qualisys_df['unix_timestamps'] >= timestamp)]
            mean_row = rows_in_range.iloc[0]
        synchronized_rows[frame_number] = mean_row
    return pd.DataFrame.from_dict(synchronized_rows, orient='index', columns=qualisys_df.columns)

def insert_qualisys_timestamp_column(df, start_timestamp, lag_in_seconds=0):
    """
    Insert a new column with Unix timestamps to the Qualisys dataframe.
    
    Parameters:
        df (pd.DataFrame): The reorganized Qualisys dataframe with a 'time' column in seconds.
        start_timestamp (str): The Qualisys start time as a string in the format '%Y-%m-%d, %H:%M:%S.%f'.
        lag_in_seconds (float, optional): The lag between Qualisys and FreeMoCap data in seconds. Default is 0.
        
    Returns:
        pd.DataFrame: The modified Qualisys dataframe with a new 'unix_timestamps' column.
    """
    # Parse the start timestamp and convert to Unix time
    start_time = datetime.strptime(start_timestamp, '%Y-%m-%d, %H:%M:%S.%f')
    start_unix = start_time.timestamp()
    
    # Adjust the 'time' column based on the calculated lag in seconds
    adjusted_time = df['time'] + lag_in_seconds
    
    # Calculate Unix timestamps
    unix_timestamps = adjusted_time + start_unix
    
    # Insert the new column with Unix timestamps
    df.insert(df.columns.get_loc('time') + 1, 'unix_timestamps', unix_timestamps)
    
    return df

def reformat_qualisys_dataframe(qualisys_dataframe: pd.DataFrame):
    # Extract necessary columns and drop unnecessary ones
    time_col = qualisys_dataframe['Time']
    qualisys_dataframe.drop(columns=['Frame', 'Time'] + [col for col in qualisys_dataframe.columns if 'Unnamed' in col], inplace=True)

    # Get the marker names and create a MultiIndex
    markers = [col.split(' ')[0] for col in qualisys_dataframe.columns[::3]]
    multi_index = pd.MultiIndex.from_product([qualisys_dataframe.index, markers], names=['frame', 'marker'])

    # Create an empty dataframe with the MultiIndex and columns for x, y, z, and time
    reorganized_qualisys_dataframe = pd.DataFrame(index=multi_index, columns=['x', 'y', 'z', 'time'])

    # Populate the dataframe with x, y, z, and time values
    for marker in markers:
        reorganized_qualisys_dataframe.loc[pd.IndexSlice[:, marker], 'x'] = qualisys_dataframe[f"{marker} X"].values
        reorganized_qualisys_dataframe.loc[pd.IndexSlice[:, marker], 'y'] = qualisys_dataframe[f"{marker} Y"].values
        reorganized_qualisys_dataframe.loc[pd.IndexSlice[:, marker], 'z'] = qualisys_dataframe[f"{marker} Z"].values

    # Add the time column
    reorganized_qualisys_dataframe['time'] = time_col.repeat(len(markers)).values

    # Reset the index to convert MultiIndex to columns
    reorganized_qualisys_dataframe.reset_index(inplace=True)

    return reorganized_qualisys_dataframe

def reformat_freemocap_dataframe(freemocap_dataframe: pd.DataFrame):

    reorganized_freemocap_data= [
        [frame, col.split('_x')[0], row[col], row[f"{col.split('_x')[0]}_y"], row[f"{col.split('_x')[0]}_z"]]
        for frame, row in freemocap_dataframe.iterrows() for col in freemocap_dataframe.columns[::3]
    ]

    reorganized_freemocap_dataframe = pd.DataFrame(reorganized_freemocap_data, columns=['frame', 'marker', 'x', 'y', 'z'])
    return reorganized_freemocap_dataframe

def normalize(signal: pd.Series) -> pd.Series:
    """
    Normalize a signal to have zero mean and unit variance.
    
    Parameters:
        signal (pd.Series): The signal to normalize.

    Returns:
        pd.Series: The normalized signal.
    """
    return (signal - signal.mean()) / signal.std()


def calculate_optimal_lag(freemocap_data: pd.Series, qualisys_data: pd.Series) -> int:
    """
    Calculate the optimal lag between FreeMoCap and Qualisys data using cross-correlation.

    Parameters:
        freemocap_data (pd.Series): The FreeMoCap data series to compare.
        qualisys_data (pd.Series): The Qualisys data series to compare.

    Returns:
        int: The optimal lag between the two data series.
    """
    # Ensure the two signals are of the same length (trimming the longer one if necessary)
    min_length = min(len(freemocap_data), len(qualisys_data))
    freemocap_data = freemocap_data[:min_length]
    qualisys_data = qualisys_data[:min_length]


    normalized_freemocap = normalize(freemocap_data)
    normalized_qualisys = normalize(qualisys_data)

    # Compute the cross-correlation
    cross_corr = np.correlate(normalized_freemocap, normalized_qualisys, mode='full')

    # Find the lag that maximizes the cross-correlation
    optimal_lag = np.argmax(cross_corr) - (len(normalized_freemocap) - 1)
    print(f"The optimal lag is: {optimal_lag}")

    return optimal_lag

def plot_shifted_signals(freemocap_data: pd.Series, qualisys_data: pd.Series, optimal_lag: int):
    """
    Plot the original and shifted signals to visualize the synchronization.
    
    Parameters:
        freemocap_data (pd.Series): The FreeMoCap data series.
        qualisys_data (pd.Series): The Qualisys data series.
        optimal_lag (int): The optimal lag for synchronization.
    """
    # Normalize the signals
    normalized_freemocap = normalize(freemocap_data)
    normalized_qualisys = normalize(qualisys_data)

    # Shift the extended Qualisys data by the optimal lag
    if optimal_lag > 0:
        shifted_qualisys = np.concatenate([np.zeros(optimal_lag), normalized_qualisys[:-optimal_lag]])
    else:
        shifted_qualisys = np.concatenate([normalized_qualisys[-optimal_lag:], np.zeros(-optimal_lag)])

    # Plot the original and shifted signals
    plt.figure(figsize=(14, 6))
    plt.subplot(1, 2, 1)
    plt.title('Before Shift')
    plt.plot(normalized_freemocap, label='FreeMoCap Data')
    plt.plot(normalized_qualisys, label='Qualisys Data')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title('After Shift')
    plt.plot(normalized_freemocap, label='FreeMoCap Data')
    plt.plot(shifted_qualisys, label=f'Qualisys Data (Shifted by {optimal_lag} frames)')
    plt.legend()

    plt.show()

def dataframe_to_numpy(df):
    # Get the list of unique markers in the order they appear for frame 0
    marker_order = df['marker'].unique().tolist()
    
    # Create a dictionary to map marker names to their order
    marker_order_dict = {marker: idx for idx, marker in enumerate(marker_order)}
    
    # Sort DataFrame by 'frame' and then by the custom marker order
    df['marker_rank'] = df['marker'].map(marker_order_dict)
    df_sorted = df.sort_values(by=['frame', 'marker_rank']).drop(columns=['marker_rank'])
    
    # Extract the x, y, z columns as a NumPy array
    coords_array = df_sorted[['x', 'y', 'z']].to_numpy()
    
    # Get the number of unique frames and markers
    num_frames = df['frame'].nunique()
    num_markers = len(marker_order)
    
    # Reshape the array into the desired shape (frames, markers, dimensions)
    reshaped_array = coords_array.reshape((num_frames, num_markers, 3))
    
    return reshaped_array

def plot_3d_scatter(freemocap_data, qualisys_data):
    def plot_frame(f):
        ax.clear()
        ax.scatter(qualisys_data[f, :, 0], qualisys_data[f, :, 1], qualisys_data[f, :, 2], c='blue', label='Qualisys')
        ax.scatter(freemocap_data[f, :, 0], freemocap_data[f, :, 1], freemocap_data[f, :, 2], c='red', label='FreeMoCap')
        ax.set_xlim([-limit_x, limit_x])
        ax.set_ylim([-limit_y, limit_y])
        ax.set_zlim([-limit_z, limit_z])
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        ax.set_title(f"Frame {f}")
        fig.canvas.draw_idle()

    mean_x = (np.nanmean(qualisys_data[:, :, 0]) + np.nanmean(freemocap_data[:, :, 0])) / 2
    mean_y = (np.nanmean(qualisys_data[:, :, 1]) + np.nanmean(freemocap_data[:, :, 1])) / 2
    mean_z = (np.nanmean(qualisys_data[:, :, 2]) + np.nanmean(freemocap_data[:, :, 2])) / 2

    ax_range = 1000
    limit_x = mean_x + ax_range
    limit_y = mean_y + ax_range
    limit_z = mean_z + ax_range

    fig = plt.figure(figsize=[10, 8])
    ax = fig.add_subplot(111, projection='3d')
    slider_ax = plt.axes([0.25, 0.02, 0.65, 0.03], facecolor='lightgoldenrodyellow')
    frame_slider = Slider(slider_ax, 'Frame', 0, len(qualisys_data) - 1, valinit=0, valstep=1)

    def update(val):
        frame = int(frame_slider.val)
        plot_frame(frame)

    frame_slider.on_changed(update)
    plot_frame(0)
    plt.show()

def convert_lag_from_frames_to_seconds(lag_frames: int, framerate: float) -> float:
    """
    Convert a lag from frames to seconds.

    Parameters:
        lag_frames (int): The lag in frames.
        framerate (float): The framerate of the data.

    Returns:
        float: The lag in seconds.
    """
    return lag_frames / framerate

recording_folder_path = Path(r"D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_44_19_P01_WalkRun_Trial1")
# freemocap_csv_path = recording_folder_path / 'synchronized_videos' / 'timestamps' / 'unix_synced_timestamps.csv'
freemocap_csv_path = recording_folder_path / 'synchronized_videos' / 'unix_synced_timestamps.csv'
qualisys_tsv_path = recording_folder_path / 'qualisys_data' / 'qualisys_exported_markers.tsv'
freemocap_body_csv = recording_folder_path / 'output_data' / 'mediapipe_body_3d_xyz.csv'
header_line_count = 11
synced_tsv_name = 'synchronized_qualisys_markers.tsv'


qualisys_df, header_dict = strip_qualisys_tsv(qualisys_tsv_path, header_line_count=header_line_count)
qualisys_start_timestamp = header_dict["time_stamp"]
new_qual_dataframe = reformat_qualisys_dataframe(qualisys_df)
qualisys_df_with_unix = insert_qualisys_timestamp_column(qualisys_df.copy(), qualisys_start_timestamp, lag_in_seconds=0)

freemocap_timestamps, framerate = create_freemocap_unix_timestamps(freemocap_csv_path)
# framerate= 29.97653535971666
print(f"Calculated FreeMoCap framerate: {framerate}")

qualisys_df, header_dict = strip_qualisys_tsv(qualisys_tsv_path, header_line_count=header_line_count)
qualisys_start_timestamp = header_dict["time_stamp"]

new_qual_dataframe = reformat_qualisys_dataframe(qualisys_df)

# qualisys_df_with_unix = insert_qualisys_timestamp_column(qualisys_df.copy(), qualisys_start_timestamp, lag_in_seconds=0)

# synchronized_qualisys_df = synchronize_qualisys_data(qualisys_df_with_unix, freemocap_timestamps)

# new_qual_dataframe = reformat_qualisys_dataframe(synchronized_qualisys_df)
# print(synchronized_qualisys_df.head())

# freemocap_body_df = pd.read_csv(freemocap_body_csv)
# new_freemocap_dataframe = reformat_freemocap_dataframe(freemocap_body_df)
# freemocap_numpy = dataframe_to_numpy(new_freemocap_dataframe)


joint_centers_frame_marker_dimension, qual_markers = main(new_qual_dataframe, joint_center_weights, qualisys_marker_mappings)
qualisys_joint_centers_dataframe = pd.DataFrame(joint_centers_frame_marker_dimension.reshape(-1, 3), columns=['x', 'y', 'z'])


adjusted_settings = default_settings.copy()
# adjusted_settings[TASK_SKELETON_ROTATION][PARAM_AUTO_FIND_GOOD_FRAME] = False
# adjusted_settings[TASK_SKELETON_ROTATION][PARAM_GOOD_FRAME] = 450 

post_process_task_worker = TaskWorkerThread(raw_skeleton_data=joint_centers_frame_marker_dimension, task_list= [TASK_INTERPOLATION, TASK_FILTERING, TASK_FINDING_GOOD_FRAME, TASK_SKELETON_ROTATION], settings=default_settings, landmark_names=QualisysModelInfo.landmark_names)
post_process_task_worker.run()
processed_qualisys_data = post_process_task_worker.tasks[TASK_SKELETON_ROTATION]['result']

qualisys_joint_centers_dataframe = pd.DataFrame(processed_qualisys_data.reshape(-1, 3), columns=['x', 'y', 'z'])
qualisys_joint_centers_dataframe['frame'] = np.repeat(np.arange(processed_qualisys_data.shape[0]), len(joint_center_weights))
qualisys_joint_centers_dataframe['marker'] = np.tile(list(joint_center_weights.keys()), processed_qualisys_data.shape[0])
qualisys_joint_centers_dataframe = qualisys_joint_centers_dataframe[['frame', 'marker', 'x', 'y', 'z']]



freemocap_post_process_task_worker = TaskWorkerThread(raw_skeleton_data=freemocap_numpy, task_list= [TASK_INTERPOLATION, TASK_FILTERING, TASK_FINDING_GOOD_FRAME, TASK_SKELETON_ROTATION], settings=default_settings, landmark_names=MediapipeModelInfo.landmark_names)
freemocap_post_process_task_worker.run()
processed_freemocap_data = freemocap_post_process_task_worker.tasks[TASK_SKELETON_ROTATION]['result']

freemocap_joint_centers_dataframe = pd.DataFrame(processed_freemocap_data.reshape(-1, 3), columns=['x', 'y', 'z'])
freemocap_joint_centers_dataframe['frame'] = np.repeat(np.arange(processed_freemocap_data.shape[0]), len(MediapipeModelInfo.landmark_names))
freemocap_joint_centers_dataframe['marker'] = np.tile(list(MediapipeModelInfo.landmark_names), processed_freemocap_data.shape[0])
freemocap_joint_centers_dataframe = freemocap_joint_centers_dataframe[['frame', 'marker', 'x', 'y', 'z']]


# plot_3d_scatter(processed_freemocap_data, processed_qualisys_data)

joint_to_compare = 'left_shoulder'

freemocap_data = freemocap_joint_centers_dataframe[freemocap_joint_centers_dataframe['marker'] == joint_to_compare]['y'].values
qualisys_data = qualisys_joint_centers_dataframe[qualisys_joint_centers_dataframe['marker'] == joint_to_compare]['y'].values

optimal_lag = calculate_optimal_lag(freemocap_data, qualisys_data)

print(f"Optimal lag: {optimal_lag}")

plot_shifted_signals(freemocap_data, qualisys_data, optimal_lag)

lag_seconds = convert_lag_from_frames_to_seconds(optimal_lag, framerate)

print(f"Calculated lag in seconds: {lag_seconds}")

qualisys_joint_centers_dataframe['Time'] = qualisys_df['Time']

# qualisys_df_with_unix_lag_corrected = insert_qualisys_timestamp_column(qualisys_joint_centers_dataframe, qualisys_start_timestamp, lag_in_seconds=lag_seconds)

# synchronized_qualisys_df = synchronize_qualisys_data(qualisys_df_with_unix_lag_corrected, freemocap_timestamps)
# synced_qualisys_data = synchronized_qualisys_df[qualisys_joint_centers_dataframe['marker'] == joint_to_compare]['y'].values
# optimal_lag = calculate_optimal_lag(freemocap_data, synced_qualisys_data)

# assert synchronized_qualisys_df.shape[1] == qualisys_df.shape[
#     1], "qualisys_synchronized_df does not have the same number of columns as qualisys_original_df"
# assert synchronized_qualisys_df.shape[0] == len(
#     freemocap_timestamps), "qualisys_synchronized_df does not have the same number of rows as freemocap_timestamps"

path_to_qualisys_folder = recording_folder_path / 'qualisys_data'
path_to_qualisys_csv = path_to_qualisys_folder / 'synchronized_qualisys_markers.csv'
save_path = path_to_qualisys_folder / 'qualisys_joint_centers_3d_xyz.npy'

qualisys_joint_centers_dataframe.to_csv(path_to_qualisys_csv, index=False)

qualisys_joint_centers_frame_marker_dimension = dataframe_to_numpy(qualisys_joint_centers_dataframe)
np.save(save_path, qualisys_joint_centers_frame_marker_dimension)


