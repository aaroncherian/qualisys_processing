
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

from qualisys.joint_center_calculation.generic_mappings.full_body_generic_marker_mapping import qualisys_marker_mappings
from qualisys.joint_center_calculation.joint_center_weights.full_body_joint_center_weights import joint_center_weights
from calculate_and_process_qualisys_joint_centers import main

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
        df (pd.DataFrame): The original Qualisys dataframe with a 'Time' column in seconds.
        start_timestamp (str): The Qualisys start time as a string in the format '%Y-%m-%d, %H:%M:%S.%f'.
        lag_in_seconds (float, optional): The lag between Qualisys and FreeMoCap data in seconds. Default is 0.
        
    Returns:
        pd.DataFrame: The modified Qualisys dataframe with a new 'unix_timestamps' column.
    """
    start_time = datetime.strptime(start_timestamp[0], '%Y-%m-%d, %H:%M:%S.%f')
    start_unix = start_time.timestamp()
    
    # Adjust the 'Time' column based on the calculated lag in seconds
    adjusted_time = df['Time'] + lag_in_seconds
    
    # Insert the new column with Unix timestamps
    df.insert(df.columns.get_loc('Time') + 1, 'unix_timestamps', adjusted_time + start_unix)
    
    return df

def reformat_qualisys_dataframe(qualisys_dataframe: pd.DataFrame,):
    qualisys_dataframe.drop(columns=['Frame', 'Time', 'unix_timestamps'] + [col for col in qualisys_dataframe.columns if 'Unnamed' in col], inplace=True)


    # Create the reorganized_data list with marker names as strings
    reorganized_qualisys_data= [
        [frame, col.split(' ')[0], row[col], row[f"{col.split(' ')[0]} Y"], row[f"{col.split(' ')[0]} Z"]]
        for frame, row in qualisys_dataframe.iterrows() for col in qualisys_dataframe.columns[::3]
    ]

    reorganized_qualisys_dataframe = pd.DataFrame(reorganized_qualisys_data, columns=['frame', 'marker', 'x', 'y', 'z'])
    return reorganized_qualisys_dataframe

def reformat_freemocap_dataframe(freemocap_dataframe: pd.DataFrame):

    reorganized_freemocap_data= [
        [frame, col.split('_x')[0], row[col], row[f"{col.split('_x')[0]}_y"], row[f"{col.split('_x')[0]}_z"]]
        for frame, row in freemocap_dataframe.iterrows() for col in freemocap_dataframe.columns[::3]
    ]

    reorganized_freemocap_dataframe = pd.DataFrame(reorganized_freemocap_data, columns=['frame', 'marker', 'x', 'y', 'z'])
    return reorganized_freemocap_dataframe

recording_folder_path = Path(r"D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_15_44_19_P01_WalkRun_Trial1")
# freemocap_csv_path = recording_folder_path / 'synchronized_videos' / 'timestamps' / 'unix_synced_timestamps.csv'
freemocap_csv_path = recording_folder_path / 'synchronized_videos' / 'unix_synced_timestamps.csv'
qualisys_tsv_path = recording_folder_path / 'qualisys_data' / 'qualisys_exported_markers.tsv'
freemocap_body_csv = recording_folder_path / 'output_data' / 'mediapipe_body_3d_xyz.csv'
header_line_count = 11
synced_tsv_name = 'synchronized_qualisys_markers.tsv'


qualisys_df, header_dict = strip_qualisys_tsv(qualisys_tsv_path, header_line_count=header_line_count)
qualisys_start_timestamp = header_dict["time_stamp"]
qualisys_df_with_unix = insert_qualisys_timestamp_column(qualisys_df.copy(), qualisys_start_timestamp, lag_in_seconds=0)

freemocap_timestamps, framerate = create_freemocap_unix_timestamps(freemocap_csv_path)
# framerate= 29.97653535971666
print(f"Calculated FreeMoCap framerate: {framerate}")

qualisys_df, header_dict = strip_qualisys_tsv(qualisys_tsv_path, header_line_count=header_line_count)
qualisys_start_timestamp = header_dict["time_stamp"]
qualisys_df_with_unix = insert_qualisys_timestamp_column(qualisys_df.copy(), qualisys_start_timestamp, lag_in_seconds=0)

synchronized_qualisys_df = synchronize_qualisys_data(qualisys_df_with_unix, freemocap_timestamps)

new_qual_dataframe = reformat_qualisys_dataframe(synchronized_qualisys_df)
print(synchronized_qualisys_df.head())

freemocap_body_df = pd.read_csv(freemocap_body_csv)
new_freemocap_dataframe = reformat_freemocap_dataframe(freemocap_body_df)

qual_joints, qual_markers = main(new_qual_dataframe, joint_center_weights, qualisys_marker_mappings)



f = 2