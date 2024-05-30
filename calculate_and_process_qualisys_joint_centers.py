
from qualisys.joint_center_calculation.calculate_joint_centers import calculate_joint_centers
import pandas as pd
from pathlib import Path

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

from skeleton.create_skeleton import create_skeleton_model
from qualisys.qualisys_model_info import QualisysModelInfo

from freemocap.calculate_center_of_mass import calculate_center_of_mass_from_skeleton

import numpy as np


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


def create_generic_qualisys_marker_dataframe(qualisys_biomechanical_marker_dataframe: pd.DataFrame, qualisys_marker_mappings):

    flat_mappings = {}
    for joint, markers in qualisys_marker_mappings.items():
        for biomechanical_name, generic_name in markers.items():
            flat_mappings[biomechanical_name] = generic_name
    
    # Filter rows to keep only the markers that are in the flat_mappings dictionary
    qualisys_generic_marker_dataframe = qualisys_biomechanical_marker_dataframe[qualisys_biomechanical_marker_dataframe['marker'].isin(flat_mappings.keys())]

    # Replace the marker names in the DataFrame
    qualisys_generic_marker_dataframe['marker'] = qualisys_generic_marker_dataframe['marker'].replace(flat_mappings)

    return qualisys_generic_marker_dataframe


def main(original_qualisys_dataframe: pd.DataFrame, joint_center_weights: dict, qualisys_marker_mappings: dict):
    

    qualisys_generic_marker_dataframe = create_generic_qualisys_marker_dataframe(original_qualisys_dataframe, qualisys_marker_mappings)
    qualisys_markers_frame_marker_dimension = dataframe_to_numpy(qualisys_generic_marker_dataframe)
    marker_names = qualisys_generic_marker_dataframe['marker'].unique().tolist()
    joint_centers_frame_marker_dimension = calculate_joint_centers(qualisys_markers_frame_marker_dimension, joint_center_weights, marker_names)

    joint_centers_dataframe = pd.DataFrame(joint_centers_frame_marker_dimension.reshape(-1, 3), columns=['x', 'y', 'z'])
    joint_centers_dataframe['frame'] = np.repeat(np.arange(joint_centers_frame_marker_dimension.shape[0]), len(joint_center_weights))
    joint_centers_dataframe['marker'] = np.tile(list(joint_center_weights.keys()), joint_centers_frame_marker_dimension.shape[0])
    joint_centers_dataframe = joint_centers_dataframe[['frame', 'marker', 'x', 'y', 'z']]

    return joint_centers_frame_marker_dimension, qualisys_markers_frame_marker_dimension




if __name__ == '__main__':
    import numpy as np
    from qualisys.joint_center_calculation.generic_mappings.full_body_generic_marker_mapping import qualisys_marker_mappings
    from qualisys.joint_center_calculation.joint_center_weights.full_body_joint_center_weights import joint_center_weights

    # from qualisys.joint_center_calculation.generic_mappings.prosthetic_generic_mappings import qualisys_marker_mappings
    # from qualisys.joint_center_calculation.joint_center_weights.prosthetic_joint_center_weights import joint_center_weights
    
    from qualisys.qualisys_plotting import plot_3d_scatter

    # path_to_recording_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\mediapipe_MDN_Trial_2_yolo')
    # path_to_recording_folder = Path(r'D:\2023-05-17_MDN_NIH_data\1.0_recordings\calib_3\sesh_2023-05-17_13_48_44_MDN_treadmill_2')
    path_to_recording_folder = Path(r'D:\2024-04-25_P01\1.0_recordings\sesh_2024-04-25_14_45_59_P01_NIH_Trial1')
    path_to_qualisys_folder = path_to_recording_folder / 'qualisys_data'
    path_to_qualisys_csv = path_to_qualisys_folder / 'synchronized_qualisys_markers.csv'
    save_path = path_to_qualisys_folder / 'qualisys_joint_centers_3d_xyz.npy'
    
    center_of_mass_folder_path = path_to_qualisys_folder / 'center_of_mass'
    center_of_mass_folder_path.mkdir(parents=True, exist_ok=True)
    total_body_com_save_path = center_of_mass_folder_path/'total_body_center_of_mass_xyz.npy'
    segment_com_save_path = center_of_mass_folder_path / 'segmentCOM_frame_joint_xyz.npy'


    qualisys_dataframe = pd.read_csv(path_to_qualisys_csv)

    joint_centers_frame_marker_dimension,qualisys_markers_frame_marker_dimension = main(qualisys_dataframe, joint_center_weights, qualisys_marker_mappings)

    adjusted_settings = default_settings.copy()
    adjusted_settings[TASK_SKELETON_ROTATION][PARAM_AUTO_FIND_GOOD_FRAME] = False
    adjusted_settings[TASK_SKELETON_ROTATION][PARAM_GOOD_FRAME] = 450 

    post_process_task_worker = TaskWorkerThread(raw_skeleton_data=joint_centers_frame_marker_dimension, task_list= [TASK_INTERPOLATION, TASK_FILTERING, TASK_FINDING_GOOD_FRAME, TASK_SKELETON_ROTATION], settings=default_settings, landmark_names=QualisysModelInfo.landmark_names)
    post_process_task_worker.run()

    filt_interp_joint_centers_frame_marker_dimension = post_process_task_worker.tasks[TASK_SKELETON_ROTATION]['result']

        
    qualisys_skeleton = create_skeleton_model(
        actual_markers=QualisysModelInfo.landmark_names,
        num_tracked_points=QualisysModelInfo.num_tracked_points,
        segment_connections=QualisysModelInfo.segment_connections,
        virtual_markers=QualisysModelInfo.virtual_markers_definitions,
        center_of_mass_info=QualisysModelInfo.center_of_mass_definitions,
    )

    qualisys_skeleton.integrate_freemocap_3d_data(filt_interp_joint_centers_frame_marker_dimension)


    data_arrays_to_plot = {
        'qualisys markers': qualisys_markers_frame_marker_dimension,
        'qualisys joint centers': filt_interp_joint_centers_frame_marker_dimension,
    }
    
    qualisys_segment_com, qualisys_total_body_com = calculate_center_of_mass_from_skeleton(qualisys_skeleton)

    plot_3d_scatter(data_arrays_to_plot)
    np.save(save_path, filt_interp_joint_centers_frame_marker_dimension)
    np.save(total_body_com_save_path, qualisys_total_body_com)
    np.save(segment_com_save_path, qualisys_segment_com)

    print('Saved joint centers and center of mass data.')
    

    f = 2
 