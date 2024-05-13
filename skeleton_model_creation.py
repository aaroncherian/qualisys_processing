from skeleton.create_skeleton import create_skeleton_model
from qualisys.qualisys_model_info import QualisysModelInfo


qualisys_skeleton = create_skeleton_model(
    actual_markers=QualisysModelInfo.landmark_names,
    num_tracked_points=QualisysModelInfo.num_tracked_points,
    segment_connections=QualisysModelInfo.segment_connections,
    virtual_markers=QualisysModelInfo.virtual_markers_definitions,
    center_of_mass_info=QualisysModelInfo.center_of_mass_definitions,
)


