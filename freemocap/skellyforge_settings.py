from skellyforge.freemocap_utils.constants import(
    TASK_INTERPOLATION,
    TASK_FILTERING,
    TASK_FINDING_GOOD_FRAME,
    TASK_SKELETON_ROTATION,
    PARAM_METHOD,
    PARAM_ORDER,
    PARAM_CUTOFF_FREQUENCY,
    PARAM_SAMPLING_RATE,
    PARAM_ROTATE_DATA,
    PARAM_AUTO_FIND_GOOD_FRAME,
    PARAM_GOOD_FRAME
)

default_settings = {
TASK_INTERPOLATION:
 {PARAM_METHOD: 'linear',
  PARAM_ORDER : 3
 },
 
 TASK_FILTERING: 
 {'Method' : 'Butterworth Low Pass Filter',
  PARAM_ORDER: 4,
  PARAM_CUTOFF_FREQUENCY: 6,
  PARAM_SAMPLING_RATE:30 
 },

 TASK_SKELETON_ROTATION:
 {PARAM_ROTATE_DATA: True,
  PARAM_AUTO_FIND_GOOD_FRAME: True,
  PARAM_GOOD_FRAME: None
 }
}

