from .detectCameraAdjustment import detectCameraAdjustment

def args_(fn, param):
    p = param.p_dict
    return {
        'func': detectCameraAdjustment,
        'argkws': {
            'masked_image_file': fn['masked_image'],
            'skeletons_file':    fn['skeletons'],
            'cam_adjust_thresh':       p.get('cam_adjust_thresh', 5.0),
            'cam_adjust_extra_frames': p.get('cam_adjust_extra_frames', 1),
        },
        'input_files':  [fn['masked_image'], fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements': ['TRAJ_CREATE'],
    }
