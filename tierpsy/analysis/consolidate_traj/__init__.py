from .consolidateTrajectories import consolidateTrajectories

def args_(fn, param):
    return {
        'func': consolidateTrajectories,
        'argkws': {
            'skeletons_file': fn['skeletons'],
        },
        'input_files': [fn['skeletons']],
        'output_files': [fn['skeletons']],
        'requirements': ['TRAJ_JOIN'],
    }
