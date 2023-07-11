import os

def checkpoints_folder(config):
    """Create chekpoints_folder"""
 
    # Path of the folfer where to save checkpoint
    path = config.checkpoint_dir + config.exp_name
    
    if not os.path.exists(path):
        os.mkdir(path)
        print('Create checkpoints folder: ' + path)
    else:
        print('Checkpoints folder ' + path + ' already exists')
    return path + "/"

