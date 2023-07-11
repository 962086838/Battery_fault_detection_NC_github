import json
from easydict import EasyDict

def get_config_from_json(json_file):
    """
    Get the config from a json file
    """
    # parse the configurations from the config json file provided
    with open(json_file, 'r') as config_file:
        try:
            config_dict = json.load(config_file)
            # EasyDict allows to access dict values as attributes (works recursively).
            config = EasyDict(config_dict)
            return config, config_dict
        except ValueError:
            print("INVALID JSON file format.. Please provide a good json file")
            exit(-1)

def process_config(json_file):
    """
    Get the json file
    Processing it with EasyDict to be accessible as attributes
    then editing the path of the experiments folder
    creating some important directories in the experiment folder
    Then setup the logging in the whole program
    """
    config, _ = get_config_from_json(json_file)
    
    # making sure that you have provided the exp_name.
    try:
        print(" *************************************** ")
        print("Experiment name: {}".format(config.exp_name))
        print(" *************************************** ")
    except AttributeError:
        print("ERROR!!..Please provide the exp_name in json file..")
        exit(-1)
    return config

def save_config(dict_file, save_path):
    """Save the configuration"""

    myJSON = json.dumps(dict_file)
    with open(save_path + "exp_config.json", "w") as jsonfile:
        jsonfile.write(myJSON)
    print('Saved configuration in ' + save_path )
   

