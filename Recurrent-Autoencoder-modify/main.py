"""
__author__ = "Francesco Cannarile"
Main
-Capture the config file
-Process the json config passed
-Create an agent instance
-Run the agent
-Finalize
"""

import argparse
from utils.config import *
from agents import *

import warnings
warnings.filterwarnings("ignore")


def main():
    arg_parser = argparse.ArgumentParser(description = 'Configuration path')
    arg_parser.add_argument('config', help = 'The Configuration file in json format')
    args = arg_parser.parse_args()
    config = process_config(args.config)
    print(config.agent)
    agent_class = globals()[config.agent]
    agent = agent_class(config)
    agent.run()
    agent.finalize()
    agent.save_error()
    
if __name__ == '__main__':
    main()
