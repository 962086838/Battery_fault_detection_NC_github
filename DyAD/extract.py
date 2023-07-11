import json
import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import collate
from model import dataset
from train import extract
from utils import to_var, collate, Normalizer, PreprocessNormalizer
from model import tasks
import pickle

class Extraction:
    """
    feature extraction
    """

    def __init__(self, args, fold_num=0):
        """
        :param project: class model.projects.Project object
        """
        self.args = args
        self.fold_num = fold_num

    def main(self):
        """
        test: normalized test data
        task: task, e.g. EvTask、JeveTask
        model: model
        """
        model_params_path = os.path.join(self.args.current_model_path, "model_params.json")
        with open(model_params_path, 'r') as load_f:
            prams_dict = json.load(load_f)
        model_params = prams_dict['args']
        start_time = time.time()
        # data_pre = dataset.Dataset(model_params["test_path"])
        data_pre = dataset.Dataset(model_params["test_path"], train=False, fold_num=self.fold_num)
        self.normalizer = pickle.load(open(os.path.join(self.args.current_model_path, "norm.pkl"), 'rb'))
        test = PreprocessNormalizer(data_pre, normalizer_fn=self.normalizer.norm_func)

        task = tasks.Task(task_name=model_params["task"], columns=model_params["columns"])

        # load checkpoint
        model_torch = os.path.join(model_params["current_model_path"], "model.torch")
        model = to_var(torch.load(model_torch)).float()
        model.encoder_filter = task.encoder_filter
        model.decoder_filter = task.decoder_filter
        model.noise_scale = model_params["noise_scale"]
        data_loader = DataLoader(dataset=test, batch_size=model_params["batch_size"], shuffle=True,
                                 num_workers=model_params["jobs"], drop_last=False,
                                 pin_memory=torch.cuda.is_available(),
                                 collate_fn=collate if model_params["variable_length"] else None)

        print("sliding windows dataset length is: ", len(test))
        print("model", model)

        # extact feature
        model.eval()
        p_bar = tqdm(total=len(data_loader), desc='saving', ncols=100, mininterval=1, maxinterval=10, miniters=1)
        extract(data_loader, model, task, model_params["save_feature_path"], p_bar, model_params["noise_scale"],
                model_params["variable_length"])
        p_bar.close()
        print("Feature extraction of all test saved at", model_params["save_feature_path"])
        print("The total time consuming：", time.time() - start_time)


if __name__ == '__main__':
    import argparse

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--current_model_path', type=str,
                        default='/home/user/cleantest/2021-12-04-15-19-38/model/')
    args = parser.parse_args()
    Extraction(args).main()
