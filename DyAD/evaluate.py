import os
import sys
import numpy as np
import pandas as pd
import torch
from pyod.models.iforest import IForest
from sklearn.metrics import confusion_matrix
from tqdm import tqdm


class Evaluate:
    def __init__(self, args):
        """
        :param project: class model.projects.Project object
        """
        self.args = args

    @staticmethod
    def get_feature_label(data_path, max_group=None):
        if max_group is None:
            max_group = len(data_path) // 2
        data, label = [], []
        for f in tqdm(sorted(os.listdir(data_path), key=lambda x: x.split('_')[0])):
            if int(f.split('_')[0]) > max_group:
                break
            else:
                if f.endswith(".file"):
                    temp_label = torch.load(open(os.path.join(data_path, f), 'rb'))
                    temp_label['car']=temp_label['car'].numpy()
                    label += np.array(
                        [[i[0] for i in temp_label['label']], temp_label['car'], temp_label['rec_error']]).T.tolist()
                elif f.endswith(".npy"):
                    data += np.load(os.path.join(data_path, f)).tolist()

        return np.array(data), np.array(label)

    @staticmethod
    def calculate_rec_error(_, label):
        rec_sorted_index = np.argsort(-label[:, 2].astype(float))
        res = [label[i][[1, 0, 2]] for i in rec_sorted_index]
        return pd.DataFrame(res, columns=['car', 'label', 'rec_error'])

    def main(self):
        x, label = self.get_feature_label(self.args.feature_path, max_group=20000)
        print("Loading feature is :", x.shape)
        print("Loading label is :", label.shape)

        result = eval('self.calculate_' + self.args.use_flag)(x, label)
        result.to_csv(os.path.join(self.args.result_path, "train_segment_scores.csv"))

        x, label = self.get_feature_label(self.args.save_feature_path, max_group=20000)
        print("Loading test feature is :", x.shape)
        print("Loading test label is :", label.shape)

        result = eval('self.calculate_' + self.args.use_flag)(x, label)
        result.to_csv(os.path.join(self.args.result_path, "test_segment_scores.csv"))


if __name__ == '__main__':
    import argparse
    import json
    #from anomaly_detection.model import projects

    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    parser = argparse.ArgumentParser(description='Train Example')
    parser.add_argument('--modelparams_path', type=str,
                        default=os.path.join( '/home/user/cleantest/2021-12-04-15-19-38/model','model_params.json'))

    args = parser.parse_args()

    with open(args.modelparams_path, 'r') as file:
        p_args = argparse.Namespace()
        model_params=json.load(file)
        p_args.__dict__.update(model_params["args"])
        args = parser.parse_args(namespace=p_args)
    print("Loaded configs at %s" % args.modelparams_path)
    print("args", args)
    Evaluate(args).main()
