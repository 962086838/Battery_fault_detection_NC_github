import traceback

import torch
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy import interpolate as ip

def config_valid(config):
    config = vars(config)
    keys = config.keys()
    try:
        assert 'anneal0' in keys and type(config['anneal0']) == float
        assert 'anneal_function' in keys and type(config['anneal_function']) == str and config['anneal_function'] in [
            'logistic', 'linear']
        assert 'batch_size' in keys and type(config['batch_size']) == int
        assert 'bidirectional' in keys and type(config['bidirectional']) == bool
        assert 'cell_level' in keys and type(config['cell_level']) == bool
        assert 'config_path' in keys and type(config['config_path']) == str
        assert 'cosine_factor' in keys and type(config['cosine_factor']) == float
        assert 'dim_feedforward' in keys and type(config['dim_feedforward']) == int
        assert 'epochs' in keys and type(config['epochs']) == int
        assert 'evaluation_path' in keys and type(config['evaluation_path']) == str
        assert 'hidden_size' in keys and type(config['hidden_size']) == int
        assert 'interpolate' in keys and type(config['interpolate']) == int
        assert 'interval' in keys and type(config['interval']) == int
        assert 'jobs' in keys and type(config['jobs']) == int
        assert 'k' in keys and type(config['k']) == float
        assert 'kernel_size' in keys and type(config['kernel_size']) == int
        assert 'latent_label_weight' in keys and isinstance(config['latent_label_weight'], (int, float))
        assert 'latent_size' in keys and type(config['latent_size']) == int
        assert 'learning_rate' in keys and type(config['learning_rate']) == float
        assert 'model_type' in keys and type(config['model_type']) == str and config['model_type'] in ["rnn",
                                                                                                       "transformer"]
        assert 'nhead' in keys and type(config['nhead']) == int
        assert 'nll_weight' in keys and isinstance(config['nll_weight'], (int, float))
        assert 'noise_scale' in keys and type(config['noise_scale']) == float
        assert 'norm' in keys and type(config['norm']) == str
        assert 'num_layers' in keys and type(config['num_layers']) == int
        assert 'project' in keys and type(config['project']) == str
        assert 'ram' in keys and type(config['ram']) == bool
        assert 'rnn_type' in keys and type(config['rnn_type']) == str and config['rnn_type'] in ['rnn', 'lstm', 'gru']
        assert 'save_model_path' in keys and type(config['save_model_path']) == str
        assert 'smoothing' in keys and type(config['smoothing']) == bool
        assert 'task' in keys and type(config['task']) == str
        assert 'test_path' in keys and type(config['test_path']) == str
        assert 'train_path' in keys and type(config['train_path']) == str
        assert 'use_flag' in keys and type(config['use_flag']) == str and config['use_flag'] in ["rec_error", "l2norm",
                                                                                                 "copod_score"]
 
        assert 'x0' in keys and type(config['x0']) == int
        assert 'variable_length' in keys and type(config['variable_length']) == bool
        assert 'min_length' in keys and type(config['min_length']) == int
        assert 'granularity_all' in keys and type(config['granularity_all']) == int
        assert 'num_granularity_all' in keys and type(config['num_granularity_all']) == int
        assert 'granularity_car' in keys and type(config['granularity_car']) == int
        assert 'num_granularity_car' in keys and type(config['num_granularity_car']) == int
        print('valid config')
        return True
    except AssertionError as _:
        print('invalid config')
        traceback.print_exc()
        return False


def to_var(x):
    """
    var to gpu
    :param x: data or model
    :return: x
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def collate(batch_data):
    """
    args:
        batch_data - list of (tensor, metadata)

    return:
        (padded_sent_seq, data_lengths), metadata

    """

    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)
    seq_lengths = [len(xi[0]) for xi in batch_data]
    max_len = max(seq_lengths)

    sent_seq = [torch.FloatTensor(v[0]) for v in batch_data]
    # print('sent_seq:/n',sent_seq)
    metadata_list = [xi[1] for xi in batch_data]
    metadata = OrderedDict([('label', []), ('car', []), ('charge_segment', []), ('mileage', []), ('timestamp', [])])
    for i in range(len(metadata_list)):
        for key, value in metadata_list[i].items():
            metadata[key].append(value)

    padded_sent_seq = torch.FloatTensor([pad_tensor(v, max_len) for v in sent_seq])
    metadata['seq_lengths'] = seq_lengths
    return padded_sent_seq, metadata



class Normalizer:
    def __init__(self, dfs=None, variable_length=False):
        self.max_norm = 0
        self.min_norm = 0
        self.std = 0
        self.mean = 0
        res = []
        if dfs is not None:
            if variable_length:
                norm_length = min([len(df) for df in dfs])
                dfs = [df[0:norm_length] for df in dfs]
            res.extend(dfs)
            res = np.array(res)
            self.compute_min_max(res)
        else:
            raise Exception("df list not specified")

    def compute_min_max(self, res):
        column_max_all = np.max(res, axis=1)
        column_min_all = np.min(res, axis=1)
        column_std_all = np.std(res, axis=1)
        column_mean_all = np.mean(res, axis=1)
        self.max_norm = np.max(column_max_all, axis=0)
        self.min_norm = np.min(column_min_all, axis=0)
        self.std = np.mean(column_std_all, axis=0)
        self.mean = np.mean(column_mean_all, axis=0)

    def std_norm_df(self, df):
        return (df - self.mean) / np.maximum(1e-4, self.std)

    def norm_func(self, df):
        df_norm = df.copy()
        df_norm = (df_norm - self.mean) / np.maximum(np.maximum(1e-4, self.std), 0.1 * (self.max_norm - self.min_norm))
        return df_norm


class PreprocessNormalizer:

    def __init__(self, dataset, normalizer_fn=None):
        self.dataset = dataset
        self.normalizer_fn = normalizer_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        df, metadata = self.dataset[idx][0], self.dataset[idx][1]
        if self.normalizer_fn is not None:
            df = self.normalizer_fn(df)
        return df, metadata








