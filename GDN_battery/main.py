# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split, Subset

from sklearn.preprocessing import MinMaxScaler

from util.env import get_device, set_device
from util.preprocess import build_loc_net, construct_data
from util.net_struct import get_feature_map, get_fc_graph_struc
from util.iostream import printsep

from datasets.TimeDataset import TimeDataset

from models.GDN import GDN

from train import train
from test import test
from evaluate import get_err_scores, get_best_performance_data, get_val_performance_data, get_full_err_scores
import sys
from datetime import datetime
import time

import os
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

import json
import random

from sklearn.metrics import mean_squared_error, r2_score
from tqdm import tqdm


class Main():
    def __init__(self, train_config, env_config, debug=False):

        self.train_config = train_config
        self.env_config = env_config
        self.datestr = None

        dataset = self.env_config['dataset']
       
        feature_map = ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp']
        fc_struc = get_fc_graph_struc(dataset)

        set_device(env_config['device'])
        self.device = get_device()

        fc_edge_index = build_loc_net(fc_struc,
                                      ['volt', 'current', 'soc', 'max_single_volt', 'min_single_volt', 'max_temp'],
                                      feature_map=feature_map)
        fc_edge_index = torch.tensor(fc_edge_index, dtype=torch.long)
        self.fc_edge_index = fc_edge_index

        self.feature_map = feature_map

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        train_dataset = TimeDataset('../data/new1/train_mulmileage', fc_edge_index,
                                                             mode='train',
                                                             config=cfg, fold_num=args.fold_num, debug=False)

        train_dataloader, val_dataloader = self.get_loaders(train_dataset, train_config['seed'], train_config['batch'],
                                                            val_ratio=train_config['val_ratio'])

        self.train_dataset = train_dataset
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader

        edge_index_sets = []
        edge_index_sets.append(fc_edge_index)

        self.model = GDN(edge_index_sets, len(feature_map),
                         dim=train_config['dim'],
                         input_dim=train_config['slide_win'],
                         out_layer_num=train_config['out_layer_num'],
                         out_layer_inter_dim=train_config['out_layer_inter_dim'],
                         topk=train_config['topk']
                         ).to(self.device)

    def run(self):

        if len(self.env_config['load_model_path']) > 0:
            model_save_path = self.env_config['load_model_path']
        else:
            model_save_path = self.get_save_path()[0]

            self.train_log = train(self.model, model_save_path,
                                   config=train_config,
                                   train_dataloader=self.train_dataloader,
                                   val_dataloader=None if args.val_loader==0 else self.val_dataloader,
                                   feature_map=self.feature_map,
                                   # test_dataloader=self.test_dataloader,
                                   # test_dataset=self.test_dataset,
                                   test_dataloader=None,
                                   test_dataset=None,
                                   train_dataset=self.train_dataset,
                                   dataset_name=self.env_config['dataset']
                                   )

        print('progress flag 1')
        # test
        self.model.load_state_dict(torch.load(model_save_path))
        best_model = self.model.to(self.device)

        # reload all train data and test data
        ind_ood_car_dict = np.load('../five_fold_utils/ind_odd_dict1.npz.npy', allow_pickle=True).item()
        ind_car_num_list = ind_ood_car_dict['ind_sorted']
        # self.ind_car_num_list = [2, 193, 45, 73, 354]  # used for debug
        # self.ind_car_num_list = np.load(all_car_dict_path, allow_pickle=True).item()
        ood_car_num_list = ind_ood_car_dict['ood_sorted']

        cfg = {
            'slide_win': train_config['slide_win'],
            'slide_stride': train_config['slide_stride'],
        }

        df = pd.DataFrame()

        all_car_num_list = []
        all_label_list = []
        all_rec_error_list = []

        for each_car in tqdm(ind_car_num_list + ood_car_num_list):
            onecar_dataset = TimeDataset('../data/new1/train_mulmileage', self.fc_edge_index,
                                                      mode='train',
                                                      config=cfg, fold_num=args.fold_num, debug=False,
                                                      specific_car=each_car)
            onecar_dataloader = DataLoader(onecar_dataset, batch_size=train_config['batch'],
                                           shuffle=False, num_workers=0)
            _, onecar_result = test(best_model, onecar_dataloader)
            onecar_result = np.array(onecar_result)
            # print(onecar_result.shape)  # (3, 16416, 6)
            onecar_result = onecar_result.reshape(onecar_result.shape[0], -1,
                                                  int((128 - cfg['slide_win']) / cfg['slide_stride']),
                                                  onecar_result.shape[-1])
            # print(onecar_result.shape)  # (3, 2736, 6, 6)  # predict, y, car_label
            onecar_result_predict, onecar_result_groundtruth = onecar_result[0], onecar_result[1]
            oncar_reconstruction_error = [
                float(mean_squared_error(onecar_result_predict[i], onecar_result_groundtruth[i])) for i in
                range(onecar_result_predict.shape[0])]
            # print(len(oncar_reconstruction_error))
            all_car_num_list += [each_car] * len(oncar_reconstruction_error)
            if each_car in ind_car_num_list:
                all_label_list += [0] * len(oncar_reconstruction_error)
            else:
                all_label_list += [1] * len(oncar_reconstruction_error)
            all_rec_error_list += oncar_reconstruction_error
            print(len(all_car_num_list), len(all_label_list), len(all_rec_error_list))

        df['car'] = all_car_num_list
        df['label'] = all_label_list
        df['rec_error'] = all_rec_error_list
        time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        os.makedirs('./rec_error', exist_ok=True)
        df.to_csv('./rec_error/gdn_saved_rec_error_fold%d_%s.csv' % (args.fold_num, time_now))

    def get_loaders(self, train_dataset, seed, batch, val_ratio=0.1):
        dataset_len = int(len(train_dataset))
        train_use_len = int(dataset_len * (1 - val_ratio))
        val_use_len = int(dataset_len * val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
        train_subset = Subset(train_dataset, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
        val_subset = Subset(train_dataset, val_sub_indices)

        print('train_subset length is ', len(train_subset))
        print('val_subset length is ', len(val_subset))

        train_dataloader = DataLoader(train_subset, batch_size=batch,
                                      shuffle=True)

        val_dataloader = DataLoader(val_subset, batch_size=batch,
                                    shuffle=False)

        return train_dataloader, val_dataloader

    def get_score(self, test_result, val_result):

        feature_num = len(test_result[0][0])
        np_test_result = np.array(test_result)
        np_val_result = np.array(val_result)

        test_labels = np_test_result[2, :, 0].tolist()  # 0, 1 label from car

        test_scores, normal_scores = get_full_err_scores(test_result, val_result)
        time_now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
        np.save('test_scores_bsz%d_winlgth%d_winstrd%d_fold%d_%s.npy' % (
            args.batch, args.slide_win, args.slide_stride, args.fold_num, time_now),
                test_scores)
        np.save('normal_scores_bsz%d_winlgth%d_winstrd%d_fold%d_%s.npy' % (
            args.batch, args.slide_win, args.slide_stride, args.fold_num, time_now),
                normal_scores)
        print('get_full_err_scores done', time_now)
        # print('im main, test_scores shape', np.array(test_scores).shape) # im main, test_scores shape (27, 2044)

        top1_best_info = get_best_performance_data(test_scores, test_labels, topk=1)
        top1_val_info = get_val_performance_data(test_scores, normal_scores, test_labels, topk=1)

        print('=========================** Result **============================\n')

        info = None
        if self.env_config['report'] == 'best':
            info = top1_best_info
        elif self.env_config['report'] == 'val':
            info = top1_val_info

        print(f'F1 score: {info[0]}')
        print(f'precision: {info[1]}')
        print(f'recall: {info[2]}\n')
    def get_save_path(self, feature_name=''):

        dir_path = self.env_config['save_path']

        if self.datestr is None:
            now = datetime.now()
            self.datestr = now.strftime('%m|%d-%H:%M:%S')
        datestr = self.datestr

        paths = [f'./pretrained/{dir_path}/best_{datestr}.pt',
                 f'./results/{dir_path}/{datestr}.csv']

        for path in paths:
            dirname = os.path.dirname(path)
            Path(dirname).mkdir(parents=True, exist_ok=True)

        return paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', help='batch size', type=int, default=128)
    parser.add_argument('-epoch', help='train epoch', type=int, default=100)
    parser.add_argument('-slide_win', help='slide_win', type=int, default=15)
    parser.add_argument('-dim', help='dimension', type=int, default=64)
    parser.add_argument('-slide_stride', help='slide_stride', type=int, default=5)
    parser.add_argument('-save_path_pattern', help='save path pattern', type=str, default='')
    parser.add_argument('-dataset', help='wadi / swat', type=str, default='wadi')
    parser.add_argument('-device', help='cuda / cpu', type=str, default='cuda')
    parser.add_argument('-random_seed', help='random seed', type=int, default=0)
    parser.add_argument('-comment', help='experiment comment', type=str, default='')
    parser.add_argument('-out_layer_num', help='outlayer num', type=int, default=1)
    parser.add_argument('-out_layer_inter_dim', help='out_layer_inter_dim', type=int, default=256)
    parser.add_argument('-decay', help='decay', type=float, default=0)
    parser.add_argument('-val_ratio', help='val ratio', type=float, default=0.1)
    parser.add_argument('-topk', help='topk num', type=int, default=20)
    parser.add_argument('-report', help='best / val', type=str, default='best')
    parser.add_argument('-load_model_path', help='trained model path', type=str, default='')
    parser.add_argument('--fold_num', help='fold_num', type=int, default=0)
    parser.add_argument('--val_loader', help='val_loader', type=int, default=1)  # use val_loader

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(args.random_seed)

    train_config = {
        'batch': args.batch,
        'epoch': args.epoch,
        'slide_win': args.slide_win,
        'dim': args.dim,
        'slide_stride': args.slide_stride,
        'comment': args.comment,
        'seed': args.random_seed,
        'out_layer_num': args.out_layer_num,
        'out_layer_inter_dim': args.out_layer_inter_dim,
        'decay': args.decay,
        'val_ratio': args.val_ratio,
        'topk': args.topk,
    }

    env_config = {
        'save_path': args.save_path_pattern,
        'dataset': args.dataset,
        'report': args.report,
        'device': args.device,
        'load_model_path': args.load_model_path
    }

    main = Main(train_config, env_config, debug=False)
    main.run()

