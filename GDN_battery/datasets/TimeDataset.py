import torch
from torch.utils.data import Dataset, DataLoader

import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np


class TimeDataset(Dataset):
    def __init__(self, raw_data_path, edge_index, mode='train', config=None, fold_num=0, debug=False,
                 specific_car=None):
        self.debug = debug
        self.specific_car = specific_car
        self.raw_data_path = raw_data_path

        self.config = config
        self.edge_index = edge_index
        self.mode = mode
        self.fold_num = fold_num

        self.x, self.y, self.labels = self.process(raw_data_path)

    def __len__(self):
        return len(self.x)

    def process(self, raw_data_path):
        from glob import glob
        from tqdm import tqdm

        pkl_filepaths = []
        ind_ood_car_dict = np.load('../five_fold_utils/ind_odd_dict1.npz.npy', allow_pickle=True).item()
        self.ind_car_num_list = ind_ood_car_dict['ind_sorted']
        # self.ind_car_num_list = [2, 193, 45, 73, 354]  # used for debug
        # self.ind_car_num_list = np.load(all_car_dict_path, allow_pickle=True).item()
        self.ood_car_num_list = ind_ood_car_dict['ood_sorted'] 
        # self.ood_car_num_list = [186, 204, 349, 236, 136]  # used for debug
        self.all_car_dict = np.load('../five_fold_utils/all_car_dict.npz.npy',
                                    allow_pickle=True).item()
        if self.specific_car != None:
            print('using fold ', self.fold_num)
        _ind_car_num_list_len = len(self.ind_car_num_list)
        if self.mode == 'train':
            car_number = self.ind_car_num_list[:int(self.fold_num * _ind_car_num_list_len / 5)] \
                         + self.ind_car_num_list[int((self.fold_num + 1) * _ind_car_num_list_len / 5):]
        else:  # test
            car_number = self.ind_car_num_list[
                         int(self.fold_num * _ind_car_num_list_len / 5):int(
                             (self.fold_num + 1) * _ind_car_num_list_len / 5)] + self.ood_car_num_list

        if self.specific_car == None:
            pass
        else:
            car_number = [self.specific_car]

        for each_num in car_number:
            for each_pkl in self.all_car_dict[each_num]:
                if self.debug:
                    if len(pkl_filepaths) > 1000:
                        continue
                pkl_filepaths.append(each_pkl)  # since data are in the upper folder

        x_arr, y_arr = [], []
        labels_arr = []

        slide_win, slide_stride = [self.config[k] for k
                                   in ['slide_win', 'slide_stride']
                                   ]
        is_train = self.mode == 'train'

        _this_pkl_file = torch.load(pkl_filepaths[0])
        each_time_len = _this_pkl_file[0].shape[0]

        each_piece_rang = range(slide_win, each_time_len, slide_stride) if is_train else range(slide_win, each_time_len)

        print('loading data')
        for each_pkl_path in tqdm(pkl_filepaths):
            this_pkl_file = torch.load(each_pkl_path)
            this_pkl_file_label = int(this_pkl_file[1]['label'][0])  # '00' or '10'
            this_pkl_file_time_data = this_pkl_file[0].T[:6, :]  # (8, 128)
            for i in each_piece_rang:
                ft = this_pkl_file_time_data[:, i - slide_win:i]
                tar = this_pkl_file_time_data[:, i]
                x_arr.append(torch.FloatTensor(ft))
                y_arr.append(torch.FloatTensor(tar))
                labels_arr.append(this_pkl_file_label)

        x = torch.stack(x_arr).contiguous()
        y = torch.stack(y_arr).contiguous()

        labels = torch.Tensor(labels_arr).contiguous()

        return x, y, labels

    def __getitem__(self, idx):
        feature = self.x[idx].double()
        y = self.y[idx].double()

        edge_index = self.edge_index.long()

        label = self.labels[idx].double()

        return feature, y, label, edge_index
