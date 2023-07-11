import os
import torch
from glob import glob
import numpy as np

class Dataset:
    '''
    If you want to use another vendor, just switch paths
    ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict1.npz.npy'
    ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict2.npz.npy'
    ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict3.npz.npy'
    '''
    def __init__(self, data_path, all_car_dict_path='../five_fold_utils/all_car_dict.npz.npy',
     ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict1.npz.npy',
     train=True, fold_num=0):
        ind_ood_car_dict = np.load(ind_ood_car_dict_path, allow_pickle=True).item()
        self.ind_car_num_list = ind_ood_car_dict['ind_sorted']
        # self.ind_car_num_list = [2, 193, 45, 73, 354]  # used for debug
        # self.ind_car_num_list = np.load(all_car_dict_path, allow_pickle=True).item()
        self.ood_car_num_list = ind_ood_car_dict['ood_sorted']
        # self.ood_car_num_list = [186, 204, 349, 236, 136]  # used for debug
        self.all_car_dict = np.load(all_car_dict_path, allow_pickle=True).item()

        if train:
            car_number = self.ind_car_num_list[
                         :int(fold_num * len(self.ind_car_num_list) / 5)] + self.ind_car_num_list[
                                                                            int((fold_num + 1) * len(
                                                                                self.ind_car_num_list) / 5):]
        else:  # test
            car_number = self.ind_car_num_list[
                         int(fold_num * len(self.ind_car_num_list) / 5):int(
                             (fold_num + 1) * len(self.ind_car_num_list) / 5)] + self.ood_car_num_list

        self.data_path = data_path
        self.battery_dataset = []

        print('car_number is ', car_number)

        for each_num in car_number:
            for each_pkl in self.all_car_dict[each_num]:
                train1 = torch.load(each_pkl)
                self.battery_dataset.append(train1)

    def __len__(self):
        return len(self.battery_dataset)

    def __getitem__(self, idx):
        file = self.battery_dataset[idx]
        return file

