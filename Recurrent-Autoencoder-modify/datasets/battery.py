import random
import numpy as np
# from utils.samplers import StratifiedSampler

import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset, Subset
from tqdm import tqdm


class BatteryDataLoader:
    def __init__(self, config, specific_car=None):
        self.config = config

        train_pkl_filepaths = []
        test_pkl_filepaths = []
        ind_ood_car_dict = np.load('../five_fold_utils/ind_odd_dict1.npz.npy', allow_pickle=True).item()
        self.ind_car_num_list = ind_ood_car_dict['ind_sorted']
        self.ood_car_num_list = ind_ood_car_dict['ood_sorted']
        self.all_car_dict = np.load('../five_fold_utils/all_car_dict.npz.npy',
                                    allow_pickle=True).item()
        if specific_car==None:
            print('using fold ', self.config.fold_num)
        _ind_car_num_list_len = len(self.ind_car_num_list)
        train_car_number = self.ind_car_num_list[:int(self.config.fold_num * _ind_car_num_list_len / 5)] \
                           + self.ind_car_num_list[int((self.config.fold_num + 1) * _ind_car_num_list_len / 5):]
        test_car_number = self.ind_car_num_list[
                          int(self.config.fold_num * _ind_car_num_list_len / 5):int(
                              (self.config.fold_num + 1) * _ind_car_num_list_len / 5)] + self.ood_car_num_list

        if self.config.debug==1:
            train_car_number = train_car_number[0:2]
            test_car_number = test_car_number[0:10]

        if specific_car==None:
            pass
        else:
            train_car_number = [specific_car]
            test_car_number = [specific_car]

        for each_num in train_car_number:
            for each_pkl in self.all_car_dict[each_num]:
                # if self.config.debug == 1:
                #     if len(train_pkl_filepaths) > 1000:
                #         continue
                train_pkl_filepaths.append(each_pkl)
        for each_num in test_car_number:
            for each_pkl in self.all_car_dict[each_num]:
                # if self.config.debug == 1:
                #     if len(test_pkl_filepaths) > 1000:
                #         continue
                test_pkl_filepaths.append(each_pkl)

        X_train, y_train = [], []
        X_test, y_test = [], []
        train_labels_arr = []
        test_labels_arr = []

        for each_pkl_path in tqdm(train_pkl_filepaths, desc='loading train pickle'):
            this_pkl_file = torch.load(each_pkl_path)  # since data are in the upper folder
            this_pkl_file_label = int(this_pkl_file[1]['label'][0])  # '00' or '10'
            this_pkl_file_time_data = this_pkl_file[0][:, :6]
            X_train.append(torch.FloatTensor(this_pkl_file_time_data))
            y_train.append(this_pkl_file_label)
            train_labels_arr.append(this_pkl_file_label)
        for each_pkl_path in tqdm(test_pkl_filepaths, desc='loading test pickle'):
            this_pkl_file = torch.load(each_pkl_path)  # # since data are in the upper folder
            this_pkl_file_label = int(this_pkl_file[1]['label'][0])  # '00' or '10'
            this_pkl_file_time_data = this_pkl_file[0][:, :6]
            X_test.append(torch.FloatTensor(this_pkl_file_time_data))
            y_test.append(this_pkl_file_label)
            test_labels_arr.append(this_pkl_file_label)

        X_train = torch.stack(X_train).contiguous()
        X_test = torch.stack(X_test).contiguous()

        y_train = torch.from_numpy(np.array(y_train))
        y_test = torch.from_numpy(np.array(y_test))

        # Tensordataset
        training = TensorDataset(X_train, y_train)
        test_set = TensorDataset(X_test, y_test)

        dataset_len = int(len(training))
        train_use_len = int(dataset_len * (1 - self.config.val_ratio))
        val_use_len = int(dataset_len * self.config.val_ratio)
        val_start_index = random.randrange(train_use_len)
        indices = torch.arange(dataset_len)

        train_sub_indices = torch.cat([indices[:val_start_index], indices[val_start_index + val_use_len:]])
        train_subset = Subset(training, train_sub_indices)

        val_sub_indices = indices[val_start_index:val_start_index + val_use_len]
        val_subset = Subset(training, val_sub_indices)

        print('train_subset length is ', len(train_subset))
        print('val_subset length is ', len(val_subset))
        print('test_subset length is ', len(test_set))

        # Dataloader
        if self.config.training_type == 'one_class':

            self.train_loader = DataLoader(train_subset, batch_size=self.config.batch_size, shuffle=True)
        else:
            raise NotImplementedError
            # sampler = StratifiedSampler(y_train,
            #                             batch_size=self.config.batch_size,
            #                             random_state=self.config.sampler_random_state)
            # self.train_loader = DataLoader(training, batch_sampler=sampler)

        self.valid_loader = DataLoader(val_subset, batch_size=self.config.batch_size_val, shuffle=False)
        self.test_loader = DataLoader(test_set, batch_size=self.config.batch_size_val, shuffle=False)

        # Number of batches
        self.train_iterations = len(self.train_loader)
        self.valid_iterations = len(self.valid_loader)

    def finalize(self):
        pass
