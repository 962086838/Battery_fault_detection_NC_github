import argparse
import json
import os
import sys
import numpy as np
import torch

from pyod.models.deep_svdd import DeepSVDD
from pyod.models.auto_encoder_torch import AutoEncoder, check_array, inner_autoencoder, check_is_fitted, \
    pairwise_distances_no_broadcast
from pyod.utils.data import evaluate_print


class PyODDataset(torch.utils.data.Dataset):
    """PyOD Dataset class for PyTorch Dataloader
    """

    def __init__(self, X, y=None, mean=None, std=None):
        super(PyODDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        # if self.mean.any():
        #     sample = (sample - self.mean) / (self.std + 1e-5)

        return torch.from_numpy(sample), idx


class bug_fixed_AutoEncoder(AutoEncoder):
    def fit(self, X, y=None):
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        n_samples, n_features = X.shape[0], X.shape[1]

        # conduct standardization if needed
        if self.preprocessing:
            self.mean, self.std = np.mean(X, axis=0), np.std(X, axis=0)
            train_set = PyODDataset(X=X, mean=self.mean, std=self.std)

        else:
            train_set = PyODDataset(X=X)

        train_loader = torch.utils.data.DataLoader(train_set,
                                                   batch_size=self.batch_size,
                                                   shuffle=True)

        # initialize the model
        self.model = inner_autoencoder(
            n_features=n_features,
            hidden_neurons=self.hidden_neurons,
            dropout_rate=self.dropout_rate,
            batch_norm=self.batch_norm,
            hidden_activation=self.hidden_activation)

        # move to device and print model information
        self.model = self.model.to(self.device)
        print(self.model)

        # train the autoencoder to find the best one
        self._train_autoencoder(train_loader)

        self.model.load_state_dict(self.best_model_dict)
        self.decision_scores_ = self.decision_function(X)

        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model', 'best_model_dict'])
        # X = check_array(X)

        # note the shuffle may be true but should be False
        if self.preprocessing:
            dataset = PyODDataset(X=X, mean=self.mean, std=self.std)
        else:
            dataset = PyODDataset(X=X)

        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_size=self.batch_size,
                                                 shuffle=False)
        # enable the evaluation mode
        self.model.eval()

        # construct the vector for holding the reconstruction error
        outlier_scores = np.zeros([X.shape[0], ])
        with torch.no_grad():
            for data, data_idx in dataloader:
                data_cuda = data.to(self.device).float()
                # this is the outlier score
                outlier_scores[data_idx] = pairwise_distances_no_broadcast(
                    data, self.model(data_cuda).cpu().numpy())

        return outlier_scores


def load_dataset(fold_num, train=True):
    ind_ood_car_dict = np.load('../five_fold_utils/ind_odd_dict1.npz.npy', allow_pickle=True).item()
    ind_car_num_list = ind_ood_car_dict['ind_sorted']
    # self.ind_car_num_list = [2, 193, 45, 73, 354]  # used for debug
    # self.ind_car_num_list = np.load(all_car_dict_path, allow_pickle=True).item()
    ood_car_num_list = ind_ood_car_dict['ood_sorted']
    all_car_dict = np.load('../five_fold_utils/all_car_dict.npz.npy', allow_pickle=True).item()

    if train:
        car_number = ind_car_num_list[:int(fold_num * len(ind_car_num_list) / 5)] + ind_car_num_list[
                                                                  int((fold_num + 1) * len(ind_car_num_list) / 5):]
    else:  # test
        car_number = ind_car_num_list[
                     int(fold_num * len(ind_car_num_list) / 5):int((fold_num + 1) * len(ind_car_num_list) / 5)] + ood_car_num_list

    X = []
    y = []

    print('car_number is ', car_number)

    for each_num in car_number:
        for each_pkl in all_car_dict[each_num]:
            train1 = torch.load(each_pkl)
            X.append(train1[0][:, 0:6].reshape(1, -1))
            y.append(int(train1[1]['label'][0]))

    X = np.vstack(X)
    y = np.vstack(y)

    return X, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch battery Example')
    parser.add_argument('--method', type=str, default='auto_encoder')
    parser.add_argument('--normalize', action='store_true')  # normalize data
    parser.add_argument('--max_normalize', action='store_true')  # normalize data
    parser.add_argument('--fold_num', type=int, default=0)

    args = parser.parse_args()


    X_train, y_train = load_dataset(fold_num=args.fold_num, train=True)
    X_test, y_test = load_dataset(fold_num=args.fold_num, train=False)

    if args.normalize:
        _mean = np.mean(X_train, axis=0)
        _std = np.std(X_train, axis=0)
        X_train = (X_train - _mean) / (_std + 1e-4)
        X_test = (X_test - _mean) / (_std + 1e-4)
    if args.max_normalize:
        _max = np.max(X_train, axis=0)
        _min = np.min(X_train, axis=0)
        X_train = (X_train - _min) / (_max - _min + 1e-4)
        X_test = (X_test - _min) / (_max - _min + 1e-4)


    print('loaded data')
    print('X_train shape', X_train.shape, 'X_test shape', X_test.shape)

    if args.method == 'auto_encoder':
        clf_name = 'auto_encoder'
        clf = bug_fixed_AutoEncoder(hidden_neurons=[64, 32, 32, 64], preprocessing=True, batch_size=128, epochs=20,
                                    hidden_activation='sigmoid')
        clf.fit(X_train)

        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores


        save_prefix = ''
        save_prefix = save_prefix + clf_name + '_'
        if args.normalize:
            save_prefix = save_prefix + 'normalize_'
        if args.max_normalize:
            save_prefix = save_prefix + 'max_normalize_'
        save_prefix = save_prefix + 'fold%d_' % args.fold_num
        
        os.makedirs('traditional_save', exist_ok=True)
        np.save('traditional_save/' + save_prefix + 'y_train_pred.npy', y_train_pred)
        np.save('traditional_save/' + save_prefix + 'y_train_scores.npy', y_train_scores)

        # get the prediction on the test data
        y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(X_test)  # outlier scores
        np.save('traditional_save/' + save_prefix + 'y_test_pred.npy', y_test_pred)
        np.save('traditional_save/' + save_prefix + 'y_test_scores.npy', y_test_scores)
        np.save('traditional_save/' + save_prefix + 'y_test.npy', y_test)

    elif args.method == 'deepsvdd':
        clf_name = 'DeepSVDD'
        clf = DeepSVDD(epochs=10, batch_size=64, use_ae=True, preprocessing=False)
        clf.fit(X_train)

        y_train_pred = clf.labels_  # binary labels (0: inliers, 1: outliers)
        y_train_scores = clf.decision_scores_  # raw outlier scores

        save_prefix = ''
        save_prefix = save_prefix + clf_name + '_'
        if args.normalize:
            save_prefix = save_prefix + 'normalize_'
        if args.max_normalize:
            save_prefix = save_prefix + 'max_normalize_'
        save_prefix = save_prefix + 'fold%d_' % args.fold_num
        
        os.makedirs('traditional_save', exist_ok=True)
        np.save('traditional_save/' + save_prefix + 'y_train_pred.npy', y_train_pred)
        np.save('traditional_save/' + save_prefix + 'y_train_scores.npy', y_train_scores)

        # get the prediction on the test data
        y_test_pred = clf.predict(X_test)  # outlier labels (0 or 1)
        y_test_scores = clf.decision_function(X_test)  # outlier scores
        np.save('traditional_save/' + save_prefix + 'y_test_pred.npy', y_test_pred)
        np.save('traditional_save/' + save_prefix + 'y_test_scores.npy', y_test_scores)
        np.save('traditional_save/' + save_prefix + 'y_test.npy', y_test)
        
    else:
        raise NotImplementedError

    # print("Loaded configs at %s" % args.config_path)
    # print("args", args)
