import odditylib as od
import torch
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--fold_num', help='fold_num', type = int, default=0)
parser.add_argument('--dataset', help='dataset', type = str, default='brand1')
parser.add_argument('--start', help='start', type = int, default=0)
parser.add_argument('--end', help='end', type = int, default=0)


args = parser.parse_args()

# Reading data
fold_num = args.fold_num
all_car_dict = np.load('../five_fold_utils/all_car_dict.npz.npy', allow_pickle=True).item()
# print(all_car_dict)
# assert 1==0

# brand1
if args.dataset=='brand1':
    ind_car_num_list = [129, 158, 152, 79, 95
        , 114, 34, 177, 99, 138, 163, 54, 45, 115, 66, 87, 47, 57, 31, 195, 36, 102,
                        72, 173, 94, 51, 92, 61, 153, 125, 103, 3, 50, 10, 7, 146, 166, 48, 75, 86, 15, 175, 64, 2, 110, 13,
                        23, 93, 116, 62, 8, 41, 22, 6, 24, 101, 46, 187, 198, 142, 131, 18, 160, 56, 29, 141, 148, 168, 71,
                        53, 104, 120, 154, 20, 17, 111, 133, 63, 35, 83, 5, 88, 159, 145, 176, 127, 77, 118, 52, 81, 121,
                        59, 38, 80, 109, 179, 28, 123, 44, 180, 149, 135, 164, 74, 40, 14, 65, 69, 42, 193, 12, 60, 73, 126,
                        161, 188, 32, 30, 170, 128, 167, 9, 155, 156, 43, 100, 33, 90, 139, 1, 112, 25, 4, 16, 189, 147,
                        124, 178, 55, 85, 122, 96, 162, 132, 89, 19, 27, 84, 39, 151, 67, 26, 172, 76, 37, 143, 58, 165, 97,
                        134, 82, 113, 137, 144, 70, 11, 117, 106]  # 169
    ood_car_num_list = [91, 192, 169, 130, 140
        , 171, 190, 186, 105, 49, 181, 157, 183, 185, 194, 98, 191, 136, 119, 196, 107, 68, 108, 78, 182, 150, 174, 21, 184, 197]
elif args.dataset=='brand2':
    # brand2
    ind_car_num_list = [214, 231, 233, 234, 218, 201, 211, 248, 222, 203, 223, 246, 229, 249, 227, 207, 232, 250, 208, 245,
                        213, 228, 220, 244, 217, 238, 221, 224, 226, 216, 202, 242, 235]  # 33
    ood_car_num_list = [205, 247, 241, 204, 206, 210, 243, 240, 219, 225, 209, 237, 236, 212, 215, 239]
elif args.dataset == 'brand3':
    # brand3
    ind_car_num_list = [411, 410, 434, 449, 423, 414, 431, 485, 446, 466, 453, 480, 455, 488, 402, 439, 427, 409, 442,
                        448, 428, 452, 429, 484, 482, 457, 459, 499, 413, 461, 403, 470, 481, 493, 417, 496, 407, 495,
                        418, 426, 436, 491, 500, 474, 476, 487, 430, 451, 498, 401, 463, 494, 406, 420, 497, 433, 435,
                        440, 416, 464, 445, 479, 425, 460, 490, 478, 467, 447, 412, 489, 444, 422, 477, 437, 415, 486,
                        441, 421, 471, 432, 483, 450, 468, 443, 456, 469, 472, 438, 408, 458, 454]  # 91
    ood_car_num_list = [424, 419, 473, 462, 492, 465, 404, 405, 475]



train_car_number = ind_car_num_list[args.start:  args.start+1]
train_X = []
train_y = []
ind_snippet_number = 0
ood_snippet_number = 0
for each_num in train_car_number:
    for each_pkl in all_car_dict[each_num]:
        if each_num in ind_car_num_list:
            ind_snippet_number += 1
        else:
            ood_snippet_number += 1
        train1 = torch.load(each_pkl)
        train_X.append(np.array(train1[0])[:, 0:6].reshape(1, 128, 6))  # (128, 6)
        train_y.append(int(train1[1]['label'][0]))

train_X = np.concatenate(train_X, axis=0)
train_y = np.vstack(train_y)

print(train_X.shape)
print(train_y.shape)

y_train_scores = []
for each_X, each_y in tqdm(zip(train_X, train_y)):
    fit_error = 0
    for each_channel_num in range(each_X.shape[1]):
        each_data = each_X[:, each_channel_num]
        detector = od.Oddity()  # Creating a default Oddity detector
        detector.fit(each_data.reshape(-1, 1))  # Fitting the detector on our data
        mu, cov = detector.mu, detector.cov
        fit_error += sum((mu - each_data) ** 2)
    y_train_scores.append(fit_error)
os.makedirs("gaussian_process", exist_ok=True)
os.makedirs(f"gaussian_process/{args.dataset}", exist_ok=True)
np.save(f"gaussian_process/{args.dataset}/" + f'y_train_scores_fold{args.fold_num}_start{args.start}.npy', y_train_scores)




test_car_number = ood_car_num_list[args.start:  args.start+1 ]
test_X = []
test_y = []
ind_snippet_number = 0
ood_snippet_number = 0
for each_num in test_car_number:
    for each_pkl in all_car_dict[each_num]:
        if each_num in ind_car_num_list:
            ind_snippet_number += 1
        else:
            ood_snippet_number += 1
        train1 = torch.load(each_pkl)
        test_X.append(np.array(train1[0])[:, 0:6].reshape(1, 128, 6))
        test_y.append(int(train1[1]['label'][0]))
test_X = np.concatenate(test_X, axis=0)
test_y = np.vstack(test_y)

print(test_X.shape)
print(test_y.shape)

y_test_pred = []
y_test_scores = []
y_test = []
for each_X, each_y in tqdm(zip(test_X, test_y)):
    fit_error = 0
    for each_channel_num in range(each_X.shape[1]):
        each_data = each_X[:, each_channel_num]
        detector = od.Oddity()  # Creating a default Oddity detector
        detector.fit(each_data.reshape(-1, 1))  # Fitting the detector on our data
        mu, cov = detector.mu, detector.cov
        fit_error += sum((mu - each_data) ** 2)
    y_test_scores.append(fit_error)
    y_test.append(each_y)

os.makedirs("gaussian_process", exist_ok=True)
os.makedirs(f"gaussian_process/{args.dataset}", exist_ok=True)


np.save(f"gaussian_process/{args.dataset}/" + f'y_test_pred_fold{args.fold_num}_start{args.start}.npy', y_test_pred)
np.save(f"gaussian_process/{args.dataset}/" + f'y_test_scores_fold{args.fold_num}_start{args.start}.npy', y_test_scores)
np.save(f"gaussian_process/{args.dataset}/" + f'y_test_fold{args.fold_num}_start{args.start}.npy', y_test)

