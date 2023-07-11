This is the official code repository for our Nature Communication paper entitled
"Realistic fault detection of Li-ion battery via dynamical deep learning approach"

# Environment requirement
We recommend using the conda environment.
```
# basic environment
CUDA 10.2  # Must use this specific version. Please follow https://developer.nvidia.com/cuda-10.2-download-archive. 
python 3.6

# pytorch version
pytorch==1.5.1

# run after installing correct Pytorch package
pip install --no-index torch-scatter -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install --no-index torch-sparse -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install --no-index torch-cluster -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install --no-index torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.5.0+cu102.html
pip install torch-geometric==1.5.0
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirement.txt
```
# Dataset preparation
## Download
Download from the link in our paper and unzip them. 
Please make sure the data structure is like the following. 


```
    |--data
        |--battery_brand1
            |--label
            |--train
            |--test
            |column.pkl
        |--battery_brand2
            |--...
        |--battery_brand3
            |--...
    
```


## File content

Each `pkl` file is a tuple including two parts. The first part is charging time series
data. `column.pkl` contains column names for the time sequence data. 
The second part is meta data which contains fault label, car number, charge segment number
and mileage. 

The `label` folder contains car numbers and their anomaly labels.

## Generate path information for five-fold validation

In our paper, we provide experiments of training with different brands.
To facilitate the organization of training and test data, we use 1) a python dict to save 
`car number-snippet paths` information, which is named as `all_car_dict.npz.npy`, and 2) a dict to save the
randomly shuffled normal and abnormal car number to perform five-fold training and testing, which is 
named as `ind_odd_dict*.npz.npy`. By default, the code is running on the first brand. So our code
is now running on `ind_odd_dict1.npz.npy`. 

The details of **running experiments on other brands** requires modification on some code, which is 
illustrated in this readme file later. 

To build the `all_car_dict.npz.npy` and `ind_odd_dict*.npz.npy`, run

`cd data`

Run `five_fold_train_test_split.ipynb` and then you get all the files saved in 
`nature code\five_fold_utils\`.
(Running each cell of the `five_fold_train_test_split.ipynb` may take 
a few minutes. If not, please check the data path carefully.)

The cell output of each cell contains randomly shuffled `ind_car_num_list` 
and `ood_car_num_list`. You may print it out to see the car numbers you are using. 


# Run DyAD(ours)

**Setting another brand:** By default, we are using brand 1. To run experiments on other brands, 
one should manually change the variable
`ind_ood_car_dict_path` in `DyAD/model/dataset.py`. For example, if you want to use brand 2, then you
should use `ind_ood_car_dict_path='../five_fold_utils/ind_odd_dict2.npz.npy',`. 
(An easy way to do so is to use Ctrl+F to search the name of the variables.)

## train
Please check the `model_params_battery_brand*.json` files carefully for hyperparameter settings. 
`model_params_battery_brand1/2/3.json` are used to train vehicles of brand1/2/3 separately.
And use `fold_num` to do the five-fold training and testing. To start training, run
```
cd DyAD
python main_five_fold.py --config_path model_params_battery_brand1.json --fold_num 0
```
If you want to fully run the five-fold experiments, you should run five times with different 
`--fold_num`.
After training, the reconstruction errors of data are recorded  in `save_model_path` configured by the
`params_fivefold.json` file.

# AutoEncoder & SVDD

**Setting another brand:** By default, we are using brand 1. To run experiments on other brands, 
one should manually change the variable
`ind_ood_car_dict` in `AE_and_SVDD/traditional_methods.py` similarly as above. 

## train
To start training, run
```
cd AE_and_SVDD
python traditional_methods.py --method auto_encoder --normalize --fold_num 0
python traditional_methods.py --method deepsvdd --normalize --fold_num 0
```
If you want to fully run the five-fold experiments, you should run five times with different 
`--fold_num`.


# LSTM-AD

**Setting another brand:** By default, we are using brand 1. To run experiments on other brands, 
one should manually change the variable
`ind_ood_car_dict` in `Recurrent-Autoencoder-modify/agents/rnn_autoencoder.py`
and `Recurrent-Autoencoder-modify/datasets/battery.py` similarly as above. 

## train
To start training, run
```
cd Recurrent-Autoencoder-modify
python main.py configs/config_lstm_ae_battery_0.json
```
where `...0.json` means the first fold in five-fold validation. 

If you want to fully run the five-fold experiments, you should run five times with different 
config files (`config_lstm_ae_battery_*.json` where `*` can be 1/2/3/4).

# GDN

**Setting another brand:** By default, we are using brand 1. To run experiments on other brands, 
one should manually change the variable
`ind_ood_car_dict` in `GDN_battery/datasets/TimeDataset.py` and `GDN_battery/main.py` 
similarly as above. 

## train
```
cd GDN_battery
bash run_battery.sh 3 battery 1 0 20
```
where `3` is the gpu number, `battery` is the dataset name, 
`1` is the fold number (also can be 0/2/3/4) `0` means use all data to train and `20` is epoch number.
For details, please see the `.sh` file. 

# Calculated ROC score
For all the mentioned algorithms, we calculated the AUROC values with 
jupyter-notebooks in `notebooks`. For each notbook file, the suffix `threshold` 
represents the calculation of robust fault scores, and the suffix 
`threshold_no` represents the calculation of average fault scores. 

**Necessary modification:** Since the save path may be time dependent and machine dependent, one needs
to change the path information in each jupyter notebook.
One should also modify the path of the saved reconstruction error
if one is using different brands. 

# Data availability
The datasets are available at links below
https://1drv.ms/u/s!AiSrJIRVqlQAgcjKGKV0fZmw5ifDd8Y?e=CnzELH237
or
https://disk.pku.edu.cn:443/link/37D733DF405D8D7998B8F57E4487515A238.

# Code Reference
We use partial code from 
```
https://github.com/yzhao062/pyod
https://github.com/d-ailin/GDN
https://github.com/PyLink88/Recurrent-Autoencoder
``` 
