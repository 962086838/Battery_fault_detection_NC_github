import argparse
import json
import os
import sys
import train
import extract
import evaluate

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch battery Example')
    parser.add_argument('--config_path', type=str,
                        default='./params.json')
    parser.add_argument('--fold_num', type=int, default=0)

    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        p_args = argparse.Namespace()
        p_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=p_args)
    print("Loaded configs at %s" % args.config_path)
    print("args", args)


    # train and save feature
    tr = train.Train_fivefold(args, fold_num=args.fold_num)
    print('train start............................')
    tr.main()
    print('train end............................')
    modelparams_path=tr.getmodelparams()
    del tr
    parser.add_argument('--modelparams_path', type=str,
                        default=modelparams_path)
    args = parser.parse_args()
    with open(args.modelparams_path, 'r') as file:
        p_args = argparse.Namespace()
        model_params=json.load(file)
        p_args.__dict__.update(model_params["args"])
        args = parser.parse_args(namespace=p_args)
    # feature extraction
    ext = extract.Extraction(args, fold_num=args.fold_num)
    print('feature extraction start............................')
    ext.main()
    print('feature extraction end............................')
    del ext

    # detection
    ev = evaluate.Evaluate(args)
    print('anomaly detection start............................')
    ev.main()
    print('anomaly detection end............................')
    del ev
