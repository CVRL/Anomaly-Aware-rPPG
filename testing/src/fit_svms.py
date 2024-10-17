import os
import torch
import numpy as np
import argparse
import scipy.signal as signal
from sklearn.svm import OneClassSVM
from sklearn.svm import SVC
import pickle
import time

import dataset_loader


def main(args):
    print()
    debug = bool(int(args.debug))
    K = args.K

    ## Use the features from the DDPM validation set to fit the model
    pos_loader = dataset_loader.load('DDPM', split='val', debug=debug, return_video=False)

    model_tags = [
        'modrpnet_dbddpm_np_negprob0.00_sigma3.00_nfft5400',
        'modrpnet_dbnegall_specentropy_negprob0.50_sigma3.00_nfft5400',
        'modrpnet_dbnegall_deviation_negprob0.50_sigma3.00_nfft5400',
        'modrpnet_dbnegall_specflatness_negprob0.50_sigma3.00_nfft5400',
        'chrom',
        'pos'
    ]
    print('model_tags:', model_tags)
    print('Loader root:', pos_loader.root)

    for i_tag, model_tag in enumerate(model_tags):
        if model_tag in ['chrom', 'pos']:
            output_root = '../baselines_classifiers'
            output_dir = os.path.join(output_root, model_tag)
            pos_feat_root = os.path.join(pos_loader.root, 'baselines_features')
            pos_feat_dir = os.path.join(pos_feat_root, model_tag)
        else:
            output_root = '../classifiers'
            output_dir = os.path.join(output_root, K, model_tag)
            pos_feat_root = os.path.join(pos_loader.root, 'features')
            pos_feat_dir = os.path.join(pos_feat_root, K, model_tag)

        print('output dir:', output_dir)
        os.makedirs(output_dir, exist_ok=True)

        pos_feat_sets = {}
        neg_feat_sets = {}

        ## Iterate through the positive dataset loader
        for _, label, feat_tag in pos_loader:
            feat_path = os.path.join(pos_feat_dir, feat_tag)
            feats = np.load(feat_path)
            for f in feats.files:
                if f.endswith('nonorm'):
                    if f not in pos_feat_sets:
                        pos_feat_sets[f] = []
                    pos_feat_sets[f].append(feats[f])

        ## Iterate through the different types of negative samples
        for negative_dataset in ['NDDPM_SHUFFLE', 'NDDPM_UNIFORM', 'NDDPM_NORMAL']:
            neg_loader = dataset_loader.load(negative_dataset, split='val', debug=debug, return_video=False)

            if model_tag in ['chrom', 'pos']:
                neg_feat_dir = os.path.join(neg_loader.root, 'baselines_features', model_tag)
            else:
                neg_feat_dir = os.path.join(neg_loader.root, 'features', K, model_tag)

            ## Iterate through the negative dataset loaders
            for _, label, feat_tag in neg_loader:
                feat_path = os.path.join(neg_feat_dir, feat_tag)
                feats = np.load(feat_path)
                for f in feats.files:
                    if f.endswith('nonorm'):
                        if f not in neg_feat_sets:
                            neg_feat_sets[f] = []
                        neg_feat_sets[f].append(feats[f])

        ## Flatten to a single feature and many samples
        for f in pos_feat_sets.keys():
            pos_train_X = np.hstack(pos_feat_sets[f]).reshape(-1,1)
            pos_num_samples = pos_train_X.shape[0]
            pos_feat_sets[f] = pos_train_X
            neg_train_X = np.hstack(neg_feat_sets[f]).reshape(-1,1)
            neg_num_samples = neg_train_X.shape[0]
            neg_feat_sets[f] = neg_train_X

        if debug:
            N = 1000 #debuggin
        else:
            N = 100000

        np.random.seed(4) #set same seed for all runs
        sample_idcs = np.random.choice(pos_num_samples, N, replace=False)

        exper_set = [f for f in pos_feat_sets.keys() if f.endswith('nonorm')]
        exper_tag = 'fusion_nonorm'

        X_pos = np.column_stack([pos_feat_sets[k] for k in exper_set])
        X_neg = np.column_stack([neg_feat_sets[k] for k in exper_set])
        X_pos = X_pos[sample_idcs[:N//2]]
        X_neg = X_neg[sample_idcs[N//2:]]
        print('X_pos:', X_pos.shape)
        print('X_neg:', X_neg.shape)
        X = np.vstack((X_pos, X_neg))
        Y = np.ones(N)
        Y[N//2:] = 0

        start = time.time()
        clf = SVC(kernel='rbf').fit(X, Y)
        total_time = time.time() - start
        print('Took %.2f seconds for RBF' % total_time)
        clf_path = os.path.join(output_dir, f'rbf_{exper_tag}.pkl')
        print('clf path:', clf_path)
        with open(clf_path,'wb') as f:
            pickle.dump(clf,f)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--K',
                        help='Which model initialization to use [1-5].')
    parser.add_argument('--debug',
                        default=False,
                        help='Are we in debugging mode to decrease dataset size?')
    args = parser.parse_args()
    main(args)
