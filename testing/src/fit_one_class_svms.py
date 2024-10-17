import os
import numpy as np
import argparse
from sklearn.svm import OneClassSVM
import pickle
import time

import dataset_loader


def main(args):
    print()
    debug = bool(int(args.debug))
    K = args.K

    ## Use the features from the DDPM validation set to fit the model
    loader = dataset_loader.load('DDPM', split='val', debug=debug, return_video=False)
    iter_length = len(loader)
    print('Number of samples:', iter_length)

    model_tags = [
        'modrpnet_dbddpm_np_negprob0.00_sigma3.00_nfft5400',
        'modrpnet_dbnegall_specentropy_negprob0.50_sigma3.00_nfft5400',
        'modrpnet_dbnegall_deviation_negprob0.50_sigma3.00_nfft5400',
        'modrpnet_dbnegall_specflatness_negprob0.50_sigma3.00_nfft5400',
        'chrom',
        'pos'
    ]

    for i_tag, model_tag in enumerate(model_tags):
        if model_tag in ['chrom', 'pos']:
            feat_root = os.path.join(loader.root, 'baselines_features')
            feat_dir = os.path.join(feat_root,  model_tag)
            output_root = '../baselines_classifiers'
            output_dir = os.path.join(output_root, model_tag)
        else:
            feat_root = os.path.join(loader.root, 'features')
            feat_dir = os.path.join(feat_root, K, model_tag)
            output_root = '../classifiers'
            output_dir = os.path.join(output_root, K, model_tag)

        print('output root:', output_root)
        os.makedirs(output_root, exist_ok=True)

        print()
        print(f'Reading features from {feat_dir}')

        print('output dir:', output_dir)
        os.makedirs(output_dir, exist_ok=True)

        feat_sets = {}

        for _, label, feat_tag in loader:
            feat_path = os.path.join(feat_dir, feat_tag)
            print(f'Reading from {feat_path}')
            feats = np.load(feat_path)
            for f in feats.files:
                if f.endswith('nonorm'):
                    if f not in feat_sets:
                        feat_sets[f] = []
                    feat_sets[f].append(feats[f])

        ## Flatten to a single feature and many samples
        for f in feat_sets.keys():
            train_X = np.hstack(feat_sets[f]).reshape(-1,1)
            feat_sets[f] = train_X

        num_samples = train_X.shape[0]
        print(f'Number of total samples:', num_samples)
        if debug:
            N = 10000
        else:
            N = 100000
        print(f'Using {N} samples for training.')

        np.random.seed(4) #set same seed for all runs
        sample_idcs = np.random.choice(num_samples, N, replace=False)

        print('feat_set keys:', feat_sets.keys())
        exper_set = [f for f in feat_sets.keys() if f.endswith('nonorm')]
        print('Using exper set:', exper_set)

        exper_tag = 'fusion_nonorm'
        clf_path = os.path.join(output_dir, f'oneclass_{exper_tag}.pkl')
        print('clf path:', clf_path)

        X = [feat_sets[f] for f in exper_set]
        X = np.column_stack(X)
        X = X[sample_idcs]
        print('X:', X.shape)

        start = time.time()
        clf = OneClassSVM(gamma='auto').fit(X)
        total_time = time.time() - start
        print('Took %.2f seconds' % total_time)
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
