import os
import numpy as np
import argparse
import pickle
import dataset_loader


def main(args):
    print()
    task_id = int(args.task_id) - 1
    task_count = int(args.task_count)
    testing_dataset = args.testing_dataset
    debug = bool(int(args.debug))
    K = args.K

    print('Testing on dataset:', testing_dataset)
    loader = dataset_loader.load(testing_dataset, debug=debug, return_video=False)
    iter_length = len(loader)
    print('Number of samples:', iter_length)
    print('Loader root:', loader.root)

    model_tags = [
        'modrpnet_dbddpm_np_negprob0.00_sigma3.00_nfft5400',
        'modrpnet_dbnegall_specentropy_negprob0.50_sigma3.00_nfft5400',
        'modrpnet_dbnegall_deviation_negprob0.50_sigma3.00_nfft5400',
        'modrpnet_dbnegall_specflatness_negprob0.50_sigma3.00_nfft5400',
        'chrom',
        'pos'
    ]

    print('model_tags:', model_tags)

    for model_tag in model_tags:
        if model_tag in ['pos', 'chrom']:
            feat_dir = os.path.join(loader.root, 'baselines_features', model_tag)
            svm_root = '../baselines_classifiers'
            svm_dir = os.path.join(svm_root, model_tag)
            output_dir = os.path.join(loader.root, 'baseline_svm_scores', model_tag)
        else:
            feat_dir = os.path.join(loader.root, 'features', K, model_tag)
            svm_root = '../classifiers'
            svm_dir = os.path.join(svm_root, K, model_tag)
            output_dir = os.path.join(loader.root, 'svm_scores', K, model_tag)

        print(f'Loading svm from directory {svm_dir}')
        if not os.path.isdir(svm_dir):
            print(f'SVM model not trained for model {model_tag}')
            continue

        print()
        print(f'Reading features from directory {feat_dir}')

        os.makedirs(output_dir, exist_ok=True)
        print(f'Saving output data to directory {output_dir}')

        clfs = load_svms(svm_dir)
        clf_keys = [clf for clf in clfs if 'fusion' in clf]
        print('clfs:')
        print([k for k in clfs])

        i = 0
        for _, label, feat_tag in loader:
            if (i % task_count) != task_id:
                i += 1
                continue

            feat_path = os.path.join(feat_dir, feat_tag)

            output_head, output_tail = os.path.split(feat_tag)
            output_head = os.path.join(output_dir, output_head)
            os.makedirs(output_head, exist_ok=True)
            output_path = os.path.join(output_head, output_tail)

            print(f'Reading from {feat_path}')
            print(f'Writing to {output_path}')
            assert(feat_path != output_path)

            feats = np.load(feat_path)

            predictions = {}

            for clf_key in clf_keys:
                X = [feats[fk] for fk in feats.files if 'nonorm' in fk]
                X = np.column_stack(X)
                print('X:', X.shape)
                if X.shape[0] > 0:
                    prediction = clfs[clf_key].decision_function(X)
                    predictions[clf_key] = prediction
                    print('prediction:', prediction.shape)

            np.savez_compressed(output_path, **predictions)
            i += 1

    print(f'Finished after {i} samples')
    print(20*'-')
    return


def load_svms(svm_dir):
    clfs = {}
    svm_paths = sorted(os.listdir(svm_dir))
    for svm_path in svm_paths:
        with open(os.path.join(svm_dir, svm_path), 'rb') as infile:
            clf = pickle.load(infile)
            clfs[svm_path] = clf
    return clfs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--K',
                        help='Which model initialization to use [1-5].')
    parser.add_argument('--task_id',
                        help='ID for the task array index.')
    parser.add_argument('--task_count',
                        help='Number of tasks in array.')
    parser.add_argument('--testing_dataset',
                        help='Dataset to process.')
    parser.add_argument('--debug',
                        default=False,
                        help='Are we in debugging mode to decrease dataset size?')
    args = parser.parse_args()
    main(args)
