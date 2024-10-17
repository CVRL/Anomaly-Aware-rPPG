import os
import numpy as np
import pandas as pd
import gzip
import pickle
import argparse
import dataset_loader


def main(args):
    print()
    debug = bool(int(args.debug))
    all_datasets = ["DDPM", "CDDPM", "DFDC", "HKBU_MARs_V2", "KITTI", "PURE", "ARPM", "UBFC_rPPG"]
    model_score_dict = {}
    rpnet_tags = [
        'modrpnet_dbddpm_np_negprob0.00_sigma3.00_nfft5400',
        'modrpnet_dbnegall_specentropy_negprob0.50_sigma3.00_nfft5400',
        'modrpnet_dbnegall_deviation_negprob0.50_sigma3.00_nfft5400',
        'modrpnet_dbnegall_specflatness_negprob0.50_sigma3.00_nfft5400',
        'chrom',
        'pos'
    ]

    svm_tags = [
        'oneclass_fusion_nonorm.pkl',
        'rbf_fusion_nonorm.pkl'
    ]

    for testing_dataset in all_datasets:

        if not testing_dataset in model_score_dict:
            model_score_dict[testing_dataset] = {}

        print('Testing on dataset:', testing_dataset)
        loader = dataset_loader.load(testing_dataset, debug=debug, return_video=False, split='test', types='both')
        iter_length = len(loader)
        print('Number of samples:', iter_length)

        print('Loader root:', loader.root)
        output_root = os.path.join(loader.root, 'analysis')
        if not os.path.exists(output_root):
            os.makedirs(output_root)

        print('rpnet_tags:', rpnet_tags)
        for rpnet_tag in rpnet_tags:
            if rpnet_tag in ['chrom', 'pos']:
                score_root = os.path.join(loader.root, 'baseline_svm_scores')
                k_range = [1]
            else:
                score_root = os.path.join(loader.root, 'svm_scores')
                k_range = range(1,6)

            score_dict = {}
            label_dict = {}

            if rpnet_tag not in model_score_dict[testing_dataset]:
                model_score_dict[testing_dataset][rpnet_tag] = {}

            for K in k_range:
                print()

                score_dict[K] = {}
                label_dict[K] = {}
                if not K in model_score_dict[testing_dataset][rpnet_tag]:
                    model_score_dict[testing_dataset][rpnet_tag][K] = {}

                if rpnet_tag in ['chrom', 'pos']:
                    score_dir = os.path.join(score_root, rpnet_tag)
                    output_path = os.path.join(output_root, rpnet_tag+'.csv')
                else:
                    score_dir = os.path.join(score_root, str(K), rpnet_tag)
                    output_path = os.path.join(output_root, rpnet_tag+f'_{K}.csv')
                print(f'Writing results to {output_path}')

                if not os.path.isdir(score_dir):
                    print(f'SVM scores not produced for model {rpnet_tag}')
                    continue
                print(f'Reading scores from {score_dir}')

                ## Iterate through samples in a dataset and store all scores from SVM
                for _, label, score_tag in loader:
                    score_path = os.path.join(score_dir, score_tag)
                    if not os.path.exists(score_path):
                        continue
                    scores = np.load(score_path, allow_pickle=True)
                    if len(scores.files) < 1:
                        continue
                    N_frames = len(scores[svm_tags[0]])
                    label_fbf = np.repeat(label, N_frames)
                    for s in svm_tags:
                        if not s in score_dict[K]:
                            score_dict[K][s] = []
                        if not s in label_dict[K]:
                            label_dict[K][s] = []
                        score_dict[K][s].append(scores[s])
                        label_dict[K][s].append(label_fbf)

                ## Evaluate predictions for entire dataset
                for s in svm_tags:
                    scores_k = np.hstack(score_dict[K][s])
                    labels = np.hstack(label_dict[K][s])
                    preds_k = (scores_k > 0).astype(int)
                    N = len(labels)
                    TP = np.sum((labels == 1) & (preds_k == 1))
                    TN = np.sum((labels == 0) & (preds_k == 0))
                    accuracy = (TP + TN) / N

                    df = {
                        'N': N,
                        'TP': TP,
                        'TN': TN,
                        'accuracy': accuracy,
                    }
                    model_score_dict[testing_dataset][rpnet_tag][K][s] = df

    with open('../auxiliary/score_dict.pkl', 'wb') as outfile:
        pickle.dump(model_score_dict, outfile)

    svm_tags = [
        'oneclass_fusion_nonorm.pkl',
        'rbf_fusion_nonorm.pkl'
    ]

    ## Calculate scores for a model over all datasets
    for model_tag in model_score_dict[all_datasets[0]]:
        ## Iterate through svms
        for s in svm_tags:
            ## Iterate through datasets
            for testing_dataset in all_datasets:
                svm_accs = []
                if model_tag in ['chrom','pos']:
                    k_range = [1]
                else:
                    k_range = range(1, 6)
                ## Iterate through Ks
                for K in k_range:
                    svm_scores = model_score_dict[testing_dataset][model_tag][K][s]
                    acc = 100 * ((svm_scores['TP'] + svm_scores['TN']) / svm_scores['N'])
                    svm_accs.append(acc)

                mean_acc = np.mean(svm_accs)
                std_acc = np.std(svm_accs)
                print(f'{testing_dataset} {model_tag} {s}')
                print(f'{round(mean_acc,2):.2f} $\pm$ {round(std_acc,2):.2f}')
                print()

    ## Calculate errors for model over all datasets
    for rpnet_tag in rpnet_tags:
        for svm_tag in svm_tags:
            svm_df = {}
            accs = []
            if rpnet_tag in ['chrom','pos']:
                k_range = [1]
            else:
                k_range = range(1, 6)
            for K in k_range:
                svm_df[K] = {}
                svm_df[K]['N'] = 0
                svm_df[K]['TN'] = 0
                svm_df[K]['TP'] = 0
                for testing_dataset in all_datasets:
                    svm_df[K]['N'] += model_score_dict[testing_dataset][rpnet_tag][K][svm_tag]['N']
                    svm_df[K]['TP'] += model_score_dict[testing_dataset][rpnet_tag][K][svm_tag]['TP']
                    svm_df[K]['TN'] += model_score_dict[testing_dataset][rpnet_tag][K][svm_tag]['TN']
                acc = 100*((svm_df[K]['TP'] + svm_df[K]['TN']) / svm_df[K]['N'])
                accs.append(acc)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            print(f'All {rpnet_tag} {svm_tag}')
            print(f'{round(mean_acc,2):.2f} $\pm$ {round(std_acc,2):.2f}')
            print()

    print(20*'-')
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        default=False,
                        help='Are we in debugging mode to decrease dataset size?')
    args = parser.parse_args()
    main(args)
