import os
import numpy as np
import argparse
import dataset_loader
from scipy.stats import pearsonr
from extract_features import estimate_hr_framestride


def main(args):
    ## Constant parameters
    width = 10
    fps = 90
    min_hz = 2.0 / 3.0
    max_hz = 4.0
    nfft = 5400
    maf_width = 271 #For hilbert transform
    normed = False

    print()
    print(f'Using normed={normed}')
    debug = bool(int(args.debug))

    all_datasets = ["DDPM", "PURE", "UBFC_rPPG"]

    rpnet_tags = [
        'modrpnet_dbddpm_np_negprob0.00_sigma3.00_nfft5400',
        'modrpnet_dbnegall_specentropy_negprob0.50_sigma3.00_nfft5400',
        'modrpnet_dbnegall_deviation_negprob0.50_sigma3.00_nfft5400',
        'modrpnet_dbnegall_specflatness_negprob0.50_sigma3.00_nfft5400',
        'chrom',
        'pos'
    ]

    print('rpnet_tags:', rpnet_tags)

    for rpnet_tag in rpnet_tags:

        print()
        all_gt_HRs = []
        all_pred_HRs = []

        for testing_dataset in all_datasets:

            print('Testing on dataset:', testing_dataset)
            loader = dataset_loader.load(testing_dataset, debug=debug, return_waveform=True)

            pred_root = os.path.join(loader.root, 'output_data')

            ## Iterate through samples in a dataset and store all waveform preds
            pred_HRs = []
            gt_HRs = []
            pred_HRs = []
            for _, gt_wave, pred_tag in loader:
                gt_HR = estimate_hr_framestride(gt_wave, fps=fps, width=width, stride=1, min_hz=min_hz, max_hz=max_hz, nfft=nfft)
                pred_HRs_K = []
                gt_HRs_K = []
                for K in range(1,6):
                    pred_path = os.path.join(pred_root, str(K), rpnet_tag, pred_tag)

                    if not os.path.exists(pred_path):
                        break

                    if normed:
                        pred_wave = np.load(pred_path)['pred_normed']
                    else:
                        pred_wave = np.load(pred_path)['pred_nonorm']
                    pred_HR = estimate_hr_framestride(pred_wave, fps=fps, width=width, stride=1, min_hz=min_hz, max_hz=max_hz, nfft=nfft)

                    gt_HR_K = gt_HR.copy()
                    if len(gt_HR_K) < len(pred_HR):
                        pred_HR = pred_HR[:len(gt_HR_K)]
                    if len(pred_HR) < len(gt_HR_K):
                        gt_HR_K = gt_HR_K[:len(pred_HR)]
                    gt_HRs_K.append(gt_HR_K)
                    pred_HRs_K.append(pred_HR)

                ## Should be [S,K,T]
                gt_HRs.append(gt_HRs_K)
                pred_HRs.append(pred_HRs_K)

            ME_HR_K = []
            MAE_HR_K = []
            RMSE_HR_K = []
            r_HR_K = []
            for i in range(5):
                flat_gt_HRs_K = []
                flat_pred_HRs_K = []
                for S in range(len(gt_HRs)):
                    S_gt = gt_HRs[S][i]
                    S_pred = pred_HRs[S][i]
                    flat_gt_HRs_K.append(S_gt)
                    flat_pred_HRs_K.append(S_pred)
                ME_HR, MAE_HR, RMSE_HR, r_HR = get_errors(flat_gt_HRs_K, flat_pred_HRs_K)
                ME_HR_K.append(ME_HR)
                MAE_HR_K.append(MAE_HR)
                RMSE_HR_K.append(RMSE_HR)
                r_HR_K.append(r_HR)
                print()
                print(f'K={i}')
                print(testing_dataset)
                print(rpnet_tag)
                print('ME_HR, MAE_HR, RMSE_HR, r_HR')
                print(round(ME_HR, 2), '&', round(MAE_HR, 2), '&', round(RMSE_HR, 2), '&', round(r_HR, 2))

            print()
            print(f'All Ks')
            print(testing_dataset)
            print(rpnet_tag)
            print('\t\t ME_HR, MAE_HR, RMSE_HR, r_HR')
            print('mean', round(np.mean(ME_HR_K), 2), '&', round(np.mean(MAE_HR_K), 2), '&', round(np.mean(RMSE_HR_K), 2), '&', round(np.mean(r_HR_K), 2))
            print('std',  round(np.std(ME_HR_K), 2),  '&', round(np.std(MAE_HR_K), 2),  '&', round(np.std(RMSE_HR_K), 2),  '&', round(np.std(r_HR_K), 2))

            all_gt_HRs.append(gt_HRs)
            all_pred_HRs.append(pred_HRs)

        ## all_gt_HRs if of shape [N_datasets, N_subjects, K, T]
        ## Need to flatten to [K, T]
        ## Model errors over all pulse datasets:
        ME_HR_K = []
        MAE_HR_K = []
        RMSE_HR_K = []
        r_HR_K = []
        for i in range(5):
            flat_gt_HRs_K = []
            flat_pred_HRs_K = []
            for N in range(len(all_gt_HRs)):
                N_gt = all_gt_HRs[N]
                N_pred = all_pred_HRs[N]
                for S in range(len(N_gt)):
                    S_gt = N_gt[S][i]
                    S_pred = N_pred[S][i]
                    flat_gt_HRs_K.append(S_gt)
                    flat_pred_HRs_K.append(S_pred)

            flat_gt_HRs_K = np.hstack(flat_gt_HRs_K)
            flat_pred_HRs_K = np.hstack(flat_pred_HRs_K)
            ME_HR, MAE_HR, RMSE_HR, r_HR = get_errors(flat_gt_HRs_K, flat_pred_HRs_K)
            ME_HR_K.append(ME_HR)
            MAE_HR_K.append(MAE_HR)
            RMSE_HR_K.append(RMSE_HR)
            r_HR_K.append(r_HR)
        print()
        print('All datasets')
        print(rpnet_tag)
        print('\t\t ME_HR, MAE_HR, RMSE_HR, r_HR')
        print('mean', round(np.mean(ME_HR_K), 2), '&', round(np.mean(MAE_HR_K), 2), '&', round(np.mean(RMSE_HR_K), 2), '&', round(np.mean(r_HR_K), 2))
        print('std',  round(np.std(ME_HR_K), 2),  '&', round(np.std(MAE_HR_K), 2),  '&', round(np.std(RMSE_HR_K), 2),  '&', round(np.std(r_HR_K), 2))
        print()

    print(20*'-')
    return


def get_errors(gt_HRs, pred_HRs):
    gt_HRs = np.hstack(gt_HRs.copy())
    pred_HRs = np.hstack(pred_HRs.copy())
    ME_HR = np.mean(gt_HRs - pred_HRs)
    MAE_HR = np.mean(np.abs(gt_HRs - pred_HRs))
    RMSE_HR = np.sqrt(np.mean(np.square(gt_HRs - pred_HRs)))
    r_HR, p_HR = pearsonr(gt_HRs, pred_HRs)
    return ME_HR, MAE_HR, RMSE_HR, r_HR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug',
                        default=False,
                        help='Are we in debugging mode to decrease dataset size?')
    args = parser.parse_args()
    main(args)
