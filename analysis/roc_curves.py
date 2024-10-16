import numpy as np
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.svm import LinearSVC
import pickle
import os

np.random.seed(101)


def main():

    width = 30
    model_bases = ['modrpnet_dbmspec_npentropy', 'modrpnet_dbnegnormal_specentropy', 'modrpnet_dbnegshuffle_specentropy',
            'modrpnet_dbnegshuffle_specflatness', 'modrpnet_dbneguniform_specflatness', 'RemotePulseNet']

    outdir = os.path.join(f'pickles/pickles_{width}')
    if not os.path.isdir(outdir):
        os.makedirs(outdir)

    indir = os.path.join(f'snrs/snrs_{width}')

    for i, model_base in enumerate(model_bases):
        labels = []
        snrs = []
        d = {}
        files = [f for f in sorted(os.listdir(indir)) if f.startswith(model_base)]
        for file in files:
            infile = os.path.join(indir, file)
            print(infile, os.path.exists(infile))
            snr = np.load(infile, allow_pickle=True)
            snr = np.hstack(snr)
            snr = -snr # lower snrs should give higher prob of negative sample
            if 'evalmspec' in file:
                label = np.zeros_like(snr)
            else:
                label = np.ones_like(snr)
            labels.append(label)
            snrs.append(snr)
            print(snr)

        labels = np.hstack(labels)
        snrs = np.hstack(snrs)
        if i == 0:
            N = len(labels)
            pivot = N // 100
            all_idcs = np.arange(N)
            np.random.shuffle(all_idcs)
            train_idcs = all_idcs[:pivot]
            test_idcs = all_idcs[pivot:]

        clf = LinearSVC()
        train_X, train_Y = snrs[train_idcs].reshape(-1, 1), labels[train_idcs]
        clf.fit(train_X, train_Y)
        test_X = snrs[test_idcs].reshape(-1, 1)
        test_preds = clf.predict(test_X)
        acc = accuracy_score(labels[test_idcs], test_preds)

        fpr, tpr, thresholds = roc_curve(labels, snrs)
        roc_auc = auc(fpr, tpr)
        d['fpr'] = fpr
        d['tpr'] = tpr
        d['thresholds'] = thresholds
        d['auc'] = roc_auc
        d['accuracy'] = acc

        outpath = os.path.join(outdir, model_base+'.pickle')
        with open(outpath, 'wb') as outfile:
            pickle.dump(d, outfile)


if __name__ == '__main__':
    main()
