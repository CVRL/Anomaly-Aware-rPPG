import numpy as np
import os
import subprocess


def experiment(split):
    model_root = '../best_saved_models'

    model_type = 'rpnet'
    noise_width = 3
    nfft = 5400
    mod_negative_prob = 0.5
    fpc = 270
    step = 135
    skip = 1

    mod_db = 'negshuffle'
    #mod_db = 'neguniform'
    #mod_db = 'negnormal'

    loss_type = 'specentropy'
    #loss_type = 'specflatness'
    #loss_type = 'deviation'

    model_path = f'{model_root}/mod{model_type}_db{mod_db}_{loss_type}_negprob{mod_negative_prob:.2f}_sigma{noise_width:.2f}_nfft{nfft}.pth'

    negative_prob = 1
    #val_db = 'negnormal'
    #val_db = 'neguniform'
    val_db = 'negshuffle'

    #negative_prob = 0
    #val_db = 'ddpm'

    comm = "sh ../submitters/validate_emitted.sh %s %s %d %s %s %f %f %d %d %d" % (model_path, model_type, split, val_db, loss_type, negative_prob, noise_width, fpc, step, skip)
    print(comm)
    subprocess.call(comm.split(' '))
    print('')



def main():
    experiment(2)


if __name__ == '__main__':
    main()

