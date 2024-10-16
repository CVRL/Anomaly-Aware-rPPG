import sys

def get_dataset(split, arg_obj):
    dataset = arg_obj.dataset.lower()
    if dataset == 'ddpm':
        from datasets.DDPM import DDPM as DataSet
    elif dataset == 'negshuffle':
        from datasets.DDPM_negative_shuffle import DDPMNegativeShuffle as DataSet
    elif dataset == 'negnormal':
        from datasets.DDPM_negative_normal import DDPMNegativeNormal as DataSet
    elif dataset == 'neguniform':
        from datasets.DDPM_negative_uniform import DDPMNegativeUniform as DataSet
    elif dataset == 'negall':
        from datasets.DDPM_negative_all import DDPMNegativeAll as DataSet
    else:
        print('Dataset not found. Exiting.')
        sys.exit(-1)

    return DataSet(split, arg_obj)
