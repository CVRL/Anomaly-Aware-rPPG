import sys
sys.path.append('../../datasets')


def load(dataset, debug=False, return_video=True, return_waveform=False, split='test', types='both'):
    if dataset == 'DDPM':
        from DDPM.src.dataloader import Dataset
        return Dataset(split=split, debug=debug, return_video=return_video, return_waveform=return_waveform)
    elif dataset.startswith('NDDPM'):
        from NDDPM.src.dataloader import Dataset
        noise_type = dataset.split('_')[1].lower()
        return Dataset(split=split, noise_type=noise_type, debug=debug, return_video=return_video)
    elif dataset == 'CDDPM':
        from CDDPM.src.dataloader import Dataset
        return Dataset(split=split, debug=debug, return_video=return_video, return_waveform=return_waveform)
    elif dataset == 'DFDC':
        from DFDC.src.dataloader import Dataset
        return Dataset(types=types)
    elif dataset == 'HKBU_MARs_V2':
        from HKBU_MARs_V2.src.dataloader import Dataset
        return Dataset(debug=debug, return_video=return_video, types=types)
    elif dataset == 'KITTI':
        from KITTI.src.dataloader import Dataset
    elif dataset == 'PURE':
        from PURE.src.dataloader import Dataset
        return Dataset(debug=debug, return_video=return_video, return_waveform=return_waveform)
    elif dataset == 'ARPM':
        from ARPM.src.dataloader import Dataset
    elif dataset == 'UBFC_rPPG':
        from UBFC_rPPG.src.dataloader import Dataset
        return Dataset(debug=debug, return_video=return_video, return_waveform=return_waveform)
    else:
        dataset_list = ['DDPM', 'NDDPM_SHUFFLE', 'NDDPM_NORMAL', 'NDDPM_UNIFORM', 'CDDPM', 'DFDC', 'HKBU', 'KITTI', 'PURE', 'ARPM', 'UBFC']
        print('Dataset not found. Must be in:')
        print(dataset_list)
        print('Exiting.')
        sys.exit(-1)

    return Dataset(debug=debug, return_video=return_video)




