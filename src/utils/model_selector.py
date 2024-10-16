import sys
import os
import torch


def select_model(arg_obj, device):
    model_type = arg_obj.model_type.lower()
    tk = int(arg_obj.tk)
    channels = arg_obj.channels
    dropout = float(arg_obj.dropout)
    dtype = arg_obj.dtype
    num_channels = len(channels)

    if model_type == 'rpnet':
        from models.RPNet import RPNet
        model = RPNet(input_channels=num_channels, drop_p=dropout, t_kern=tk)
    else:
        print('Could not find model specified.')
        sys.exit(-1)

    if dtype[0] == 'f':
        model = model.float()
    else:
        model = model.double()

    return model


def load_best_model(args, best_save_root, device):
    model = select_model(args, device)
    tag = get_output_tag(args.model_type,
            args.dataset,
            args.loss_type,
            float(args.negative_prob),
            float(args.noise_width),
            int(args.nfft),
            epoch=None)
    checkpoint_path = os.path.join(best_save_root, tag+'.pth')
    checkpoint = torch.load(checkpoint_path)
    print('Using save path:', checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    return model, tag


def get_best_checkpoint(best_save_root, model_type, dataset, loss_type, negative_prob, noise_width, nfft):
    tag = get_output_tag(model_type, dataset, loss_type, negative_prob, noise_width, nfft, epoch=None)
    checkpoint_path = os.path.join(best_save_root, tag+'.pth')
    checkpoint = torch.load(checkpoint_path)
    return checkpoint


def get_best_loss(best_save_root, model_type, dataset, loss_type, negative_prob, noise_width, nfft):
    checkpoint = get_best_checkpoint(best_save_root, model_type, dataset, loss_type, negative_prob, noise_width, nfft)
    loss = checkpoint['loss']
    return loss


def get_output_tag(model_type, dataset, loss_type, negative_prob, noise_width, nfft, epoch=None):
    if epoch is None:
        tag = 'mod%s_db%s_%s_negprob%.2f_sigma%.2f_nfft%d' % (model_type, dataset, loss_type, negative_prob, noise_width, nfft)
    else:
        tag = 'mod%s_db%s_%s_negprob%.2f_sigma%.2f_nfft%d_e%d' % (model_type, dataset, loss_type, negative_prob, noise_width, nfft, epoch)
    return tag


def get_last_checkpoint(save_root, end_epoch, model_type, dataset, loss_type, negative_prob, noise_width, nfft):
    checkpoint = None
    last_epoch = -1
    for epoch in range(end_epoch):
        tag = get_output_tag(model_type, dataset, loss_type, negative_prob, noise_width, nfft, epoch=epoch)
        model_path = os.path.join(save_root, tag)
        if os.path.exists(model_path):
            checkpoint = model_path
            last_epoch = epoch
    return checkpoint, last_epoch
