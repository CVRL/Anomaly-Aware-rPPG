import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datasets.utils import get_dataset
from losses import select_loss
import utils.model_selector as model_utils
import numpy as np
import args
import json
import time
import os
import sys
from copy import deepcopy
import validate as validate_script
from tqdm import tqdm

np.random.seed(1)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    start = time.time()
    arg_obj = args.get_input()
    load_path = arg_obj.load_path
    continue_training = bool(int(arg_obj.continue_training))
    dataset = arg_obj.dataset.lower()
    model_type = arg_obj.model_type.lower()
    loss_type = arg_obj.loss.lower()
    channels = arg_obj.channels
    dropout = float(arg_obj.dropout)
    batch_size = int(arg_obj.batch_size)
    fpc = int(arg_obj.fpc)
    step = int(arg_obj.step)
    skip = int(arg_obj.skip)
    dtype = arg_obj.dtype
    lr = float(arg_obj.lr)
    start_epoch = int(arg_obj.start_epoch)
    end_epoch = int(arg_obj.end_epoch)
    shuffle = bool(int(arg_obj.shuffle))
    aug = arg_obj.augmentation.lower()
    num_workers = int(arg_obj.num_workers)
    use_hanning = True
    negative_prob = float(arg_obj.negative_prob)
    noise_width = float(arg_obj.noise_width)
    nfft = int(arg_obj.nfft)
    K = arg_obj.K

    model_seed, train_seed = load_seeds(arg_obj.rerun_seeds, K)
    print('model, training seeds:', model_seed, train_seed)
    print('K:', K)

    save_root = f'../saved_models/{K}'
    best_save_root = f'../best_saved_models/{K}'
    output_dir = f'../outputs/{K}'

    os.makedirs(save_root, exist_ok=True)
    os.makedirs(best_save_root, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(model_seed)
    hrcnn = model_utils.select_model(arg_obj, device)
    hrcnn = hrcnn.to(device)

    torch.manual_seed(train_seed)
    train_set = get_dataset('train', arg_obj)
    val_set = get_dataset('val', arg_obj)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=shuffle, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=1, shuffle=False,
                                             num_workers=num_workers)
    print('Train len: ', len(train_loader))
    print('Val len: ', len(val_loader))

    criterion = select_loss(loss_type)
    optimizer = optim.Adam(hrcnn.parameters(), lr=lr)

    val_count = 0
    print_iter = 200
    best_loss = np.inf

    if load_path is not None:
        print(f'Loading model from load_path: {load_path}')
        checkpoint_path = load_path
        checkpoint = torch.load(checkpoint_path)
        hrcnn.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        ave_loss = checkpoint['loss']
        val_count= checkpoint['val_count']
    elif continue_training:
        checkpoint_path, last_epoch = model_utils.get_last_checkpoint(save_root, end_epoch, model_type, dataset, loss_type, negative_prob, noise_width, nfft)
        if checkpoint_path is not None:
            start_epoch = last_epoch + 1
            best_loss = model_utils.get_best_loss(best_save_root, model_type, dataset, loss_type, negative_prob, noise_width, nfft)
            print(f'Continuing model training from {checkpoint_path} with best_loss of {best_loss}.')
            checkpoint = torch.load(checkpoint_path)
            hrcnn.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            ave_loss = checkpoint['loss']
            val_count= checkpoint['val_count']
            print('start_epoch:', start_epoch)
            print('ave_loss:', ave_loss)
            print('best_loss:', best_loss)

    hrcnn.train()
    print(hrcnn)

    comment = f'_{model_type}_{aug}_{dataset}_{loss_type}_{channels}_{dropout}_{dtype}_{batch_size}_{fpc}_{step}_{skip}_{negative_prob}_{noise_width}_{nfft}_{K}'
    writer = SummaryWriter(comment=comment)

    start = time.time()
    for epoch in range(start_epoch, end_epoch):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            frames, wave, live = (data[0].to(device), data[1].to(device), data[3].to(device))

            ## forward + backward + optimize
            optimizer.zero_grad()
            outputs = hrcnn(frames)
            loss = criterion(outputs, wave, live, nfft)
            loss.backward()
            optimizer.step()

            # Print the current loss and add to tensorboard writer
            running_loss += loss.item()
            if i % print_iter == (print_iter - 1):
                writer.add_scalar('*** Training loss',
                                  running_loss / print_iter,
                                  epoch * len(train_loader) + i)
                print('[%d, %5d] Train loss: %.6f' %
                     (epoch+1, i+1, running_loss / print_iter))
                running_loss = 0.0

        ## Validate model every so often
        hrcnn.eval()
        val_output_path = model_utils.get_output_tag(model_type, dataset, loss_type, negative_prob, noise_width, nfft, epoch=epoch)
        ave_loss = validate(hrcnn, model_type, val_loader, criterion, output_dir, val_output_path, fpc, step,
                 skip, use_hanning, nfft, writer=writer, val_count=val_count)
        val_count += 1

        save_path = os.path.join(save_root, val_output_path)
        save_model(save_path, epoch, hrcnn, optimizer, ave_loss, val_count)

        if ave_loss < best_loss:
            val_output_path = model_utils.get_output_tag(model_type, dataset, loss_type, negative_prob, noise_width, nfft, epoch=None)
            best_model_path = os.path.join(best_save_root, f'{val_output_path}.pth')
            save_model(best_model_path, epoch, hrcnn, optimizer, ave_loss, val_count)
            best_loss = ave_loss

        hrcnn.train()

    print('Finished Training')
    print('Took %.3f seconds total.' % (time.time() - start))
    writer.close()



def save_model(save_path, epoch, hrcnn, optimizer, loss, val_count):
    for i in range(5):
        try:
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': hrcnn.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': loss,
                        'val_count': val_count
                        }, save_path)
            break
        except:
            print('----- [%d] Could not save model. Trying again. -----' % i)
            time.sleep(5)


def validate(hrcnn, model_type, loader, criterion, output_dir, output_path, fpc, step, skip, use_hanning, nfft, writer=None, val_count=0):
    subjs = []
    waves = []
    HRs = []
    wave_preds = []
    HR_preds = []
    losses = []
    running_loss = 0
    all_losses = 0
    start = time.time()
    loader_iterator = iter(loader)
    iter_length = len(loader)
    print('')
    print('*** Validating Model ***')
    print('Total iters: ', iter_length)
    pbar = tqdm(total=iter_length)
    for i in range(iter_length):
        try:
            data = next(loader_iterator)
        except StopIteration:
            loader_iterator = iter(loader)
            data = next(loader_iterator)

        frames, wave, HR, live, subj = (data[0].to(device), data[1].to(device), data[2], data[3].to(device), data[4])

        with torch.set_grad_enabled(False):
            outputs = hrcnn(frames)
            loss = criterion(outputs, wave, live, nfft)
            all_losses += loss.item()
            wave_pred_copy = deepcopy(outputs.cpu().numpy())
            wave_preds.append(wave_pred_copy)
            subj_copy = deepcopy(subj.cpu().numpy())
            subjs.append(subj_copy)
            HR_copy = deepcopy(HR.cpu().numpy())
            HRs.append(HR_copy)
            wave_copy = deepcopy(wave.cpu().numpy())
            waves.append(wave_copy)
            del subj
            del HR
            del loss
            del wave
            del outputs

        pbar.update(1)

    pbar.close()

    del loader_iterator
    ave_loss = all_losses / iter_length
    print('Loss: %.6f' % (ave_loss))
    print('************************')
    print('')

    wave_pred = np.vstack(wave_preds)
    wave = np.vstack(waves)
    HRs = np.vstack(HRs)
    subjs = np.hstack(subjs)
    print('HR: ', HRs.shape)
    print('Waves: ', wave.shape, wave_pred.shape)
    print('Subjects: ', subjs.shape)

    wave_preds, wave_arrs, HR_arrs = validate_script.partition_by_subject(wave_pred, wave, HRs, subjs)
    print('After partitioning: ', wave_preds.shape, wave_arrs.shape, HR_arrs.shape)
    pred_arrs = validate_script.overlap_add(wave_preds, fpc, step, use_hanning=use_hanning)
    wave_arrs, HR_arrs = validate_script.flatten_ground_truth(wave_arrs, HR_arrs)
    print('After overlap_add: ', pred_arrs.shape, wave_arrs.shape, HR_arrs.shape)

    ## Save the outputs for each model
    np.save(os.path.join(output_dir, '%s_oadd_preds_1.npy' % (output_path)), pred_arrs)
    np.save(os.path.join(output_dir, '%s_pred_wave_1.npy' % (output_path)), wave_preds)
    if val_count == 0:
        np.save(os.path.join(output_dir, '%s_gt_wave_1.npy' % (output_path)), wave_arrs)
        np.save(os.path.join(output_dir, '%s_gt_HR_1.npy' % (output_path)), HR_arrs)
        np.save(os.path.join(output_dir, '%s_subj_1.npy' % (output_path)), subjs)

    writer.add_scalar('*** Validation loss',
                      ave_loss,
                      val_count)
    print('[%5d] Validation Loss: %.6f' %
          (val_count+1, ave_loss))
    print('Took %.3f seconds.' % (time.time() - start))
    print('************************')
    print('')
    return ave_loss



def load_seeds(seed_file, K):
    with open(seed_file, 'r') as infile:
        data = json.load(infile)
    return int(data[str(K)]['model']), int(data[str(K)]['training'])


if __name__ == '__main__':
    main()

