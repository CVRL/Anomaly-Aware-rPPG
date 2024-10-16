import numpy as np
import pandas as pd
import cv2
import os

OUT_IMG_HEIGHT = 64
OUT_IMG_WIDTH = 64


def landmarks_to_array(lmrk_path):
    df = pd.read_csv(lmrk_path)
    df.columns = df.columns.str.replace(' ', '')

    ## Convert to [T,68,2] array
    x_lmrks = [df['x_%d' % i] for i in range(0, 68)]
    y_lmrks = [df['y_%d' % i] for i in range(0, 68)]
    x_lmrks = np.asarray(x_lmrks).T
    y_lmrks = np.asarray(y_lmrks).T
    lmrks = np.stack((x_lmrks, y_lmrks), -1)

    ## Clean the array
    bad_idcs = bad_lmrks(lmrks)
    clean_lmrks = close_good_lmrks(lmrks, bad_idcs)
    assert(lmrks.shape == clean_lmrks.shape)
    return clean_lmrks


def bad_lmrks(lmrks):
    zero_row = np.zeros_like(lmrks[0])
    idcs = np.where(np.all(lmrks == zero_row, axis=(1,2)))[0]
    return idcs


def close_good_lmrks(lmrks, bad_idcs):
    consec_idcs = consecutive(bad_idcs)
    for consec in consec_idcs:
        if consec.size > 0:
            upper_idx = consec[-1] + 1
            lower_idx = consec[0] - 1
            if upper_idx == lmrks.shape[0]:
                upper_idx = lower_idx
            if lower_idx == -1:
                lower_idx = upper_idx
            split_idx = int((consec[-1] + consec[0]) / 2)
            lmrks[consec[:split_idx]] = lmrks[lower_idx]
            lmrks[consec[split_idx:]] = lmrks[upper_idx]
    return lmrks


def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


def make_video_array_from_directory(vid_dir, lmrks):
    vid_len = len(lmrks)
    video_idx = 0
    frame_list = [os.path.join(vid_dir, f) for f in sorted(os.listdir(vid_dir))]
    output_video = np.zeros((vid_len, OUT_IMG_HEIGHT, OUT_IMG_WIDTH, 3), dtype=np.uint8)
    successful = True

    for frame_path in frame_list:
        frame = cv2.imread(frame_path)
        if video_idx == 0:
            img_h, img_w = frame.shape[:2]

        if video_idx < vid_len:
            lmrk = lmrks[video_idx]
        else: #lmrks are shorter than video
            successful = False
            print('ERROR: Fewer landmarks than video frames, must relandmark with OpenFace.')
            break

        lmrk = lmrk.astype(int)
        bbox = get_bbox(lmrk, img_w, img_h)
        square_bbox = get_square_bbox(bbox, img_w, img_h)

        x1,y1,x2,y2 = square_bbox
        cropped = frame[y1:y2, x1:x2]
        if cropped.size < 1:
            resized = np.zeros((OUT_IMG_HEIGHT, OUT_IMG_WIDTH, 3), dtype=cropped.dtype)
        else:
            resized = cv2.resize(cropped, (OUT_IMG_HEIGHT, OUT_IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
        output_video[video_idx] = resized
        video_idx += 1

    if video_idx < vid_len:
        successful = False
        print(f'ERROR: Reached video idx {video_idx} while video was expected to be length {vid_len}.')

    return output_video, successful


def make_video_array(vid_path, lmrks, dtype=np.uint8):
    vid_len = len(lmrks)
    cap = cv2.VideoCapture(vid_path, cv2.CAP_FFMPEG)
    video_idx = 0
    output_video = np.zeros((vid_len, OUT_IMG_HEIGHT, OUT_IMG_WIDTH, 3), dtype=dtype)
    successful = True

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if video_idx == 0:
                img_h, img_w = frame.shape[:2]

            if video_idx < vid_len:
                lmrk = lmrks[video_idx]
            else: #lmrks are shorter than video
                successful = False
                print('ERROR: Fewer landmarks than video frames, must relandmark with OpenFace.')
                break

            lmrk = lmrk.astype(int)
            bbox = get_bbox(lmrk, img_w, img_h)
            square_bbox = get_square_bbox(bbox, img_w, img_h)

            x1,y1,x2,y2 = square_bbox
            cropped = frame[y1:y2, x1:x2]
            if cropped.size < 1:
                resized = np.zeros((OUT_IMG_HEIGHT, OUT_IMG_WIDTH, 3), dtype=cropped.dtype)
            else:
                resized = cv2.resize(cropped, (OUT_IMG_HEIGHT, OUT_IMG_WIDTH), interpolation=cv2.INTER_CUBIC)
            output_video[video_idx] = resized
            video_idx += 1
        else:
            break

    cap.release()

    if video_idx < vid_len:
        successful = False
        print(f'ERROR: Reached video idx {video_idx} while video was expected to be length {vid_len}.')

    return output_video, successful


def get_spatial_averages_from_directory(vid_dir, lmrks):
    vid_len = len(lmrks)
    video_idx = 0
    frame_list = [os.path.join(vid_dir, f) for f in sorted(os.listdir(vid_dir))]
    successful = True
    signals = []
    bboxes = []

    for frame_path in frame_list:
        frame = cv2.imread(frame_path)

        if video_idx == 0:
            img_h, img_w = frame.shape[:2]

        if video_idx < vid_len:
            lmrk = lmrks[video_idx]
        else: #lmrks are shorter than video
            successful = False
            print('ERROR: Fewer landmarks than video frames, must relandmark with OpenFace.')
            break

        lmrk = lmrk.astype(int)
        bbox = get_bbox(lmrk, img_w, img_h)
        x1,y1,x2,y2 = bbox
        cropped = frame[y1:y2, x1:x2]
        if cropped.size > 0:
            sigs = np.mean(cropped, axis=(0,1))
        else:
            try:
                sigs = signals[-1]
            except:
                sigs = [0,0,0]
        bboxes.append(bbox)
        signals.append(sigs)
        video_idx += 1

    signals = np.vstack(signals)
    bboxes = np.vstack(bboxes)

    if video_idx < vid_len:
        successful = False
        print(f'ERROR: Reached video idx {video_idx} while video was expected to be length {vid_len}.')

    return signals, bboxes, successful


def get_spatial_averages(vid_path, lmrks):
    vid_len = len(lmrks)
    cap = cv2.VideoCapture(vid_path, cv2.CAP_FFMPEG)
    video_idx = 0
    signals = []
    bboxes = []
    successful = True

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            if video_idx == 0:
                img_h, img_w = frame.shape[:2]

            if video_idx < vid_len:
                lmrk = lmrks[video_idx]
            else: #lmrks are shorter than video
                successful = False
                print('ERROR: Fewer landmarks than video frames, must relandmark with OpenFace.')
                break

            lmrk = lmrk.astype(int)
            bbox = get_bbox(lmrk, img_w, img_h)
            x1,y1,x2,y2 = bbox
            cropped = frame[y1:y2, x1:x2]
            if cropped.size > 0:
                sigs = np.mean(cropped, axis=(0,1))
            else:
                try:
                    sigs = signals[-1]
                except:
                    sigs = [0,0,0]
            bboxes.append(bbox)
            signals.append(sigs)
            video_idx += 1
        else:
            break
    signals = np.vstack(signals)
    bboxes = np.vstack(bboxes)
    cap.release()

    if video_idx < vid_len:
        successful = False
        print(f'ERROR: Reached video idx {video_idx} while video was expected to be length {vid_len}.')

    return signals, bboxes, successful


def get_bbox(lmrks, img_w, img_h):
    x_min, y_min = lmrks.min(axis=0)
    x_max, y_max = lmrks.max(axis=0)
    x_diff = x_max - x_min
    x_upper_pad = x_diff * 0.05
    x_lower_pad = x_diff * 0.05
    x_min -= x_upper_pad
    x_max += x_lower_pad
    if x_min < 0:
        x_min = 0
    if x_max > img_w:
        x_max = img_w
    y_diff = y_max - y_min
    y_upper_pad = y_diff * 0.3
    y_lower_pad = y_diff * 0.05
    y_min -= y_upper_pad
    y_max += y_lower_pad
    if y_min < 0:
        y_min = 0
    if y_max > img_h:
        y_max = img_h
    bbox = np.array([x_min, y_min, x_max, y_max]).astype(int)
    return bbox


def draw_lmrks(frame, lmrks):
    for lmrk in lmrks:
        lmrk = lmrk.astype(int)
        frame = cv2.circle(frame, tuple(lmrk), 1, (0, 255, 0), -1)
    return frame


def draw_bbox(frame, bbox):
    x1, y1, x2, y2 = bbox.astype(int)
    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return frame


def shift_inside_frame(x1,y1,x2,y2,img_w,img_h):
    if y1 < 0:
        y2 -= y1
        y1 -= y1
    if y2 > img_h:
        shift = y2 - img_h
        y1 -= shift
        y2 -= shift

    if x1 < 0:
            x2 -= x1
            x1 -= x1
    if x2 > img_w:
        shift = x2 - img_w
        x1 -= shift
        x2 -= shift

    return x1,y1,x2,y2


def get_square_bbox(bbox, img_w, img_h):
    x1,y1,x2,y2 = bbox
    w = x2 - x1
    h = y2 - y1
    x1,y1,x2,y2 = shift_inside_frame(x1,y1,x2,y2,img_w,img_h)
    w = x2 - x1
    h = y2 - y1

    ## Push the rectangle out into a square
    if w > h:
        d = w - h
        pad = int(d/2)
        y1 -= pad
        y2 += pad + (d % 2 == 1)
        x1,y1,x2,y2 = shift_inside_frame(x1,y1,x2,y2,img_w,img_h)
    elif w < h:
        d = h - w
        pad = int(d/2)
        x1 -= pad
        x2 += pad + (d % 2 == 1)
        x1,y1,x2,y2 = shift_inside_frame(x1,y1,x2,y2,img_w,img_h)

    if x1 < 0:
        x1 = 0
    if x2 > img_w:
        x2 = img_w
    if y1 < 0:
        y1 = 0
    if y2 > img_h:
        y2 = img_h

    w = x2 - x1
    h = y2 - y1
    return int(x1), int(y1), int(x2), int(y2)

