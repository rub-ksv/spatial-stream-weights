import numpy as np
import torch
import pickle
import cv2
import argparse
import joblib

from shapely.geometry import box
from scipy.signal import stft
from itertools import chain
from PIL import Image
from yolo.yolo import detect_img
from spatial_stream_weights.models import AudioLocalizer
from yolo.yolo import YOLO



def compute_evaluation_metrics(outputs, targets, detection_threshold=5):
    outputs = outputs.cpu().detach().numpy().flatten()
    targets = targets.cpu().detach().numpy().flatten()

    residuals = outputs - targets

    azimuth_errors = np.abs(np.rad2deg(np.arctan2(np.sin(residuals), np.cos(residuals))))
    num_samples = len(azimuth_errors)

    crmse = np.mean(azimuth_errors)

    true_positives = np.sum(azimuth_errors < detection_threshold)
    accuracy = true_positives / num_samples

    return crmse, accuracy


def convert_bb(x, y, w, h):
    x_min = x
    x_max = x + w
    y_min = y
    y_max = y + h
    return x_min, y_min, x_max, y_max


def convert_annotation(gt):
    output = [[]]
    for speaker in gt:
        x_min, x_max, y_min, y_max = convert_bb(speaker[1], speaker[2], speaker[3], speaker[4])
        output.append([x_min, x_max, y_min, y_max])
    return output


def face_detected_correctly(gt, predicted, threshold):
    if calculate_iou(convert_bb(gt[0], gt[1], gt[2], gt[3]),
                     convert_bb(predicted[0], predicted[1], predicted[2], predicted[3])) > threshold:
        return 1
    else:
        return 0


# metrics from https://github.com/rafaelpadilla/Object-Detection-Metrics#precision-x-recall-curve
def calculate_iou(box_1, box_2):
    if all(item == 0 for item in box_1) and all(item == 0 for item in box_2):
        return 1
    poly_1 = box(box_1[0], box_1[1], box_1[2], box_1[3])
    poly_2 = box(box_2[0], box_2[1], box_2[2], box_2[3])
    intersection = poly_1.intersection(poly_2).area
    union = poly_1.union(poly_2).area
    iou = intersection / union
    return iou


def get_rating(gt, predicted, threshold, yolo=False):
    if all(item == 0 for item in predicted) and not all(item == 0 for item in gt):
        return 'FN'
    else:
        iou = calculate_iou(convert_bb(gt[0], gt[1], gt[2], gt[3]),
                            convert_bb(predicted[0], predicted[1], predicted[2], predicted[3]))
        if iou >= threshold:
            return 'TP'
        else:
            return 'FP'


def pad_image(image, multiple):
    pad_0 = (0, 0)
    pad_1 = (0, 0)

    if not (image.shape[0] % multiple == 0):
        size_0 = image.shape[0] + (multiple - image.shape[0] % multiple)
        pad_size_0 = size_0 - image.shape[0]
        if pad_size_0 % 2 == 0:
            pad_0 = (int(pad_size_0/2), int(pad_size_0/2))
        else:
            pad_0 = (int(np.floor(pad_size_0/2) + 1), int(np.floor(pad_size_0/2)))

    if not (image.shape[1] % multiple == 0):
        size_1 = image.shape[1] + (multiple - image.shape[1] % multiple)
        pad_size_1 = size_1 - image.shape[1]
        if pad_size_1 % 2 == 0:
            pad_1 = (int(pad_size_1/2), int(pad_size_1/2))
        else:
            pad_1 = (int(np.floor(pad_size_1/2) + 1), int(np.floor(pad_size_1/2)))

    image = np.pad(image, (pad_0, pad_1), mode='constant')
    return image


def calculate_precision(TP, FP):
    return TP/(TP + FP)


def calculate_recall(TP, FN):
    return TP/(TP + FN)


def sec_to_samples(sec, sampling_rate):
    out = int(sec*sampling_rate)
    return out


def point_to_grid(point, grid_size, resolution):
    pixel_per_cell = (resolution[0]/grid_size[0], resolution[1]/grid_size[1])
    cell = (int(point[0]//pixel_per_cell[0]),
            int(point[1]//pixel_per_cell[1]))
    cell = cap_cell(cell, grid_size)
    return cell


def cap_cell(cell, grid_size):
    if cell[0] > grid_size[0] - 1:
        cell = (grid_size[0] - 1, cell[1])
    if cell[1] > grid_size[1] - 1:
        cell = (cell[0], grid_size[1] - 1)
    return cell


def grid_to_point(cell, grid_size, resolution):
    pixel_per_cell = (resolution[0] // grid_size[0], resolution[1] // grid_size[1])
    point = (int(cell[0] * pixel_per_cell[0]), int(cell[1] * pixel_per_cell[1]))
    return point


def bb_to_center(x, y, w, h):
    return x + h/2, y + w/2


def calculate_distance(predicted, target):
    predicted = np.array(predicted)
    target = np.array(target)
    return np.linalg.norm(predicted-target)


def calculate_error(true, pred, grid_size):
    true = true.to('cpu')
    pred = pred.to('cpu')
    error = 0
    for t, p in zip(true.data, pred.data):
        t = t.reshape(grid_size).numpy()
        p = p.reshape(grid_size).numpy()
        t_idx = np.unravel_index(t.argmax(), t.shape)
        p_idx = np.unravel_index(p.argmax(), p.shape)
        error = error + np.sqrt(np.power(t_idx[0]-p_idx[0], 2) + np.power(t_idx[1] - p_idx[1], 2))
    return error


def normalize_sound(x):
    x = x.astype(np.float)
    x = (x - x.min()) / (x.max() - x.min())
    return x


def apply_fft(signal, sampling_rate, num_channels):
    n_fft = 256
    l_window_sec = 0.005
    l_window_samples = sec_to_samples(l_window_sec, sampling_rate)
    overlap_samples = l_window_samples // 4
    # hop_samples = ut.sec_to_samples(l_window_sec-overlap_sec, sampling_rate)
    stfts = [stft(signal[:, i].astype(float), sampling_rate, nfft=n_fft,
                  nperseg=l_window_samples, noverlap=overlap_samples)
             for i in range(num_channels)]
    x = torch.from_numpy(np.moveaxis(np.array(list(chain.from_iterable(
        (np.abs(channel[2][1:].T), np.angle(channel[2])[1:].T) for channel in stfts))), 1, 2))
    return x


def transform_dataset(dataset=None, dataset_path="cached_files", sets=("train", "test", "val")):
    if dataset is None:
        # Restoring dataset from file:
        print("Loading dataset from: ", dataset_path)
        if "train" in sets:
            train_set = joblib.load(dataset_path + "/train_set")
        else:
            train_set = None
        if "test" in sets:
            test_set = joblib.load(dataset_path + "/test_set")
        else:
            test_set = None
        if "val" in sets:
            val_set = joblib.load(dataset_path + "/val_set")
        else:
            val_set = None
    else:
        train_set = dataset[0]
        test_set = dataset[1]
        val_set = dataset[2]

        print("Converting Pytorch dataset to Keras Format.")
        train_data = train_set.get_data_array()
        test_data = test_set.get_data_array()
        val_data = val_set.get_data_array()

        train_set = (np.array([np.squeeze(tup[0]) for tup in train_data]),
                     np.array([np.squeeze(tup[2]) for tup in train_data]))

        test_set = (np.array([np.squeeze(tup[0]) for tup in test_data]),
                    np.array([np.squeeze(tup[2]) for tup in test_data]))

        val_set = (np.array([np.squeeze(tup[0]) for tup in val_data]),
                   np.array([np.squeeze(tup[2]) for tup in val_data]))

        # Saving dataset to Cache
        print("Caching Dataset to: ", dataset_path)
        with open(dataset_path, 'wb+') as f:
            pickle.dump([train_set, test_set, val_set], f, protocol=4)

    return train_set, test_set, val_set


def swap(tup):
    return tup[1], tup[0]


def extract_target_table(heatmap):
    positions = np.zeros((4, 2))
    persons_visible = np.zeros(4)
    for i in range(4):
        index = np.unravel_index(heatmap.argmax(), heatmap.shape)
        if heatmap[index] > 0 and np.count_nonzero(index) > 0:
            heatmap[index] = 0
            positions[i] = index
            persons_visible[i] = 1
    return persons_visible, positions


def mark_cell(target, x, y, w, h, grid_size, resolution):
    cell = point_to_grid(bb_to_center(x, y, h, w), grid_size, resolution)
    target[cell] = target[cell] + 1
    return target


def detect_faces(input_frame, yolo_detector):
    pil_frame = Image.fromarray(cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB))
    bbs, _ = detect_img(yolo_detector, True, pil_frame)
    index = 0
    for bb in bbs:
        top, left, bottom, right = bb
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(pil_frame.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(pil_frame.size[0], np.floor(right + 0.5).astype('int32'))
        bbs[index] = np.array([left, top, right-left, bottom-top])
        index = index + 1
    return bbs


def get_audio_features(data_array, grid_size):
    # Get Raw Data
    train_output = []

    # Creating Audio Localizer
    predictor = AudioLocalizer(grid_size=grid_size, fft=False)

    # Extract Features and Labels
    counter = 0
    total = len(data_array)
    for data in data_array:

        # Getting Target data
        true_label = np.reshape(data[2], grid_size)
        persons_visible, positions = extract_target_table(true_label)

        # Prediction from Audio
        audio_prediction = predictor.predict_direction(data[0], heatmap=False)

        train_output.append((audio_prediction, persons_visible, positions, data[3].split('-')[0][3:]))

        counter += 1
        print("\rcalculated: {0:.2f}%".format(100*counter/total), end='')
    return train_output


def get_video_features(input_array, output_array, grid_size, resolution):
    # Creating  YOLO detector
    args = argparse.Namespace()
    args.model = 'model-weights/YOLO_Face.h5'
    args.anchors = 'cfg/yolo_anchors.txt'
    args.classes = 'cfg/face_classes.txt'
    args.score = 0.5
    args.iou = 0.45
    args.img_size = (832, 832)
    yolo_detector = YOLO(args)

    total = len(input_array)
    for i in range(total):
        # Prediction from Video
        data = input_array[i]
        faces_frame = detect_faces(np.squeeze(data[1]).astype(np.uint8), yolo_detector)
        video_prediction = np.zeros(grid_size)
        for bb in faces_frame:
            video_prediction = mark_cell(video_prediction, bb[1], bb[0], bb[2], bb[3], grid_size, resolution)
        output_array[i] = (video_prediction, ) + output_array[i]

        print("\rcalculated: {0:.2f}%".format(100 * (i + 1) / total), end='')

    return output_array


def get_speaker_number(array):
    speakers = np.full(len(np.where(array > 0.5)[0]), 1)
    if len(speakers) >= 4:
        return np.array([1, 1, 1, 1])
    else:
        speakers.resize(4, refcheck=False)
        return speakers


def calculate_metrics(test_list, grid_size, threshold=0.5):
    counter = 0
    accuracy = 0
    total_speakers = 0
    for pred in test_list:
        # Frame recall
        prediction = np.squeeze(pred[0][0])
        target = pred[1]
        prediction[prediction < threshold] = 0
        speaker_predicted = len(np.argwhere(prediction))
        speaker_target = len(np.argwhere(target))
        total_speakers += speaker_target

        if speaker_target == speaker_predicted:
            counter += 1
            # binary frame accuracy
            prediction_image = np.squeeze(pred[0][1])
            target_image = np.squeeze(pred[2])
            targets = []
            for i in range(speaker_target):
                ind = np.argmax(target_image, axis=None)
                ind = np.unravel_index(ind, grid_size)
                target_image[ind] = 0
                targets.append(ind)
            for i in range(speaker_predicted):
                ind = np.argmax(prediction_image, axis=None)
                ind = np.unravel_index(ind, grid_size)
                prediction_image[ind] = 0
                for target in targets:
                    if coord_in_range(ind, target):
                        accuracy += 1
                        break

    return counter/len(test_list), accuracy/total_speakers


def coord_in_range(predicted, target):
    if (target[0] - 1 <= predicted[0] <= target[0] + 1) and (target[1] - 1 <= predicted[1] <= target[1] + 1):
        return True
    else:
        return False
