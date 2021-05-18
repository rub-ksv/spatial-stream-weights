import tensorflow as tf
import numpy as np
import gc
import cv2
import argparse

from spatial_stream_weights import utils as ut
from keras.models import load_model
from keras.backend.tensorflow_backend import set_session
from scipy.signal import stft
from itertools import chain
from keras.backend.tensorflow_backend import clear_session
from yolo.yolo import detect_img, YOLO
from PIL import Image
from numba import cuda


class AudioLocalizer:
    def __init__(self, model_path="model/audio_recognizer_1920.h5", resolution=(450, 720), grid_size=(10, 12), fft=True):
        config = tf.ConfigProto(
            gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
            # device_count = {'GPU': 1}
        )
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        set_session(session)
        self.fft = fft
        self.model = load_model(model_path)
        self.grid_size = grid_size
        self.resolution = resolution
        self.pixel_per_cell = (self.resolution[0]//self.grid_size[0], self.resolution[1]//self.grid_size[1])

    def predict_direction(self, audio_input, heatmap=True):
        if self.fft:
            audio_input = self._get_stft(audio_input)
        prediction = np.reshape(self.model.predict(np.expand_dims(audio_input, 0), batch_size=1), self.grid_size)
        if heatmap:
            prediction = self._prediction_to_heatmap(prediction)
        return prediction

    def _prediction_to_heatmap(self, prediction):
        upscaled_prediction = cv2.resize(prediction, ut.swap(self.resolution))
        return upscaled_prediction

    def _prediction_to_bb(self, prediction):
        bb = ([0, 0, 0, 0],)
        for p in prediction.data:
            p = p.reshape(self.grid_size).numpy()
            p_idx = ut.grid_to_point(np.unravel_index(p.argmax(), p.shape), self.grid_size, self.resolution)
            bb = ([p_idx[0], p_idx[1], self.pixel_per_cell[1], self.pixel_per_cell[0]],)
        return bb

    @staticmethod
    def _get_stft(signal, sampling_rate=48000, num_channels=6):
        n_fft = 2048
        l_window_sec = 0.04
        l_window_samples = ut.sec_to_samples(l_window_sec, sampling_rate)
        overlap_samples = l_window_samples//2

        stfts = [stft(signal[:, i].astype(float), sampling_rate, nfft=n_fft,
                      nperseg=l_window_samples, noverlap=overlap_samples)
                 for i in range(num_channels)]
        x = np.moveaxis(np.array(list(chain.from_iterable((np.abs(channel[2][1:].T), np.angle(channel[2])[1:].T)
                                                          for channel in stfts))), 0, 2)
        return x

    def clear(self, clear_model=False):
        clear_session()
        if clear_model:
            del self.model
            cuda.select_device(0)
            cuda.close()
        for _ in range(5):
            gc.collect()


class VideoLocalizer:
    def __init__(self):
        args = argparse.Namespace()
        args.model = 'model-weights/YOLO_Face.h5'
        args.anchors = 'cfg/yolo_anchors.txt'
        args.classes = 'cfg/face_classes.txt'
        args.score = 0
        args.iou = 0.5
        args.img_size = (832, 832)
        self.yolo_detector = YOLO(args)

    def predict_faces(self, frame):
        bbs = self._detect_faces(frame)
        if len(bbs) == 0:
            bbs = ([0, 0, 0, 0],)
        return bbs[0]

    def _detect_faces(self, input_frame):
        pil_frame = Image.fromarray(cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB))
        bbs, _ = detect_img(self.yolo_detector, True, pil_frame)
        index = 0
        for bb in bbs:
            top, left, bottom, right = bb
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(pil_frame.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(pil_frame.size[0], np.floor(right + 0.5).astype('int32'))
            bbs[index] = np.array([left, top, right - left, bottom - top])
            index = index + 1
        return bbs