import cv2
import numpy as np
import os
import json
import csv
import re

from keras.utils import Sequence
from scipy.io import wavfile
from torch.utils.data import Dataset
import torch
from spatial_stream_weights import utils as ut
from scipy.signal import stft
from itertools import chain


class AVDIAR(Dataset):
    """
    Dataset class for the AVDIAR_TRAIN dataset
    """
    def __init__(self, root: str, number_speakers: int,
                 sample_length: float = 5, test_set: bool = False, rgb: bool = False, room_ids: tuple = (1, 2, 3), fft: bool = False):
        """
        Constructor of the AVDIAR_TRAIN class
        :param root: path to the root folder
        :param number_speakers: number of speakers the dataset should contain
        :param sample_length: length of video samples in seconds
        """
        self.test_set = test_set
        self.root = root
        self.number_speakers = number_speakers
        self.sample_length = sample_length
        self.room_ids = room_ids
        self.dirs = self._get_sequence_paths()
        self.rgb = rgb
        self.fft = fft

    def _get_sequence_paths(self) -> list:
        """
        Function that returns the paths to all subfolders in the root directory as a list
        :return: path list
        """
        paths = [os.path.join(self.root, path) for path in os.listdir(self.root)
                 if (int(path.split('-')[1][0]) == self.number_speakers) or (self.number_speakers == -1)]
        # Check for Room IDs only leaving the paths where the Room IDs match
        room_ids = list(self.room_ids)
        for path in paths:
            with open(os.path.join(path, 'summary.json')) as json_file:
                summary = json.load(json_file)
                if not summary['CalibrationID'] in room_ids:
                    paths.remove(path)
        return paths

    def _get_video_sequence(self, summary: dict, path: str, sample_length_frames: int,
                            start_frame: int, length_frames: int) -> np.ndarray:
        """
        Reads in a sequence of a video inside the specified folder, starting at the given start frame with a specified
        length.
        :param summary: dict containing the meta information summary
        :param path: path to the sample subfolder
        :param sample_length_frames: total length of the sequence as a number of frames
        :param start_frame: starting frame
        :param length_frames: total number of frames of the video file
        :return: video sequence as numpy array
        """
        # Preallocate memory
        resolution = summary["Image_Resoultion_WH"]
        if not self.rgb:
            frames = np.zeros((sample_length_frames, int(resolution[1]), int(resolution[0])))
        else:
            frames = np.zeros((sample_length_frames, int(resolution[1]), int(resolution[0]), 3))

        # Opening video file
        cap = cv2.VideoCapture(os.path.join(path, "Video/" + summary["SequenceName"] + "_CAM1.mp4"))
        if not cap.isOpened():
            raise IOError("Error opening video stream or file: ", path)

        # forwarding to start frame
        if not start_frame == 0:
            for i in range(length_frames + start_frame):
                _, _ = cap.read()
                if i == start_frame - 1:
                    break

        # picking out frames
        for i in range(sample_length_frames):
            ret, frame = cap.read()

            if ret:
                if not self.rgb:
                    frames[i] = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    frames[i] = frame
            else:
                raise IOError("Error opening video frame: ", frame)

        # Close file and return video sequence
        cap.release()
        return frames

    def _get_annotation(self, path: str, sample_length_frames: int, start_frame: int) -> np.ndarray:
        """
        Reads the annotation file and returns the annotation for the specified frames
        :param path: path to the sample subfolder
        :param sample_length_frames: total length of the sequence as a number of frames
        :param start_frame: starting frame
        :return: annotation as numpy array in the format: [frame_no, spaker_no, bb_left, bb_top, bb_width, bb_height]
        """
        # Reading csv file
        file = open(os.path.join(path, "GroundTruth/face_bb.txt"), 'r')
        csv_file = csv.reader(file, delimiter=',')

        # Reading first file to determine the first annotated frame
        first_row = csv_file.__next__()

        # Preallocate annotation array
        face_bbs = np.zeros((sample_length_frames, self.number_speakers, 5))
        frame_count = 0

        speaker_counter = 0
        # If sequence starts before the speaker enters the video the array is zero padded
        if int(first_row[0]) > start_frame:
            for _ in range(int(first_row[0]) - start_frame):
                for j in range(self.number_speakers):
                    face_bbs[frame_count, j] = np.array([start_frame + frame_count, 0, 0, 0, 0])
                frame_count = frame_count + 1
            # Entering the first frame into the array
            face_bbs[int(first_row[0]), 0] = np.array([int(first_row[0]), float(first_row[2]), float(first_row[3]), float(first_row[4]), float(first_row[5])])
            if self.number_speakers == 1:
                frame_count = frame_count + 1
            else:
                speaker_counter = speaker_counter + 1

        # If the sequence starts with the first annotated frame the this frame is put into the output array
        elif int(first_row[0]) == start_frame:
            face_bbs[0, speaker_counter] = np.array([start_frame, float(first_row[2]), float(first_row[3]), float(first_row[4]), float(first_row[5])])
            if self.number_speakers == 1:
                frame_count = frame_count + 1
            else:
                speaker_counter = speaker_counter + 1

        # Reading the remaining annotation
        for row in csv_file:
            if not (frame_count + start_frame == int(row[0])) and (frame_count + start_frame == int(row[0]) - 1):
                frame_count = frame_count + 1
                speaker_counter = 0
                if frame_count >= sample_length_frames:
                    break
            if frame_count + start_frame == int(row[0]):
                face_bbs[frame_count, speaker_counter] = np.array([int(row[0]), float(row[2]), float(row[3]), float(row[4]), float(row[5])])
                if speaker_counter == self.number_speakers - 1:
                    speaker_counter = 0
                    frame_count = frame_count + 1
                else:
                    speaker_counter = speaker_counter + 1
            if frame_count >= sample_length_frames:
                break

        # Closing file and returning result
        file.close()
        return face_bbs

    def _get_sound_sequence(self, path: str, summary: dict, start_frame: int, sample_length_frames: int) -> np.ndarray:
        """

        :param path: path to the sample subfolder
        :param summary: dict containing the meta information summary
        :param start_frame: starting frame
        :param sample_length_frames: total length of the sequence as a number of frames
        :return: sound sequence as numpy array
        """
        # Reading wav file
        sampling_rate, signal = wavfile.read(os.path.join(path, 'Audio/' + summary["SequenceName"] + ".wav"))

        # Calculating starting sample from start frame
        start_sample = int(np.floor(start_frame/25*sampling_rate))

        # Calculating length of sound sequence in samples from the length in frames
        length_sample = int(np.floor(sample_length_frames/25*sampling_rate))

        # Cutting output array to the specified length
        signal = signal[start_sample:start_sample + length_sample]
        signal = np.reshape(signal, (sample_length_frames, -1, 6))
        return signal

    def _get_fft(self, signal, sampling_rate, num_channels):
        n_fft = 256
        l_window_sec = 0.005
        l_window_samples = ut.sec_to_samples(l_window_sec, sampling_rate)
        overlap_samples = l_window_samples//4
        # hop_samples = ut.sec_to_samples(l_window_sec-overlap_sec, sampling_rate)
        stfts = [stft(signal[:, i].astype(float), sampling_rate, nfft=n_fft,
                      nperseg=l_window_samples, noverlap=overlap_samples)
                 for i in range(num_channels)]
        x = torch.from_numpy(np.moveaxis(np.array(list(chain.from_iterable(
            (np.abs(channel[2][1:].T), np.angle(channel[2])[1:].T) for channel in stfts))), 1, 2))
        return x

    def __len__(self):
        return 2*len(self.dirs)

    def __getitem__(self, index) -> tuple:
        """
        Function to get the new item from the dataset
        :param index: index of the new item in the dataset
        :return: tuple of video_sequence(ndarray) sound_sequence(ndarray) and annotation(ndarray)
        """
        # Caching current path
        path = self.dirs[index]

        # Reading meta information
        sum_path = os.path.join(path, "summary.json")
        with open(sum_path) as json_file:
            summary = json.load(json_file)
        length_frames = summary["Number_of_Image"]

        # Getting Video Path
        video_path = os.path.join(path, "Video/" + summary["SequenceName"] + "_CAM1.mp4")

        # Getting sample length in frames
        sample_length_frames = int(np.floor(self.sample_length * summary["Video_FPS"]))

        # Getting data and annotation
        if self.test_set:
            # Last second is reserved for audio training
            start_frames = range(0, sample_length_frames*(length_frames//sample_length_frames), sample_length_frames -
                                 (5 * summary["Video_FPS"]))
            annotation = []
            video_sequence = []
            sound_sequence = []
            for frame in start_frames:
                annotation.append(self._get_annotation(path, sample_length_frames, frame))
                video_sequence.append(self._get_video_sequence(summary, path, sample_length_frames, frame, length_frames))
                sound_sequence.append(self._get_sound_sequence(path, summary, frame, sample_length_frames))
            annotation = np.array(annotation)
            video_sequence = np.array(video_sequence, dtype="uint8")
            sound_sequence = np.array(sound_sequence)
        else:
            # Last second is reserved for audio training
            start_frame = np.random.randint(0, length_frames - sample_length_frames)
            annotation = self._get_annotation(path, sample_length_frames, start_frame)
            video_sequence = self._get_video_sequence(summary, path, sample_length_frames, start_frame, length_frames)
            sound_sequence = self._get_sound_sequence(path, summary, start_frame, sample_length_frames)
        return video_sequence, sound_sequence, annotation, video_path


class AvCal(Dataset):
    def __init__(self, root, room_id, grid_size, samples):
        self.target_resolution = (450, 720)
        self.resolution = (1200, 1920)
        self.samples = samples
        self.grid_size = grid_size
        self.root = root
        self.room_id = room_id
        self.dirs = self._get_sequence_paths()
        self.min_sound_length = self._get_min_sound_length()

    def _get_sequence_paths(self) -> list:
        """
        Function that returns the paths to all subfolders with specified room id in the root directory as a list
        :return: path list
        """
        sub_folder = self.root + "AV_CALIB-ID-" + str(self.room_id) + "/wn"
        paths = [os.path.join(sub_folder, path) for path in os.listdir(sub_folder)]
        np.random.shuffle(paths)
        return paths

    def _get_min_sound_length(self):
        min_length = min([len(wavfile.read(path)[1]) for path in self.dirs])
        return min_length

    def __len__(self):
        return int(np.floor(self.min_sound_length / self.samples) * len(self.dirs))

    def __getitem__(self, index):
        """
        Function to get a new item from the dataset
        :param index: index of the new item in the dataset
        :return: tuple of video_sequence(ndarray) sound_sequence(ndarray) and annotation(ndarray)
        """
        return self._get_audio_features(index), self._get_position_data(index)

    def _get_audio_features(self, index):
        # Reading wav file
        mode = np.floor(index/len(self.dirs))
        index = index % len(self.dirs)
        sampling_rate, signal = wavfile.read(self.dirs[index])
        x = signal[int(mode*self.samples):int(mode*self.samples + self.samples)].astype(np.float)
        # return x
        return self._get_stft_tf(x, sampling_rate, signal.shape[1])


    def _get_stft_tf(self, signal, sampling_rate, num_channels):
        n_fft = 2048
        l_window_sec = 0.04
        # overlap_sec = 0.008
        l_window_samples = ut.sec_to_samples(l_window_sec, sampling_rate)
        # overlap_samples = ut.sec_to_samples(overlap_sec, sampling_rate)
        overlap_samples = l_window_samples//2
        # hop_samples = ut.sec_to_samples(l_window_sec-overlap_sec, sampling_rate)
        stfts = [stft(signal[:, i].astype(float), sampling_rate, nfft=n_fft,
                      nperseg=l_window_samples, noverlap=overlap_samples)
                 for i in range(num_channels)]
        x = np.moveaxis(np.array(list(chain.from_iterable((np.abs(channel[2][1:].T), np.angle(channel[2])[1:].T)
                                                          for channel in stfts))), 0, 2)
        return torch.from_numpy(x)

    def _get_position_data(self, index):
        index = index % len(self.dirs)
        y = np.zeros(self.grid_size)
        path = self.root + "AV_CALIB-ID-" + str(self.room_id) + "/speaker_position.txt"

        # Reading csv file
        file = open(path, 'r')
        csv_file = csv.reader(file, delimiter=',')

        # Forwarding to index
        for _ in range(index):
            csv_file.__next__()

        scale_x = self.target_resolution[0]/self.resolution[0]
        scale_y = self.target_resolution[1]/self.resolution[1]
        row = csv_file.__next__()
        speaker_x = float(row[4]) * scale_x
        speaker_y = float(row[3]) * scale_y
        # cell = ut.point_to_grid((speaker_x, speaker_y), self.grid_size, self.target_resolution)
        # y[cell] = 1
        # y = np.reshape(y, self.grid_size[0]*self.grid_size[1])
        return np.array([speaker_x, speaker_y])


class AVDIARCal(Sequence):
    def __init__(self, root="AVDIAR_TRAIN/", room_ids=(1, 2, 3), grid_size=(20, 24),
                 sample_length=48000*2, resolution=(450, 720), shuffle=True, vad=False):
        self.resolution = resolution
        self.shuffle = shuffle
        self.vad = vad
        if sample_length % 1920 == 0:
            self.sample_length = int(sample_length)
        else:
            self.sample_length = int(1920*np.ceil(sample_length/1920))
            print("Sample length must be a multiple of 1920, using {} instead".format(self.sample_length))
        self.grid_size = grid_size
        self.root = root
        self.room_ids = room_ids

        self.dirs = self._get_sequence_paths()
        self.data = self._read_all_data()

    def get_data_array(self):
        return self.data

    def _read_all_data(self):
        data = []
        counter = 0
        sample_length_frames = int(self.sample_length / 1920)
        for path in self.dirs:
            # Reading audio file
            audio = wavfile.read(path)[1]
            num_frames = np.floor(len(audio)/1920)
            audio = audio[:int(num_frames*1920)]
            audio = np.split(audio, num_frames)

            # Reading video file
            sequence_name = re.split(r'/|\\', path)[1]
            cap = cv2.VideoCapture(os.path.join(self.root, sequence_name, 'Video', sequence_name + '_CAM1.mp4'))
            if not cap.isOpened():
                raise IOError("Error opening video stream or file: ", path)

            # Reading csv file
            annotation_path = os.path.join(self.root, sequence_name, 'GroundTruth', 'face_bb.txt')
            file = open(annotation_path, 'r')
            csv_file = csv.reader(file, delimiter=',')
            annotation = [(int(row[0]), float(row[3]), float(row[2]), float(row[4]), float(row[5])) for row in csv_file]
            for i in reversed(range(annotation[0][0])):
                annotation.insert(0, (i, 0., 0., 0., 0.))

            for i in range(annotation[len(annotation) - 1][0], int(num_frames)):
                annotation.append((i, 0., 0., 0., 0.))

            current_frame_index = 0
            frames = []
            current_frame = []
            for row in annotation:
                if row[0] == current_frame_index:
                    current_frame.append(row)
                else:
                    if row[0] > num_frames:
                        break
                    current_frame_index = row[0]
                    frames.append(np.array(current_frame))
                    current_frame.clear()
                    current_frame.append(row)

            audio_sequence = np.zeros((self.sample_length, 6))

            # Reading vad

            if self.vad:
                vad_path = os.path.join(self.root, sequence_name, 'GroundTruth', 'vad.rttm')
                file = open(vad_path, 'r')
                csv_file = csv.reader(file, delimiter=' ')
                # Skipping header
                _ = csv_file.__next__()
                vad_data = [(float(row[3]), float(row[3]) + float(row[4])) for row in csv_file]
            else:
                vad_data = None
            # Preparing data for training
            y = np.zeros(self.grid_size)
            current_frame_index = 0
            marked_counter = 0
            for anno, samples in zip(frames, audio):
                if current_frame_index == sample_length_frames:
                    if marked_counter == 0:
                        marked_counter = 1
                    video = [cap.read()[1] for _ in range(current_frame_index)]
                    data.append((self._get_stft(audio_sequence, 48000, 6), np.array(video),
                                 np.reshape(y/marked_counter, self.grid_size[0] * self.grid_size[1]), sequence_name))
                    y = np.zeros(self.grid_size)
                    audio_sequence = np.zeros((self.sample_length, 6))
                    marked_counter = 0
                    current_frame_index = 0
                if self._check_vad(anno[0][0]/25, vad_data):
                    for face in anno:
                        y = self._mark_cell(y, face[1], face[2], face[3], face[4])
                        marked_counter += 1
                audio_sequence[current_frame_index*1920:current_frame_index*1920 + 1920] = samples
                current_frame_index += 1
            counter += 1
            print("\rreading data... {0:.2f}%".format(counter / len(self.dirs) * 100), end='')
        if self.shuffle:
            np.random.shuffle(data)
        print("")
        return data

    def _get_sequence_paths(self) -> list:
        """
        Function that returns the paths to all subfolders with specified room id in the root directory as a list
        :return: path list
        """
        paths = [os.path.join(self.root, path, 'Audio', path + '.wav') for path in os.listdir(self.root)]
        # Check for Room IDs only leaving the paths where the Room IDs match
        for path in paths:
            with open(os.path.join(self.root, re.split(r'/|\\', os.path.split(path)[0])[1], 'summary.json')) as json_file:
                summary = json.load(json_file)
                if not summary['CalibrationID'] in self.room_ids:
                    paths.remove(path)
        if self.shuffle:
            np.random.shuffle(paths)
        return paths

    def _mark_cell(self, target, x, y, w, h):
        cell = ut.point_to_grid(ut.bb_to_center(x, y, h, w),
                                self.grid_size, self.resolution)
        target[cell] = target[cell] + 1
        return target

    def _get_stft(self, signal, sampling_rate, num_channels):
        n_fft = 2048
        l_window_sec = 0.04
        # overlap_sec = 0.008
        l_window_samples = ut.sec_to_samples(l_window_sec, sampling_rate)
        # overlap_samples = ut.sec_to_samples(overlap_sec, sampling_rate)
        overlap_samples = l_window_samples//2
        # hop_samples = ut.sec_to_samples(l_window_sec-overlap_sec, sampling_rate)
        stfts = [stft(signal[:, i].astype(float), sampling_rate, nfft=n_fft,
                      nperseg=l_window_samples, noverlap=overlap_samples)
                 for i in range(num_channels)]
        x = np.moveaxis(np.array(list(chain.from_iterable((np.abs(channel[2][1:].T), np.angle(channel[2])[1:].T)
                                                          for channel in stfts))), 0, 2)
        return x

    @staticmethod
    def _check_vad(frame, vad):
        if vad is None:
            return True
        for time_frame in vad:
            if (time_frame[0] <= frame) and (frame <= time_frame[1]):
                return True
        return False

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Function to get a new item from the dataset
        :param index: index of the new item in the dataset
        :return: tuple of video_sequence(ndarray) sound_sequence(ndarray) and annotation(ndarray)
        """
        return self.data[index][0], self.data[index][1]
