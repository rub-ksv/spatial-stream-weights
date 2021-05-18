from spatial_stream_weights.utils import get_video_features, get_audio_features
from spatial_stream_weights import datasets
import joblib
grid_size = (20, 24)
sample_length = 1920
resolution = (450, 720)
cache_path = "G:\\Python Cache\\test"
audio = False

if audio:
    dataset = datasets.AVDIARCal(root="AVDIAR_TEST/", grid_size=grid_size, sample_length=sample_length,
                                      resolution=resolution, shuffle=False)
    dataset = dataset.get_data_array()
    joblib.dump(dataset, cache_path + "_raw")
    audio_set = get_audio_features(dataset, grid_size)
    joblib.dump(audio_set, cache_path + "_audio")
else:
    dataset = joblib.load(cache_path + "_raw")
    audio_set = joblib.load(cache_path + "_audio")
    complete = get_video_features(dataset, audio_set, grid_size, resolution)
    joblib.dump(complete, cache_path)


