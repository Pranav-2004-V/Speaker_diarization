import pyaudio
import librosa
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, LSTM, Dense
from sklearn.mixture import GaussianMixture

# Record audio using PyAudio
def record_audio(seconds):
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    frames = []
    for i in range(0, int(RATE / CHUNK * seconds)):
        data = stream.read(CHUNK)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    return np.frombuffer(b''.join(frames), dtype=np.int16)

# Extract features using Librosa
def extract_features(audio):
    # Convert audio to floating-point format
    audio_float = librosa.util.buf_to_float(audio, dtype=np.float32)
    # Use Librosa to extract features like Mel spectrograms, MFCCs, etc.
    mel_spectrogram = librosa.feature.melspectrogram(y=audio_float, sr=44100)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    return mel_spectrogram_db

# CNN model
def create_cnn_model(input_shape):
    model = Sequential()
    model.add(Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Flatten())
    return model

# RNN model
def create_rnn_model(input_shape):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(128))
    return model

# Combine CNN and RNN models
def create_combined_model(cnn_model, rnn_model):
    combined_model = Sequential()
    combined_model.add(cnn_model)
    combined_model.add(rnn_model)
    return combined_model

# GMM clustering
def perform_gmm_clustering(features):
    gmm = GaussianMixture(n_components=2)  # Assuming 2 speakers for simplicity
    gmm.fit(features)
    labels = gmm.predict(features)
    return labels

# Main function
if __name__ == '__main__':
    # Record audio
    audio = record_audio(5)  # Record for 5 seconds

    # Extract features
    features = extract_features(audio)

    # CNN model
    cnn_model = create_cnn_model(input_shape=features.shape)

    # RNN model
    rnn_model = create_rnn_model(input_shape=features.shape)

    # Combined model
    combined_model = create_combined_model(cnn_model, rnn_model)

    # Cluster speakers using GMM
    labels = perform_gmm_clustering(features)

    print(labels)
