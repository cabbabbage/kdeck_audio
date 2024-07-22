import os
import io
import librosa
import numpy as np
import pandas as pd
import pyaudio
import webrtcvad
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
import joblib
from moviepy.editor import VideoFileClip
import tkinter as tk
from tkinter import filedialog

class SpeakerIdentification:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.model = SVC(kernel='linear', probability=True)
        self.features = []
        self.labels = []
        self.model_path = "speaker_identification_model.pkl"
        self.label_encoder_path = "label_encoder.pkl"

    def play_audio_segment(self, audio_segment):
        p = pyaudio.PyAudio()
        stream = p.open(format=p.get_format_from_width(audio_segment.sample_width),
                        channels=audio_segment.channels,
                        rate=audio_segment.frame_rate,
                        output=True)
        stream.write(audio_segment.raw_data)
        stream.stop_stream()
        stream.close()
        p.terminate()

    def extract_features(self, audio_segment):
        # Save the audio segment to a temporary file
        with io.BytesIO() as buffer:
            audio_segment.export(buffer, format="wav")
            buffer.seek(0)
            audio, sample_rate = librosa.load(buffer, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            return mfccs_scaled

    def extract_audio(self, file_path):
        if file_path.lower().endswith(('.mp4', '.mov')):
            video = VideoFileClip(file_path)
            audio_file_path = "extracted_audio.wav"
            video.audio.write_audiofile(audio_file_path)
        else:
            audio_file_path = file_path
        
        # Compress the audio
        audio = AudioSegment.from_file(audio_file_path)
        compressed_audio_file_path = "compressed_audio.wav"
        audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
        audio.export(compressed_audio_file_path, format="wav", bitrate="32k")
        return compressed_audio_file_path

    def load_and_segment_audio(self, audio_file_path):
        # Load audio file
        audio = AudioSegment.from_file(audio_file_path)
        
        vad = webrtcvad.Vad()
        vad.set_mode(2)  # Set aggressiveness mode to most aggressive (3)

        samples_per_window = 160  # 10 ms for 16 kHz audio
        sample_rate = 16000
        audio_data = np.array(audio.get_array_of_samples())

        # Buffer to accumulate speech frames to smooth out short pauses
        buffer = []
        in_speech = False
        segments = []
        
        for i in range(0, len(audio_data), samples_per_window):
            window = audio_data[i:i + samples_per_window]
            if len(window) < samples_per_window:
                continue
            is_speech = vad.is_speech(window.tobytes(), sample_rate)
            if is_speech:
                buffer.extend(window)
                in_speech = True
            else:
                if in_speech and len(buffer) > 0:
                    # Buffer short pauses
                    if len(buffer) / samples_per_window < 3:
                        buffer.extend(window)
                    else:
                        if len(buffer) / sample_rate >= 1.5:  # Only save segments longer than 3 seconds
                            segments.append(buffer)
                        buffer = []
                        in_speech = False
                else:
                    buffer = []

        # Handle the last segment if it exists and is longer than 3 seconds
        if buffer and len(buffer) / sample_rate >= 3:
            segments.append(buffer)

        return segments

    def label_segments(self, segments):
        for segment in segments:
            segment_audio = AudioSegment(
                data=np.array(segment).tobytes(),
                sample_width=2,  # assuming 16-bit audio
                frame_rate=16000,
                channels=1
            )
            self.play_audio_segment(segment_audio)
            speaker = input("Enter the name of the speaker (or 'unknown'): ")
            if speaker.lower() == 'unknown':
                continue

            features = self.extract_features(segment_audio)
            self.features.append(features)
            self.labels.append(speaker)

    def train_and_save_model(self):
        # Ensure we have enough data to split
        if len(self.features) < 2:
            raise ValueError("Not enough segments to perform train-test split. Ensure the audio contains sufficient recognizable speech.")

        # Encode the labels
        self.labels_encoded = self.label_encoder.fit_transform(self.labels)

        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(self.features, self.labels_encoded, test_size=0.2, random_state=42)

        # Train the model
        self.model.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test)
        print(classification_report(y_test, y_pred, labels=np.unique(self.labels_encoded), target_names=self.label_encoder.classes_))

        # Save the model and label encoder
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.label_encoder, self.label_encoder_path)

def main():
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    speaker_identification = SpeakerIdentification()

    while True:
        file_path = filedialog.askopenfilename(title="Select Audio or Video File", filetypes=[("Audio/Video Files", "*.wav *.mp4 *.mov")])
        if not file_path:
            break

        audio_file_path = speaker_identification.extract_audio(file_path)
        segments = speaker_identification.load_and_segment_audio(audio_file_path)
        speaker_identification.label_segments(segments)

        more_videos = input("Do you want to process another video? (yes/no): ")
        if more_videos.lower() != 'yes':
            break

    speaker_identification.train_and_save_model()
    print(f"Model saved at: {speaker_identification.get_model_path()}")
    print(f"Label encoder saved at: {speaker_identification.get_label_encoder_path()}")

if __name__ == "__main__":
    main()
