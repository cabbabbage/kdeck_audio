import os
import io
import pandas as pd
import numpy as np
import joblib
import librosa
import webrtcvad
import tkinter as tk
from tkinter import filedialog
from pydub import AudioSegment
from pydub.silence import split_on_silence
from moviepy.editor import VideoFileClip
from create_model import SpeakerIdentification

class ScriptData:
    def __init__(self, audio_file_path):
        audio_file_path = self.extract_audio(audio_file_path)
        new_model = SpeakerIdentification(audio_file_path)
        new_model.collect_labels_and_train()
        new_model.save_model()
        model_path = new_model.get_model_path()
        label_encoder_path = new_model.get_label_encoder_path()
        self.audio_file_path = audio_file_path
        self.model = joblib.load(model_path)
        self.label_encoder = joblib.load(label_encoder_path)

    def extract_audio(self, file_path):
        if file_path.lower().endswith('.mp4'):
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

    def extract_features(self, audio_segment):
        # Save the audio segment to a temporary file
        with io.BytesIO() as buffer:
            audio_segment.export(buffer, format="wav")
            buffer.seek(0)
            audio, sample_rate = librosa.load(buffer, sr=None)
            mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
            mfccs_scaled = np.mean(mfccs.T, axis=0)
            return mfccs_scaled

    def process_full_audio(self):
        audio = AudioSegment.from_file(self.audio_file_path)
        
        # Ensure the directory for audio segments exists
        if not os.path.exists("audio_segments"):
            os.makedirs("audio_segments")

        vad = webrtcvad.Vad()
        vad.set_mode()  # Set aggressiveness mode: 0 to 3 (0 is least aggressive)

        samples_per_window = 160  # 10 ms for 16 kHz audio
        sample_rate = 16000
        audio_data = np.array(audio.get_array_of_samples())

        # Split audio into segments based on voice activity
        segments = []
        current_segment = []
        for i in range(0, len(audio_data), samples_per_window):
            window = audio_data[i:i + samples_per_window]
            if len(window) < samples_per_window:
                continue
            is_speech = vad.is_speech(window.tobytes(), sample_rate)
            if is_speech:
                current_segment.extend(window)
            else:
                if current_segment:
                    segments.append(current_segment)
                    current_segment = []

        # Handle the last segment if it exists
        if current_segment:
            segments.append(current_segment)

        data = []
        for i, segment in enumerate(segments):
            segment_audio = AudioSegment(
                data=np.array(segment).tobytes(),
                sample_width=audio.sample_width,
                frame_rate=audio.frame_rate,
                channels=audio.channels
            )

            features = self.extract_features(segment_audio)
            mfccs = features.reshape(1, -1)
            prediction = self.model.predict(mfccs)
            speaker = self.label_encoder.inverse_transform(prediction)[0]

            start_time = (i * samples_per_window) / sample_rate  # Start time in seconds
            end_time = start_time + (len(segment) / sample_rate)  # End time in seconds

            # Save audio segment to file
            segment_file = f"audio_segments/segment_{i}.wav"
            segment_audio.export(segment_file, format="wav")

            data.append({
                "Speaker": speaker,
                "Start Time": start_time,
                "End Time": end_time,
                "File Path": segment_file
            })

        # Convert to DataFrame and save to CSV
        df = pd.DataFrame(data)
        df.to_csv("full_audio_dataset.csv", index=False)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(title="Select Audio or Video File", filetypes=[("Audio/Video Files", "*.wav *.mp4")])

    if file_path:
        script_data = ScriptData(file_path)
        script_data.process_full_audio()
    else:
        print("No file selected.")
