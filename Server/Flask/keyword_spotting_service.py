import tensorflow as tf
import numpy as np
import librosa

MODEL_PATH = "model.h5"
NUM_SAMPLES_TO_CONSIDER = 22050  # 1 sec worth of sound

class _keyword_spotting_service:
    model = None
    _mapping = ["dataset\\down",
                "dataset\\go",
                "dataset\\left",
                "dataset\\no",
                "dataset\\off",
                "dataset\\on",
                "dataset\\right",
                "dataset\\stop",
                "dataset\\up",
                "dataset\\yes"]
    _instance = None

    def predict(self, file_path):
        # extract MFCCs
        MFCCs = self.preprocess(file_path)
        # convert 2d mfccs into 4d array
        MFCCs = MFCCs[np.newaxis, ..., np.newaxis]
        # prediction
        predictions = self.model.predict(MFCCs)
        predicted_index = np.argmax(predictions)
        predicted_keyword = self._mapping[predicted_index]
        return  predicted_keyword

    def preprocess(self, file_path, n_mfcc=13, n_fft=2048, hop_length=512):
        signal, sr = librosa.load(file_path)
        if len(signal) > NUM_SAMPLES_TO_CONSIDER:
            signal = signal[:NUM_SAMPLES_TO_CONSIDER]

        MFCCs = librosa.feature.mfcc(signal, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return MFCCs.T   # matrix transpose of MFCCs


def keyword_spotting_service():
    if _keyword_spotting_service._instance is None:
        _keyword_spotting_service._instance = _keyword_spotting_service()
        _keyword_spotting_service.model = tf.keras.models.load_model(MODEL_PATH)
    return _keyword_spotting_service._instance


if __name__ == "__main__":

    kss = keyword_spotting_service()

    keyword1 = kss.predict("test/down.wav")
    keyword2 = kss.predict("test/on.wav")

    print(f"Predicted Keywords: {keyword1}, {keyword2}")
