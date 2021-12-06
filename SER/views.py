import joblib
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
import numpy as np
import librosa
import soundfile


def result(request):
    if request.method == 'POST' and request.FILES['audio']:
        # Request audio from browser
        audio_file = request.FILES['audio']
        fs = FileSystemStorage()
        filename = fs.save(audio_file.name, audio_file)
        uploaded_file_url = fs.url(filename)

        # load our machine learning model
        model = joblib.load('SER_Model.sav')

        # feature of unknown audio file.
        feature = list(extract_feature(uploaded_file_url[1:], mfcc=True, chroma=True, mel=True, contrast=True, tonnetz=False))
        feature.extend([1, 0])
        feature = np.array(feature)

        # predict the emotion
        predicted_emotion = model.predict([feature])[0]

        # return the predicted emotion to result.html web page.
        return render(request, 'result.html', {
            'predicted_emotion': predicted_emotion
        })
    return render(request, 'result.html')


def home(request):
    return render(request, 'home.html')


# def result(request):
#
#     audio_file = request.GET['audio']
#     winsound.PlaySound(audio_file, winsound.SND_FILENAME)
#
#     return render(request, 'result.html')

def extract_feature(file_name, mfcc, chroma, mel, contrast, tonnetz):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate = sound_file.samplerate
        if chroma:
            stft = np.abs(librosa.stft(X))
        result = np.array([])
        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))
        if chroma:
            chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, chroma))
        if mel:
            mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result = np.hstack((result, mel))
        if contrast:
            contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
            result = np.hstack((result, contrast))
        if tonnetz:
            tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X), sr=sample_rate).T,axis=0)
            result = np.hstack((result, tonnetz))

    return result