from flask import Flask, render_template
import pickle
import numpy as np
from keras.models import load_model
import h5py
import sys
import librosa
import random
import math
app = Flask(__name__)


prd_model = load_model('models/RNN_model.hdf5')
prd_model2 = load_model('models/RNN_bolly_model1.hdf5')
print("MODEL INSTANCE HERE", file=sys.stderr)
print(prd_model, file=sys.stderr)

genres = np.array(['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz',
                   'metal', 'pop', 'reggae', 'rock'])
genres2 = np.array(['bollypop', 'carnatic', 'ghazal', 'semiclassical', 'sufi'])


def predict(model, X, genres):
    print("INSIDE PREDICTION", file=sys.stderr)
    print(X, file=sys.stderr)
    X = X[np.newaxis, ...]
    prediction = model.predict(X)
    predicted_index = np.argmax(prediction, axis=1)
    print("Outcome Index: {}".format(predicted_index))
    print("Outcome genre: {}".format(genres[predicted_index]))
    return genres[predicted_index]


def save(genres, prd_model, path, n_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):

    sample_rate = 22050
    duration = 30
    samples_per_track = sample_rate * duration
    data = {

        "mfcc": [],

    }

    num_samples_per_segment = int(samples_per_track/num_segments)
    expected_mfccv_per_segments = math.ceil(num_samples_per_segment/hop_length)
    signal, sr = librosa.load(path, sr=sample_rate)

    for s in range(num_segments):

        start = num_samples_per_segment * s
        finish = start + num_samples_per_segment
        mfcc = librosa.feature.mfcc(
            signal[start:finish], sr=sr, n_fft=n_fft, n_mfcc=n_mfcc, hop_length=hop_length)
        mfcc = mfcc.T
        if len(mfcc) == expected_mfccv_per_segments:
            data["mfcc"].append(mfcc.tolist())
            print("{}, segment:{}".format(path, s))

    genre = predict(prd_model, np.array(data["mfcc"])[1], genres)

    return genre


@app.route("/")
def home():

    musicmap = ["clsong.wav",
                "cosong.wav", "hsong.wav", "msong.wav", "song1.wav", "song2.wav", "song3.wav"]
    musicmap2 = ["bsong.wav",
                 "susong.wav", "semsong.wav", "msong.wav", "ghsong.wav", "carsong.wav"]

    print(musicmap)
    pictureList = ["https://i1.wp.com/cornellsun.com/wp-content/uploads/2019/09/a3825990458_10.jpg?w=1200", "https://substreammagazine.com/wp-content/uploads/2020/07/Valley-2020-scaled.jpg", "https://i.pinimg.com/originals/42/c4/1e/42c41e228d7bc5cf4496f787fdc2b23b.jpg",
                   "https://images-na.ssl-images-amazon.com/images/I/513VUhBNJzL.jpg", "https://images-na.ssl-images-amazon.com/images/I/513VUhBNJzL.jpg",  "https://www.gratefulweb.com/sites/default/files/images/articles/DSC_5675.jpg", "https://66.media.tumblr.com/18f3a10f6cd8cb849138f77d8a3f09a1/tumblr_inline_ptzcmmfz8B1s9on4d_540.jpg"]

    pictureList2 = ["https://rollingstoneindia.com/wp-content/uploads/2018/05/1-RSCover-MAY-18-lower-res-480x628.jpg",
                    "https://static.toiimg.com/thumb/msid-63414571,width-800,height-600,resizemode-75,imgsize-25766/63414571.jpg",
                    "https://upload.wikimedia.org/wikipedia/commons/a/a0/Prateek_Kuhad_New.jpg",
                    "https://englishtribuneimages.blob.core.windows.net/gallary-content/2020/7/2020_7$largeimg_741540610.jpg",
                    "https://static.toiimg.com/photo/78868526/78868526.jpg?v=3",
                    "https://rollingstoneindia.com/wp-content/uploads/2020/09/Armaan-3-960x1243.jpg"]
    return render_template("index.html", musicmaps=zip(musicmap, pictureList), musicmaps2=zip(musicmap2, pictureList2))


@app.route("/<name>")
def mus(name):
    path = f"static/music/{name}"
    genre = save(genres, prd_model, path, num_segments=5)
    print("The Value Of Genre Is: {}".format(genre), file=sys.stderr)
    return render_template("home.html", music=name, genre=genre[0].title())


@app.route("/hindi/<name>")
def hmus(name):
    path = f"static/music/{name}"
    genre = save(genres2, prd_model2, path, num_segments=5)
    print("The Value NAME IS: {}".format(name), file=sys.stderr)
    return render_template("home2.html", music=name, genre=genre[0].title())


if __name__ == "__main__":
    app.run(debug=True)
