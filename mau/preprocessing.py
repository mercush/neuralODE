# partitions images and puts associates to each image name its label in a csv file.

from os import listdir
from os.path import isfile, join
from PIL import Image
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def partition_images():
    """partition images and save it somewhere else"""
    mypath = './archive/Data/images_original/classical/'
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    image_leftbound = 54
    image_rightbound = 390
    image_lowerbound = 35
    image_upperbound = 252

    for file in onlyfiles:
        im = Image.open(join(mypath, file))
        width, height = im.size
        imarray = [0,0,0,0,0,0,0,0,0,0]
        for i in range(10):
            imarray[i] = im.crop((i * ((image_rightbound - image_leftbound) // 10) + image_leftbound, image_lowerbound, (i + 1) * ((image_rightbound - image_leftbound) // 10) + image_leftbound, image_upperbound))
            imarray[i].save(join('./archive/Data/partitioned/',file[:-9]+'.'+file[-9:-3]+str(i)+".png"))

def modify_csv():
    """modifies the csv so it says png instead of wav"""
    text = open("./archive/Data/features_3_sec.csv","r")
    text = ''.join([i for i in text]) \
        .replace("wav","png")
    text = ''.join([i for i in text]) \
    .replace("classical\n","0\n")
    text = ''.join([i for i in text]) \
        .replace("jazz\n","1\n")
    text = ''.join([i for i in text]) \
        .replace("rock\n","2\n")
    x = open("./archive/Data/features_3_sec_images.csv","w")
    x.writelines(text)
    x.close()

def audio_to_spectrogram(root_dir, filename, output_folder):
    y, sr = librosa.load(join(root_dir,filename))

    n_fft = 512
    hop_length = 8
    n_mels = 128

    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    for i in range(10):
        plt.xlim([3*i,3*(i+1)])
        plt.ylim([0,10000])
        plt.axis('off')
        plt.savefig(join(output_folder,filename[:-3]+str(i)+".png"), bbox_inches='tight',pad_inches=0)

def audio_to_squeezed_spectrogram(root_dir, filename, output_folder, s=None):
    y, sr = librosa.load(join(root_dir,filename))

    n_fft = 512
    hop_length = 8
    n_mels = 128

    S = librosa.feature.melspectrogram(y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
    if s is None:
        for i in range(10):
            plt.xlim([3*i,3*(i+1)])
            plt.ylim([0,10000])
            plt.axis('off')
            plt.savefig(join(output_folder,filename[:-3]+str(i)+".png"), bbox_inches='tight',pad_inches=0)
            img = Image.open(join(output_folder,filename[:-3]+str(i)+".png"))
            img = img.resize((125,92),Image.ANTIALIAS)
            img.save(join(output_folder,filename[:-3]+str(i)+".png"))
    else:
        plt.xlim([3*s,3*(s+1)])
        plt.ylim([0,10000])
        plt.axis('off')
        plt.savefig(join(output_folder,filename[:-3]+str(s)+".png"), bbox_inches='tight',pad_inches=0)
        img = Image.open(join(output_folder,filename[:-3]+str(s)+".png"))
        img = img.resize((125,92),Image.ANTIALIAS)
        img.save(join(output_folder,filename[:-3]+str(s)+".png"))

def convert_all_audio_to_spectrogram_squeezed(root_dir,mau_spect_dir):
    onlyfiles = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
    for idx, file in enumerate(onlyfiles):
            audio_to_squeezed_spectrogram(root_dir, file, mau_spect_dir)

def convert_all_audio_to_spectrogram(root_dir,mau_spect_dir):
    onlyfiles = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
    for idx, file in enumerate(onlyfiles):
        audio_to_spectrogram(root_dir, file, mau_spect_dir)

def squeeze_images(root_dir, output_dir):
    onlyfiles = [f for f in listdir(root_dir) if isfile(join(root_dir, f))]
    print(onlyfiles)
    for idx, file in enumerate(onlyfiles):
        foo = Image.open(join(root_dir,file))
        foo = foo.resize((125,92),Image.ANTIALIAS)
        foo.save(join(output_dir,file))