# partitions images and puts associates to each image name its label in a csv file.

from os import listdir
from os.path import isfile, join
from PIL import Image

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
