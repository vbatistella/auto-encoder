from model import AutoEncoder
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import random
import os

TRAIN = True

def show_image(data):
    plt.imshow(data)
    plt.show()

def get_files_from_dir(dir):
    return os.listdir(dir)

def get_image(file):
    return Image.open(file)

def get_image_data(file):
    return np.asarray(get_image(file))

def resize(im, h, w):
    return im.resize((h, w))

def print_dataset_info(dir):
    a = get_files_from_dir(dir)
    file = pick_random_from_set(dir)
    data = get_image_data(dir+"/"+file)
    print(f"total images   : {len(a)}")
    print(f"image height   : {len(data)}")
    print(f"image width    : {len(data[0])}")
    print(f"image channels : {len(data[0][0])}")

def pick_random_from_set(path):
    files = get_files_from_dir(path)
    i = random.randint(0, len(files))
    return files[i]

def preproc(im):
    im = resize(im, 64, 64)
    im = np.asarray(im)/255
    return im

def main():
    dir = "faces_dataset"
    print_dataset_info(dir)

    ae = AutoEncoder((64, 64, 3))
    ae.get_model()

    if TRAIN:
        X = []
        for f in get_files_from_dir(dir):
            im = get_image(dir+"/"+f)
            im = preproc(im)
            X.append(im)
        X = np.asarray(X)

        model = ae.train(X, epochs=1000)

        f = pick_random_from_set(dir)
        im = get_image(dir+"/"+f)
        im = np.asarray([preproc(im)])

        im2 = model.predict(im)
        show_image(im[0])
        show_image(im2[0])

if __name__ == "__main__":
    main()