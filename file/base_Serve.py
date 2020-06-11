from pybrain.tools.shortcuts import buildNetwork
from pybrain.tools.customxml import NetworkReader

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

net = NetworkReader.readFrom("dataBase-Ai.xml")


def getDataImage(path):
    # Read image
    img = Image.open(path)
    # create the pixel map
    pixels = img.load()
    largura = img.size[0]
    altura = img.size[1]
    data = []
    pixel = []
    for i in range(largura):
        for j in range(altura):
            pixel = pixels[i, j]
            data.append(pixel[0])
            data.append(pixel[1])
            data.append(pixel[2])
    exif_data = img._getexif()
    exif_data
    return data


name = ['retangulo.png', 'retangulo3.png',
        'retangulo6.png', 'retangulo8.png']

for i in range(len(name)):
    path = "image/teste/" + name[i]
    print(path)
    print(net.activate(getDataImage(path)))
