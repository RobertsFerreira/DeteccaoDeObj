from pybrain.datasets.supervised import SupervisedDataSet
from pybrain.tools.shortcuts import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.customxml import NetworkWriter

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def getDataImage(path):
    # Read image
    img = Image.open(path)
    # create the pixel map
    pixels = img.load()
    global largura
    global altura
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


dataTraining = getDataImage('image/retangulo.png')
size = largura*altura*3

network = buildNetwork(size, 120, 120, 1)
dataSet = SupervisedDataSet(size, 1)

dataSet.addSample(getDataImage('image/retangulo.png'), (1))
dataSet.addSample(getDataImage('image/retangulo2.png'), (1))
dataSet.addSample(getDataImage('image/retangulo3.png'), (1))
dataSet.addSample(getDataImage('image/retangulo4.png'), (1))
dataSet.addSample(getDataImage('image/retangulo5.png'), (1))
dataSet.addSample(getDataImage('image/retangulo6.png'), (1))
dataSet.addSample(getDataImage('image/retangulo7.png'), (1))
dataSet.addSample(getDataImage('image/retangulo8.png'), (1))

trainer = BackpropTrainer(network, dataSet)
error = 1
iteration = 0
output = []
while error > 0.001:
    error = trainer.train()
    output.append(error)
    iteration += 1
    print(iteration, error)

NetworkWriter.writeToFile(network, "dataBase-Ai.xml")
