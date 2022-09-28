import numpy as np
from PIL import Image
import pickle
import glob

dataset = []

for f in glob.glob("./torch_data/data/*.jpg"):

    img = np.array(Image.open(f))

    dataset.append(img)


print(len(dataset))
dataset = np.array(dataset)
print(dataset.shape)

with open("./kaggle_cat/data.pkl", "wb") as f:
    pickle.dump(dataset, f, -1)