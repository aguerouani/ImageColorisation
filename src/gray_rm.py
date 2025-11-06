import os
import numpy as np
from skimage import io, color

# Load image (RGB)
path = 'Data/DIV2K_train_LR_bicubic/X2/'
dir_list = os.listdir(path)
to_remove = []
for image in dir_list:
    rgb = io.imread(path + image)

    # Convert RGB → LAB
    lab = color.rgb2lab(rgb)

    # Split channels
    L = lab[:,:,0]
    a = lab[:,:,1]
    b = lab[:,:,2]

    if np.abs(np.mean(a)) < 0.1 and np.abs(np.mean(b)) < 0.1:
        to_remove.append(image)
        print(image)

print(f"Grayscale images taht should be removed : {to_remove}")