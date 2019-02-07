import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

image_path       = "images/"
train_org_df     = pd.read_csv("training.csv")

train_org_name   = train_org_df["image_name"].values

image_matrix = np.arange(480*640*3).reshape(1,480,640,3)
image_matrix.fill(0)

for i in range(len(train_org_name)) :
    img = plt.imread(image_path+train_org_name[i])
    image_matrix = np.append(image_matrix,[img],axis=0)
    print("{} done!".format(i))

image_matrix = image_matrix[1:]

np.savez_compressed('./arrays',image_matrix=image_matrix)
