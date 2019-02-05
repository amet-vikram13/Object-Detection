import pandas as pd
import numpy as np
import matplotlib.image as mpimg

image_path       = "images/"
train_org_df     = pd.read_csv("training.csv")
train_org_name   = train_org_df["image_name"].values

train_org_name = train_org_name[:10]

train_org_arr = np.zeros((1,480,640,3),dtype='int')


def generating_numpy_arr(i) :
    train_org_arr = np.append(train_org_arr,[mpimg.imread(image_path+train_org_name[i])],axis=0)
    


train_org_arr = train_org_arr[1:]
np.savez_compressed('./arrays',train_org_arr=train_org_arr,test_org_arr=test_org_arr)


generating_numpy_arr()
