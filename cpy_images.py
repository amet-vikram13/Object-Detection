import shutil
import pandas as pd 

names = pd.read_csv('training.csv')['image_name'].values

src = 'images/'
dst = 'training_images/'

for i in range(len(names)) :
    shutil.copy(src+names[i],dst)
    print(i)