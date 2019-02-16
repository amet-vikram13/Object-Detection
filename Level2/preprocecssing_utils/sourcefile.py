## All libraries required
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from keras.layers import Input,Conv2D,Activation,MaxPooling2D,Flatten,Dense,BatchNormalization,LeakyReLU
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras import backend as K
%matplotlib inline

print("available gpus : {}".format(K.tensorflow_backend._get_available_gpus()))

## Global Variables
image_path = "images/"
train_org  = pd.read_csv("training.csv")
test_org   = pd.read_csv("test.csv")

def plot_test_matrix(img_test,model) :
    img    = img_test          ## numpy matrix of shape (width=480,length=640,3)
    img_test = img_test.reshape(1,480,640,3)
    coord = model.predict_on_batch(img_test)
    coord = coord[0]
    fig,ax = plt.subplots()
    ax.imshow(img)
    width = coord[1]-coord[0]
    height = coord[3]-coord[2]
    rect = patches.Rectangle((coord[0],coord[2]),width,height,linewidth=1,edgecolor='r',facecolor='none')
    ax.add_patch(rect)
    plt.show()

### Deciding the Seed ####
SEED = 42

## Shuffle random batch ####
def shuffle_batch(X,y,seed=1) :
    np.random.seed(seed)
    idx = np.random.permutation(X.shape[0])
    split = int(0.8*X.shape[0])
    train_idx,test_idx = idx[:split],idx[split:]
    X_train,X_test = X[train_idx],X[test_idx]
    y_train,y_test = y[train_idx],y[test_idx]
    return X_train,y_train,X_test,y_test

def final_model(input_shape) :
    
    X_input = Input(input_shape,name='input')
    
    X = Conv2D(32,(3,3),strides=(1,1),padding='same',name='conv2d_1')(X_input)
    X = BatchNormalization(name='bn_1')(X)
    X = LeakyReLU(name='leaky_relu_1')(X)
    X = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='max_pooling_1')(X)
    
    X = Conv2D(64,(3,3),strides=(1,1),padding='same',name='conv2d_2')(X)
    X = BatchNormalization(name='bn_2')(X)
    X = LeakyReLU(name='leaky_relu_2')(X)
    X = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='max_pooling_2')(X)
    
    X = Conv2D(128,(3,3),strides=(1,1),padding='same',name='conv2d_3')(X)
    X = BatchNormalization(name='bn_3')(X)
    X = LeakyReLU(name='leaky_relu_3')(X)
    
    X = Conv2D(64,(1,1),strides=(1,1),padding='same',name='conv2d_4')(X)
    X = BatchNormalization(name='bn_4')(X)
    X = LeakyReLU(name='leaky_relu_4')(X)
    
    X = Conv2D(128,(3,3),strides=(1,1),padding='same',name='conv2d_5')(X)
    X = BatchNormalization(name='bn_5')(X)
    X = LeakyReLU(name='leaky_relu_5')(X)
    X = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='max_pooling_3')(X)
    
    X = Conv2D(256,(3,3),strides=(1,1),padding='same',name='conv2d_6')(X)
    X = BatchNormalization(name='bn_6')(X)
    X = LeakyReLU(name='leaky_relu_6')(X)
    
    X = Conv2D(128,(1,1),strides=(1,1),padding='same',name='conv2d_7')(X)
    X = BatchNormalization(name='bn_7')(X)
    X = LeakyReLU(name='leaky_relu_7')(X)
    
    X = Conv2D(256,(3,3),strides=(1,1),padding='same',name='conv2d_8')(X)
    X = BatchNormalization(name='bn_8')(X)
    X = LeakyReLU(name='leaky_relu_8')(X)
    X = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='max_pooling_4')(X)
    
    X = Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv2d_9')(X)
    X = BatchNormalization(name='bn_9')(X)
    X = LeakyReLU(name='leaky_relu_9')(X)
    
    X = Conv2D(256,(1,1),strides=(1,1),padding='same',name='conv2d_10')(X)
    X = BatchNormalization(name='bn_10')(X)
    X = LeakyReLU(name='leaky_relu_10')(X)
    
    X = Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv2d_11')(X)
    X = BatchNormalization(name='bn_11')(X)
    X = LeakyReLU(name='leaky_relu_11')(X)
    
    X = Conv2D(256,(1,1),strides=(1,1),padding='same',name='conv2d_12')(X)
    X = BatchNormalization(name='bn_12')(X)
    X = LeakyReLU(name='leaky_relu_12')(X)
    
    X = Conv2D(512,(3,3),strides=(1,1),padding='same',name='conv2d_13')(X)
    X = BatchNormalization(name='bn_13')(X)
    X = LeakyReLU(name='leaky_relu_13')(X)
    X = MaxPooling2D(pool_size=(2,2),strides=(2,2),padding='same',name='max_pooling_5')(X)
    
    X = Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv2d_14')(X)
    X = BatchNormalization(name='bn_14')(X)
    X = LeakyReLU(name='leaky_relu_14')(X)
    
    X = Conv2D(512,(1,1),strides=(1,1),padding='same',name='conv2d_15')(X)
    X = BatchNormalization(name='bn_15')(X)
    X = LeakyReLU(name='leaky_relu_15')(X)
    
    X = Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv2d_16')(X)
    X = BatchNormalization(name='bn_16')(X)
    X = LeakyReLU(name='leaky_relu_16')(X)
    
    X = Conv2D(512,(1,1),strides=(1,1),padding='same',name='conv2d_17')(X)
    X = BatchNormalization(name='bn_17')(X)
    X = LeakyReLU(name='leaky_relu_17')(X)
    
    X = Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv2d_18')(X)
    X = BatchNormalization(name='bn_18')(X)
    X = LeakyReLU(name='leaky_relu_18')(X)
    
    X = Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv2d_19')(X)
    X = BatchNormalization(name='bn_19')(X)
    X = LeakyReLU(name='leaky_relu_19')(X)
    
    X = Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv2d_20')(X)
    X = BatchNormalization(name='bn_20')(X)
    X = LeakyReLU(name='leaky_relu_20')(X)
    
    X = Conv2D(1024,(3,3),strides=(1,1),padding='same',name='conv2d_21')(X)
    X = BatchNormalization(name='bn_21')(X)
    X = LeakyReLU(name='leaky_relu_21')(X)

    X = Conv2D(425,(1,1),strides=(1,1),padding='same',name='conv2d_22')(X)
    
    X = Flatten(name='flatten_1')(X)
    
    X = Dense(64,activation='relu',name='fc_1')(X)
    
    X = Dense(64,activation='relu',name='fc_2')(X)
    
    X = Dense(4,name='fc_3')(X)
    
    model = Model(inputs=X_input,outputs=X)
    
    return model

#### Load fold1 -- arrays_0_999 #####
loaded = np.load('./images_arrays/arrays_0_999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold1/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=8, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold2 -- arrays_1000_1999 #####
loaded = np.load('./images_arrays/arrays_1000_1999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold1/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold2/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=8, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold3 -- arrays_2000_3999 #####
loaded = np.load('./images_arrays/arrays_2000_2999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold2/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold3/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=8, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold4 -- arrays_3000_3999 #####
loaded = np.load('./images_arrays/arrays_3000_3999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold3/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold4/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=8, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold5 -- arrays_4000_4999 #####
loaded = np.load('./images_arrays/arrays_4000_4999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold4/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold5/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=8, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold6 -- arrays_5000_5999 #####
loaded = np.load('./images_arrays/arrays_5000_5999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold5/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold6/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=8, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold7 -- arrays_6000_6999 #####
loaded = np.load('./images_arrays/arrays_6000_6999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold6/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold7/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=8, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold8 -- arrays_7000_7999 #####
loaded = np.load('./images_arrays/arrays_7000_7999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold7/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold8/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=10, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold9 -- arrays_8000_8999 #####
loaded = np.load('./images_arrays/arrays_8000_8999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold8/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold9/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=10, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold10 -- arrays_9000_9999 #####
loaded = np.load('./images_arrays/arrays_9000_9999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold9/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold10/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=10, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold11 -- arrays_10000_10999 #####
loaded = np.load('./images_arrays/arrays_10000_10999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold10/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold11/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=10, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold12 -- arrays_11000_11999 #####
loaded = np.load('./images_arrays/arrays_11000_11999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold11/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold12/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=10, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold13 -- arrays_12000_12999 #####
loaded = np.load('./images_arrays/arrays_12000_12999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold12/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold13/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=10, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))

#### Load fold14 -- arrays_13000_13999 #####
loaded = np.load('./images_arrays/arrays_13000_13999.npz')
X = loaded['image_matrix']
y = train_org.drop(['image_name'],axis=1)[:1000].values
X_train,y_train,X_test,y_test = shuffle_batch(X,y,SEED)

model = final_model((480,640,3))
model.load_weights("./weights/fold13/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])
filepath="./weights/fold14/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train,y_train,validation_split=0.2, epochs=10, batch_size=8, callbacks=callbacks_list, verbose=0)

preds = model.evaluate(x = X_test,y = y_test)
print()
print ("Loss = " + str(preds[0]))
print ("Test Accuracy = " + str(preds[1]))