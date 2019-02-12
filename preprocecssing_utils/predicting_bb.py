import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input,Conv2D,Activation,MaxPooling2D,Flatten,Dense,BatchNormalization,LeakyReLU

y = np.arange(12815*4).reshape(12815,4)

test_org = pd.read_csv('test.csv')

test_names = test_org['image_name'].values

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


def predict_bb(img,model) :
    img = img.reshape(1,480,640,3)
    coord = model.predict_on_batch(img)
    return coord[0]
    
model = final_model((480,640,3))
model.load_weights("./weights/fold12/weights.best.hdf5")
model.compile(optimizer='adam',loss='mean_squared_error',metrics=['accuracy'])

for i in range(len(test_names)) :
	img = plt.imread('images/'+test_names[i])
	y[i,:] = predict_bb(img,model)
	print("{} ---> {} done!".format(i,y[i,:]))


test_org['x1'] = y[:,0]
test_org['x2'] = y[:,1]
test_org['y1'] = y[:,2]
test_org['y2'] = y[:,3]


test_org.to_csv('final_ans.csv',index=False)


	
	
