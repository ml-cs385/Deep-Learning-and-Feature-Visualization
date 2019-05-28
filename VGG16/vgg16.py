import keras
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
from keras.optimizers import SGD
import numpy as np


# the diffenent layers' name in vgg16 can be find in 
#layername.txt
# we can get one layer's output to build our own model

def get_vgg16_feature(x,layer_name):
    base_model = VGG16(weights='imagenet',include_top=True)
    flatten = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output) # the layer after all conv
    features= flatten.predict(x)
    return features

def build_model_change_dense(InputShape,classes):    
    model=keras.Sequential()
    model.add(keras.layers.Dense(4096,input_shape=InputShape))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(4096))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(classes))
    model.add(keras.layers.Activation("softmax"))
    return model

def build_model_change_conv(InputShape,classes):
    model=keras.Sequential()
    # change the last conv
    model.add(keras.layers.Conv2D(1024,(3,3),padding="same",input_shape=InputShape))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2,2),strides=(2,2)))    
    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(4096))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(4096))
    model.add(keras.layers.Activation("relu"))
    model.add(keras.layers.Dense(classes))
    model.add(keras.layers.Activation("softmax"))
    return model


# optimize way, we can use others by add code:
# from keras.optimizers import xxx
# set para for xxx like sgd

def train_model(train_x,train_y,learning_rate,Epoch):
    train_x=preprocess_input(train_x)
    classes=train_y[0].shape[0]
    # different input with different network 
    # change the dense layer 
    #train_x=get_vgg16_feature(train_x,'flatten')
    #model=build_model_change_dense(train_x[0].shape,classes)
    # change the conv layer
    train_x=get_vgg16_feature(train_x,'block4_pool')
    model=build_model_change_conv(train_x[0].shape,classes)
    # optimize way
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True) 
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    #model.fit(train_x,train_y,epochs=Epoch,shuffle=True,validation_split=0.1) # other para you want to use set by yourself
    model.fit(train_x,train_y,epochs=Epoch)
    #model.save("cnn_model.h5")
    model.save("change_conv.h5")
    return model

def test_acc(test_x,test_y):
    test_x=preprocess_input(test_x)
    #test_x=get_vgg16_feature(test_x,'flatten')
    #model = keras.models.load_model('cnn_model.h5')
    test_x=get_vgg16_feature(test_x,'block4_pool')
    model = keras.models.load_model('change_conv.h5')
    #model.summary()
    loss,accu=model.evaluate(test_x,test_y)
    return loss,accu





if __name__ == "__main__":
    # prepare data 
    img_path = 'img_31.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    # prepare train_x and train_y
    # make sure that every vector in train_x is like: x = image.img_to_array(img) and label like [[1,0],[0,0]]
    label=np.array([[1,0]])
    train_model(x,label,0.01,1)    
    test_acc(x,label)
    





