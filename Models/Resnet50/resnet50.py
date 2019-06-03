import keras
import numpy as np
from keras.models import Model
from keras.optimizers import SGD
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions


def get_resnet_features(x,layer_name):
    base_model = ResNet50(weights='imagenet',include_top=True)
    flatten = Model(inputs=base_model.input, outputs=base_model.get_layer(layer_name).output)
    features= flatten.predict(x)
    return features

# default resnet50 has 1000 classes
def build_model_change_dense(InputShape,classes):
    model=keras.Sequential()
    model.add(keras.layers.Dense(classes,input_shape=InputShape))
    model.add(keras.layers.Activation("softmax"))
    return model


def train_model_change_dense(train_x,train_y,learning_rate,Epoch):
    train_x=preprocess_input(train_x)
    classes=train_y[0].shape[0]
    train_x=get_resnet_features(train_x,'avg_pool')
    model=build_model_change_dense(train_x[0].shape,classes)
    sgd = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True) 
    model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=['accuracy'])
    model.fit(train_x,train_y,epochs=Epoch)
    model.save("resnet_change_dense_conv.h5")
    return model

def test_acc_change_dense(test_x,test_y):
    test_x=preprocess_input(test_x)
    test_x=get_resnet_features(test_x,'avg_pool')
    model = keras.models.load_model('resnet_change_dense_conv.h5')
    #model.summary()
    loss,accu=model.evaluate(test_x,test_y)
    return loss,accu

if __name__ == "__main__":
    img_path = 'img_31.jpg'
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    label=np.array([[1,0]])
    train_model_change_dense(x,label,0.01,1)    
    test_acc_change_dense(x,label)
