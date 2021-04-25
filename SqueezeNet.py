from keras.layers.core import Activation,Dropout,Flatten,Dense
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers import concatenate,AveragePooling2D
from keras.regularizers import l2
from keras import backend as K
from keras.layers import Input 
from keras.models import Model

class SqueezeNet:
  def __init__(self,width=227,height=227,depth=3,classes=1000,reg=0.0002):
    self.inputShape = (height,width,depth)
    self.chanDim = -1 
    self.classes = classes
    self.reg = reg
    if K.image_data_format == "channel_first":
      self.inputShape = (depth,height,width)
      self.chanDim = 1
    self.inputs = Input(shape=self.inputShape)

  def squeeze(self,input,filters):
    squeeze1 = Conv2D(filters,(1,1),strides=(1,1),padding="valid",kernel_regularizer=l2(self.reg))(input)
    act1 = Activation("elu")(squeeze1)
    return act1

  def fire(self,input,sqFilters,exFilters):
    squeeze1 = self.squeeze(input,sqFilters)
    expand1 = Conv2D(exFilters,(1,1),strides=(1,1),padding="valid",kernel_regularizer=l2(self.reg))(squeeze1)
    act1 = Activation("elu")(expand1)
    expand3 = Conv2D(exFilters,(3,3),strides=(1,1),padding="same",kernel_regularizer=l2(self.reg))(squeeze1)
    act3 = Activation("elu")(expand3)
    output = concatenate([act1,act3],axis=self.chanDim)
    return output

  def run(self):
    conv1 = Conv2D(96,(7,7),strides=(2,2),kernel_regularizer=l2(self.reg))(self.inputs)
    act1 = Activation("elu")(conv1)
    pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(act1)
    fire2 = self.fire(pool1,16,64)
    fire3 = self.fire(fire2,16,64)
    fire4 = self.fire(fire3,32,128)
    pool4 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(fire4)
    fire5 = self.fire(pool4,32,128)
    fire6 = self.fire(fire5,48,192)
    fire7 = self.fire(fire6,48,192)
    fire8 = self.fire(fire7,64,256)
    pool8 = MaxPooling2D(pool_size=(3,3),strides=(2,2))(fire8)
    fire9 = self.fire(pool8,64,256)
    do9 = Dropout(0.5)(fire9)
    flatten = Flatten()(do9)
    dense = Dense(self.classes)(flatten)
    act11 = Activation("softmax")(dense)
    model = Model(self.inputs,act11)
    return model 
