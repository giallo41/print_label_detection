import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input, GlobalAveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras import regularizers
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard

class MobileNet:
    def __init__(self,
                 #input_shape,
                 train_data,
                 valid_data,
                 lr=2e-4,
                 epochs=200, 
                 batch_size=64, 
                 model_save_dir=f"./data/models/mobilnet/mobilenet.h5",
                 tf_dir="./data/models/tbhist"):
        
        self.x_train, self.y_train = train_data
        self.valid_data = valid_data
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_save_dir = model_save_dir
        self.tf_dir = tf_dir
        self.lr = lr
        self.input_shape = self.x_train.shape[1:] #input_shape
        self.optimizers = Adam(lr=self.lr)
        self._build_model()
        self._train()
    
    def _build_model(self):
        model=Sequential()
        base_model = MobileNetV2(weights='imagenet',
                                 include_top=False,
                                 input_tensor=Input(shape=self.input_shape))
        model.add(base_model)
        model.add(Conv2D(32, (1,1), 
                         activation="relu",
                         kernel_initializer="he_normal",
                         padding="same",
                         name="HeatLayer",))
        model.add(AveragePooling2D(pool_size=(7, 7)))
        model.add(Flatten(name="flatten"))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid", name = 'Output'))
        
        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(loss = 'binary_crossentropy', optimizer = self.optimizers, metrics = ['accuracy'])
        
        self.early_stopping = EarlyStopping(monitor='val_loss', 
                                       verbose = 0, 
                                       patience=10, 
                                       min_delta = .00075)
        
        self.tensorboard_history = TensorBoard(log_dir=self.tf_dir, 
                                               histogram_freq=0, 
                                               write_graph=True, 
                                               write_images=True)
        
        self.model_checkpoint = ModelCheckpoint(self.model_save_dir, 
                                           verbose = 1, 
                                           save_best_only=True,
                                           monitor = 'val_loss')
        self.lr_plat = ReduceLROnPlateau(patience = 5, mode = 'min')

        self.model = model
        
    def _train(self):
        
        augmentation =ImageDataGenerator(rotation_range = 20, 
                                         width_shift_range = .2, 
                                         height_shift_range = .2, 
                                         horizontal_flip = True, 
                                         shear_range = .15, 
                                         fill_mode = 'nearest', 
                                         zoom_range = .15)
        
        augmentation.fit(self.x_train)
        
        self.history = self.model.fit_generator(augmentation.flow(self.x_train, self.y_train, batch_size = self.batch_size),
                                                 epochs = 200, #model.epochs, 
                                                 callbacks = [self.early_stopping, 
                                                              self.model_checkpoint, 
                                                              self.lr_plat,
                                                              self.tensorboard_history],
                                                 validation_data = self.valid_data, #(x_valid, y_valid),
                                                 verbose= 1)
        
        
def make_bbox(input_shape):

    vgg = MobileNetV2(weights="imagenet", include_top=False,
            input_tensor=Input(shape=(224, 224, 3)))
    
    vgg.trainable = False
    # flatten the max-pooling output of VGG
    flatten = vgg.output
    flatten = Flatten()(flatten)
    # construct a fully-connected layer header to output the predicted
    # bounding box coordinates
    bboxHead = Dense(128, activation="relu")(flatten)
    bboxHead = Dropout(0.5)(bboxHead)
#    bboxHead = Dense(32, activation="relu")(bboxHead)
    bboxHead = Dense(4, activation="sigmoid")(bboxHead)
    # construct the model we will fine-tune for bounding box regression
    model = Model(inputs=vgg.input, outputs=bboxHead)
    
    return model