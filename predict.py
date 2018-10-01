from keras.preprocessing import image
from keras.applications import vgg16
# from keras.applications import vgg19
# from keras.applications import resnet50
from keras.applications import inception_v3
# from keras.applications import densenet
# from keras.applications import mobilenetv2 # MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Model,Sequential
from keras import optimizers 
from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import numpy as np
import argparse
import os
import glob

from keras import backend as K
K.set_image_dim_ordering('tf')
# K.set_learning_phase(0)

def parse_args():
    ap = argparse.ArgumentParser()
    # ap.add_argument("-image", "--image", type=str, default='test.jpg',help="Path of test image")
    ap.add_argument("-test","--test_dir",type=str, required=True, help="(required) the test data directory")
    ap.add_argument("-class","--num_class",type=int, default=2, help="(required) number of classes to be trained")
    ap.add_argument("-model","--model_name",type=str, default='vgg16', help="model name")
    ap.add_argument("-res","--img_size",type=int, default=224, help="image width/height size")
    ap.add_argument("-weight","--model_weight_name",type=str, default='vgg16.h5', help="model weight name")
    # ap.add_argument("-batch","--batch_size",type=int, default=16, help="training batch size")

    args = ap.parse_args()
    return args

def predict(args):
    # load base model
    if args.model_name == 'vgg16':
        base_model = vgg16.VGG16(include_top=False, weights=None, input_shape = (224,224,3)) # need specify input_shape
        preprocess_input = inception_v3.preprocess_input  # some bug on previous keras


    test_datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input) # rescale=1./255)
    test_generator = test_datagen.flow_from_directory(
        args.test_dir,
        # color_mode='grayscale', # 'rgb'
        target_size=(args.img_size, args.img_size),
        batch_size=1,
        shuffle = False,
        class_mode='categorical')

    fnames = test_generator.filenames
    label_map = test_generator.class_indices
    test_true_labels = test_generator.classes


    # add top
    x = base_model.output
    if args.model_name == 'vgg16':
        x = Flatten(name='flatten')(x)
        x = Dense(512, activation='relu', name='fc1-pretrain')(x)
        x = Dense(256, activation='relu', name='fc2-pretrain')(x)
        x = Dropout(0.5, name='dropout')(x)


    predictions = Dense(args.num_class, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    finetune_model_name = args.model_weight_name
    model.load_weights(finetune_model_name)

    folders = os.listdir(args.test_dir)
    predicted_labels = []
    true_labels = []
    for folder in folders:
        files = glob.glob(os.path.join(args.test_dir, folder) + "/*.jpg")
        print("working on folder: {}".format(folder))
        for idx in tqdm(range(len(files))):
            file = files[idx]
            img = load_img(file, target_size=(224, 224))
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = preprocess_input(img)
            # img = img / 255.0
            prob = model.predict(img)
            prob_label = prob.argmax()
            true_labels.append(true_label)


    print(confusion_matrix(true_labels, predicted_labels))
    print(classification_report(true_labels, predicted_labels))

if __name__ == "__main__":
    args = parse_args()
    predict(args)