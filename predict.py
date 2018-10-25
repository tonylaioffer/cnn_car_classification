from keras.preprocessing import image
from keras.applications import vgg16
from keras.applications import vgg19
# from keras.applications import resnet50
from keras.applications import inception_v3
# from keras.applications import densenet
# from keras.applications import mobilenetv2 # MobileNetV2
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Model, Sequential, load_model
from keras import optimizers 
from keras.callbacks import TensorBoard, ModelCheckpoint

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from tqdm import tqdm
import numpy as np
import argparse
import os
import glob

from keras import backend as K
K.set_image_dim_ordering('tf')
# K.set_learning_phase(0)


def parse_args():
    """
    parse command line parameters
    output:
        args: parsed arguments
    """
    ap = argparse.ArgumentParser()
    # ap.add_argument("-image", "--image", type=str, default='test.jpg',help="Path of test image")
    ap.add_argument("-t","--test_dir",type=str, required=True, help="(required) the test data directory")
    ap.add_argument("-c","--num_class",type=int, default=2, help="(required) number of classes to be trained")
    ap.add_argument("-m","--model_name",type=str, default='vgg19', help="model name")
    ap.add_argument("-r","--img_size",type=int, default=224, help="image width/height size")
    ap.add_argument("-w","--model_weight_name",type=str, default='vgg16.h5', help="model weight name")
    # ap.add_argument("-batch","--batch_size",type=int, default=16, help="training batch size")

    args = ap.parse_args()
    return args

def predict(args):
    # load preprocess func
    if args.model_name == 'vgg19':
        # base_model = vgg19.VGG19(include_top=False, weights=None, input_shape = (224,224,3)) # need specify input_shape
        preprocess_input = vgg19.preprocess_input

    # load test data
    test_datagen = image.ImageDataGenerator(
        preprocessing_function=preprocess_input)
    test_generator = test_datagen.flow_from_directory(
        args.test_dir,
        target_size = (args.img_size, args.img_size),
        batch_size = 1,
        shuffle = False,
        class_mode = None)  # 'categorical')

    # fnames = test_generator.filenames
    # label_map = test_generator.class_indices
    true_labels = test_generator.classes

    # load model
    model = load_model(args.model_weight_name)

    predicted_label_probs = model.predict_generator(test_generator, verbose=1)
    predicted_labels = np.argmax(predicted_label_probs, axis=1)

    # print(confusion_matrix(true_labels, predicted_labels))
    # print(classification_report(true_labels, predicted_labels))
    print("Accuracy = ", accuracy_score(true_labels, predicted_labels))


if __name__ == "__main__":
    args = parse_args()
    predict(args)
