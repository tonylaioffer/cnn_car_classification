from keras.preprocessing import image
from keras.applications import vgg16
from keras.applications import vgg19
from keras.applications import resnet50
from keras.applications import inception_v3
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten
from keras.models import Model,Sequential
from keras import optimizers
from keras.callbacks import TensorBoard, ModelCheckpoint
import argparse
from time import time

from skimage import exposure, color

from keras import backend as K
K.set_image_dim_ordering('tf')
# K.set_learning_phase(1)

#################################################################################################################################
#################################################################################################################################
# Following steps are required to fine-tune the model
#
#  1. Specify the path to training and testing data, along with number of classes and image size.
#  2. Do some random image transformations to increase the number of training samples and load the training and testing data
#  3. Create VGG16 network graph(without top) and load imageNet pre-trained weights
#  4. Add the top based on number of classes we have to the network created in step-3
#  5. Specify the optimizer, loss etc and start the training
##################################################################################################################################
##################################################################################################################################

def parse_args():
    """
    parse command line parameters
    return:
        args: parsed commandline arguments
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-t","--train_dir",type=str, required=True,help="(required) the train data directory")
    ap.add_argument("-v","--val_dir",type=str, required=True,help="(required) the validation data directory")
    ap.add_argument("-n","--num_class",type=int, default=2,help="(required) number of classes to be trained")
    ap.add_argument("-r","--img_size",type=int, default=224, help="image width/height size")
    ap.add_argument("-m","--model_name",type=str, default='vgg19', help="model name")
    ap.add_argument("-s","--suffix",type=str, default='laioffer', help="suffix for model name model name")
    ap.add_argument("-b","--batch_size",type=int, default=16, help="training batch size")
    ap.add_argument("-e","--epochs", type=int, default=30, help="training epochs")
    # ap.add_argument("-g","--epochs", type=int, default=20, help="training epochs")

    args = ap.parse_args()
    return args

def init_model(args):
    """
    initialize cnn model and training and validation data generator
    parms:
        args: parsed commandline arguments
    return:
        model: initialized model
        train_generator: training data generator
        validation_generator: validation data generator
    """
    batch_size = args.batch_size

    print('loading the model and the pre-trained weights...')

    # load base model
    if args.model_name == 'vgg19':
        base_model = vgg19.VGG19(include_top=False, weights='imagenet', input_shape = (224,224,3)) # need specify input_shape
        # this preprocess_input is the default preprocess func for given network, you can change it or implement your own 
        preprocess_input = vgg19.preprocess_input

    # initalize training image data generator
    # you can also specify data augmentation here
    train_datagen = image.ImageDataGenerator(
        # width_shift_range=0.1,
        # height_shift_range=0.1,
        # samplewise_center=True,
        # samplewise_std_normalization=True,
        # rescale=1./255,
        preprocessing_function=preprocess_input, # preprocess_input,
        # rotation_range=30,
        # shear_range=0.1,
        # zoom_range=0.1,
        # vertical_flip=True,
        horizontal_flip=True
        )

    # initalize validation image data generator
    # you can also specify data augmentation here
    validation_datagen = image.ImageDataGenerator(
        # samplewise_center=True,
        # samplewise_std_normalization=True
        # rescale=1./255
        preprocessing_function=preprocess_input # preprocess_input
        )

    train_generator = train_datagen.flow_from_directory(
        args.train_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = validation_datagen.flow_from_directory(
        args.val_dir,
        target_size=(args.img_size, args.img_size),
        batch_size=batch_size,
        class_mode='categorical')

    # fix base_model layers
    for layer in base_model.layers:
        layer.trainable = False

    # added some customized layers for your own data
    x = base_model.output
    if args.model_name == 'vgg19':
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(256, activation='relu', name='fc2-pretrain')(x)
        x = Dropout(0.3, name='dropout')(x)

    # added softmax layer
    predictions = Dense(args.num_class, activation='softmax', name='predictions')(x)

    model = Model(inputs=base_model.input, outputs=predictions)
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(), metrics=["accuracy"])

    return model, train_generator, validation_generator


def train(model, train_generator, validation_generator, args):
    """
    train the model
    parms:
        model: initialized model
        train_generator: training data generator
        validation_generator: validation data generator
        args: parsed command line arguments
    return:
    """
    # define number of steps/iterators per epoch
    stepsPerEpoch = train_generator.samples / args.batch_size
    validationSteps= validation_generator.samples / args.batch_size

    # save the snapshot of the model to local drive
    pretrain_model_name = 'pretrained_{}_{}_{}_{}.h5'.format(args.model_name, args.num_class, args.epochs, args.suffix)
    # visualize the training process
    tensorboard = TensorBoard(log_dir="logs/{}_pretrain_{}".format(args.model_name, time()), histogram_freq=0, write_graph=True)
    checkpoint = ModelCheckpoint(pretrain_model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    callbacks_list = [checkpoint, tensorboard]

    model.fit_generator(
        train_generator,
        steps_per_epoch=stepsPerEpoch,
        epochs=args.epochs,
        callbacks = callbacks_list,
        validation_data = validation_generator,
        validation_steps=validationSteps)


def fine_tune(model, train_generator, validation_generator, args):
    """
    fine tune the model
    parms:
        model: initialized model
        train_generator: training data generator
        validation_generator: validation data generator
        args: parsed command line arguments
    return:
    """
    # for specific architectures, define number of trainable layers
    if args.model_name == 'vgg19':
        trainable_layers = 6

    for layer in model.layers[:-1*trainable_layers]:
        layer.trainable = False

    for layer in model.layers[-1*trainable_layers:]:
        layer.trainable = True

    finetune_model_name = 'finetuned_{}_{}_{}_{}.h5'.format(args.model_name, args.num_class, args.epochs, args.suffix)
    tensorboard = TensorBoard(log_dir="logs/{}_finetune_{}".format(args.model_name, time()), histogram_freq=0, write_graph=True)
    checkpoint = ModelCheckpoint(finetune_model_name, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    callbacks_list = [checkpoint, tensorboard]

    model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=0.0001, momentum=0.9),metrics=["accuracy"])

    stepsPerEpoch = train_generator.samples / args.batch_size
    validationSteps= validation_generator.samples / args.batch_size
    model.fit_generator(
        train_generator,
        steps_per_epoch=stepsPerEpoch,
        epochs=args.epochs + 50,
        callbacks = callbacks_list,
        validation_data = validation_generator,
        validation_steps=validationSteps)


if __name__ == "__main__":
    args = parse_args()
    model, train_generator, validation_generator = init_model(args)
    train(model, train_generator, validation_generator, args)
    fine_tune(model, train_generator, validation_generator, args)
