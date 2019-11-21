import abc
import datetime
import glob
import os
import shutil
import warnings
from abc import abstractmethod
from typing import List

import keras
import numpy as np
from keras import optimizers
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, GlobalMaxPooling2D,
                          MaxPooling2D)
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib.model_pruning.python.pruning import get_weights

import core
import cr_analysis as cra
import cr_interface as cri
import keras_apps as ka

from . import paths, utils

DEFAULT_POOLING = 'avg'
TRAINED_WEIGHTS_DIR = 'cr_trained_weights'


def _get_prediction_index(percentages):
    """
    Utility function: get index of prediction from vector of prediction percentages.
    """
    max_p = 0
    max_i = 0
    for i, p in enumerate(percentages):
        if p > max_p:
            max_i = i
            max_p = p
    return max_i


class FineModel(metaclass=abc.ABCMeta):
    """A fine-tunable model that provides the following features

    - Load model with new top-model
    - Load weight based on string keys
    - Load ImageDataGenerator for K-fold validation data and test data
      - Includes default preprocessing functions
      - Includes augmentation
    
    Fields
    - `depths`: List of layer depths used for partial model freezing during
      layerwise fine tuning. `set_depth(index)` will freeze all layers starting from the
      bottom layer (0th layer) until the `depth[index]`th layer, exclusive. Hence, the larger
      the value of `depth[index]`, the more layers we will freeze. `depths[0]` should be the number
      of layers in the convolutional base, and `depths[-1]` should be 0.
    """

    description = 'Default fine model'
    name = 'default_finemodel'
    depths = []
    output_shape = (224, 224)

    @abstractmethod
    def _load_base_model(self):
        raise NotImplementedError()

    @abstractmethod
    def _get_preprocess_func(self):
        raise NotImplementedError()

    def __init__(self):
        self.model = None
        self.base_layer = None
        self.pooling = DEFAULT_POOLING

        output_shape = self.__class__.output_shape
        if keras.backend.image_data_format() == 'channels_first':
            self.input_shape = (3, ) + output_shape
        else:
            self.input_shape = output_shape + (3, )

    def get_weights_path(self, instance_key, exp_key=None):
        return paths.get_weights_path(exp_key, self.get_key(), instance_key)

    def save_weights(self,
                     instance_key,
                     exp_key=None,
                     makedirs=True,
                     verbose=1):
        if self.model is None:
            warnings.warn('saving weights for unloaded model')
        path = self.get_weights_path(instance_key, exp_key)
        if makedirs:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if verbose:
            print('Saving weights to {}...'.format(path))
        model = self.get_model()
        model.save_weights(path)
        if verbose:
            print('Save weights complete')

    def load_weights(self,
                     instance_key,
                     exp_key=None,
                     makedirs=True,
                     verbose=1):
        model = self.get_model()
        path = self.get_weights_path(instance_key, exp_key)
        if makedirs:
            os.makedirs(os.path.dirname(path), exist_ok=True)
        if verbose:
            print('Loading weights from {}...'.format(path))
        model.load_weights(path)
        if verbose:
            print('Weights loaded')

    def compile_model(self, lr=1e-4, decay=1e-6, optimizer=None):
        if optimizer is None:
            optimizer = optimizers.SGD(lr=lr,
                                       decay=decay,
                                       momentum=0.9,
                                       nesterov=True)
        self.get_model().compile(loss='categorical_crossentropy',
                                 optimizer=optimizer,
                                 metrics=['accuracy'])

    def get_image_data_generator(self, augment=True):
        transform_parameters = {'zx': 0.6, 'zy': 0.6}
        zoom_gen = ImageDataGenerator()

        def zoom(x):
            return self._get_preprocess_func()(zoom_gen.apply_transform(
                x, transform_parameters))

        if augment:
            augment_kwargs = dict(rotation_range=45, fill_mode='nearest')
        else:
            augment_kwargs = dict()

        return ImageDataGenerator(**augment_kwargs,
                                  preprocessing_function=zoom)

    def reload_model(self):
        if self.model is not None:
            # TODO: free model from memory
            self.model = None
        self._load_model()
        return self.model

    def get_model(self):
        if self.model is None:
            self._load_model()
        return self.model

    def get_key(self):
        return self.__class__.name

    def get_output_shape(self):
        return self.__class__.output_shape

    def _load_model(self):
        if self.model is None:
            print('Loading model {}... '.format(self.__class__.name), end='')
            self.model = self._load_base_model()
            top = Sequential()
            if not self.pooling:
                top.add(Flatten())
            top.add(Dense(256, activation='relu'))
            top.add(Dropout(0.5))
            top.add(Dense(3, activation='softmax'))
            top.name = '{}_top'.format(self.__class__.name)
            self.model = Model(inputs=self.model.input,
                               outputs=top(self.model.output))
            print('complete!')
        return self.model

    def get_depths(self):
        return self.__class__.depths

    def set_depth(self, index=None):
        if self.model is None:
            self._load_model()

        if index is None:
            index = -1

        for layer in self.model.layers:
            layer.trainable = True

        for layer in self.model.layers[:self.get_depths()[index]]:
            layer.trainable = False

    def get_test_generator(
            self,
            cr_collection: cri.CrCollection,
            augment=False,
            augment_factor=1,
            shuffle=False,
            batch_size=None,
            parent_dir='temp_images',
            verbose=1,
    ):
        """
        Get ImageDataGenerator for the test data, compatible with the given model.
        Note that subsequent calls to this method will invalidate the generator
        returned from previous calls, as these generators depend on images files being
        in a specific directory.
        """
        if (verbose):
            print('Loading test image generator'.center(80, '-'))

        gen = self.get_image_data_generator(augment=augment)
        path = os.path.join(parent_dir, 'test')
        # refresh directories
        os.makedirs(path, exist_ok=True)
        shutil.rmtree(path)
        os.makedirs(path, exist_ok=True)

        if (augment and augment_factor != 1):
            warnings.warn(
                'Augment is True and augment_factor != 1. Reseting factor to 1'
            )
            augment_factor = 1

        cr_collection.export_by_label(path, balancing=augment_factor)

        params = dict()
        if (batch_size):
            params['batch_size'] = batch_size

        test_gen = gen.flow_from_directory(path,
                                           target_size=self.get_output_shape(),
                                           class_mode='categorical',
                                           shuffle=shuffle,
                                           **params)

        assert (cr_collection.df.shape[0] == test_gen.n)

        return test_gen

    def reset_test_images(self, parent_dir='temp_images'):
        path = os.path.join(parent_dir, 'test')
        os.makedirs(path, exist_ok=True)
        shutil.rmtree(path)

    def get_train_val_generators(
            self,
            folds: List[cri.CrCollection],
            augment_factor=5,
            shuffle=True,
            batch_size=None,
            parent_dir='temp_images',
            verbose=1,
    ):
        """
        Get train/validation ImageDataGenerators for the given model for each fold.
        Note that subsequent calls to this method will invalidate the generators
        returned from previous calls.

        Train/validation images are BOTH BALANCED AND AUGMENTED

        :return: 
        tuple(train_gens, val_gens)

        train_gens: list of ImageDataGenerators for the train data in each fold
        val_gens: list of ImageDataGenerators for the validation data in each fold
        """
        print('Loading Train/Val ImageDataGenerators'.center(80, '-'))

        aug_gen = self.get_image_data_generator(augment=True)

        val_gens = []
        train_gens = []

        for i in range(len(folds)):
            val_dir = os.path.join(parent_dir, 'val_fold_{}'.format(i))
            train_dir = os.path.join(parent_dir, 'train_fold_{}'.format(i))

            # refresh directories
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(train_dir, exist_ok=True)
            shutil.rmtree(val_dir)
            shutil.rmtree(train_dir)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(train_dir, exist_ok=True)

            fold: cri.CrCollection
            for j, fold in enumerate(folds):
                if i == j:
                    # export validation data for fold i
                    fold.export_by_label(val_dir, balancing=augment_factor)
                else:
                    # export train data for fold i
                    fold.export_by_label(train_dir, balancing=augment_factor)

            params = dict()
            if (batch_size):
                params['batch_size'] = batch_size

            train_gens.append(
                aug_gen.flow_from_directory(
                    train_dir,
                    target_size=self.get_output_shape(),
                    class_mode='categorical',
                    **params))
            val_gens.append(
                aug_gen.flow_from_directory(
                    val_dir,
                    target_size=self.get_output_shape(),
                    class_mode='categorical',
                    **params))

            print(
                'Fold {}: {:<4} train images / {:<4} validation images'.format(
                    i + 1, train_gens[-1].n, val_gens[-1].n))

        return train_gens, val_gens

    def generate_test_result(self,
                             test_collection: cri.CrCollection,
                             verbose=1,
                             save_to_instance_key=None,
                             exp_key=None,
                             verbose_short_name=None,
                             params={},
                             use_multiprocessing=False,
                             workers=4,
                             description=''):
        """
        Genereates a cra.Result based on predictions against test_collection.

        When save_to_instance_key is not None, the results are saved to
            <model_key>/<save_to_instance_key>/cr_result.json
        
        You can explore these results via cra.select_result
        """
        model = self.get_model()
        test_gen = self.get_test_generator(test_collection)
        test_gen.reset()
        if (verbose):
            print('Generating predictions for {}'.format(
                self.get_key()).center(80, '-'))
        predictions = model.predict_generator(
            test_gen,
            steps=len(test_gen),
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            verbose=1)
        filenames = test_gen.filenames
        cr_codes = cri.extract_cr_codes(filenames)

        if verbose_short_name is None:
            short_name = self.get_key()
            dt = datetime.datetime.now()
            short_name += ' analyzed on {}'.format(
                dt.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            short_name = verbose_short_name

        result = cra.Result.from_predictions(predictions, cr_codes, params,
                                             short_name, description)
        if save_to_instance_key:
            result.save(self.get_key(), save_to_instance_key, exp_key)

        return result

    def predict(self, image: np.ndarray):
        """
        Get prediction index for given image and classifier
        """
        prediction_percentages = self.get_model().predict(np.stack([image]))[0]
        prediction_index = _get_prediction_index(prediction_percentages)
        return prediction_index

    @classmethod
    def get_list(cls):
        """
        # don't return
            FineVGG19,
            FineMobileNet,
            FineMobileNetV2,
            FineResNet50,
            FineNASNetLarge,
        """
        return [
            FineXception,
            FineMobileNetA25,
            FineMobileNetV2A35,
            FineVGG16,
            FineResNet50V2,
            FineInceptionV3,
            FineInceptionResNetV2,
            FineDenseNet121,
            FineNASNetMobile,
            BaselineModelV1,
        ]

    @classmethod
    def get_dict(cls):
        d = dict()
        for sub in cls.get_list():
            d[sub.name] = sub
        return d


class FineXception(FineModel):
    description = 'Custom fine model'
    name = 'xception'
    output_shape = (299, 299)
    depths = [133, 116, 26, 16, 7, 1, 0]

    def _load_base_model(self):
        return keras.applications.xception.Xception(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return keras.applications.xception.preprocess_input


class FineMobileNet(FineModel):
    description = 'Custom fine model'
    name = 'mobilenet'
    output_shape = (224, 224)
    depths = [88, 74, 37, 24, 11, 0]  # depreciated format

    def _load_base_model(self):
        return keras.applications.mobilenet.MobileNet(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return keras.applications.mobilenet.preprocess_input


class FineMobileNetV2(FineModel):
    description = 'Custom fine model'
    name = 'mobilenet_v2'
    output_shape = (224, 224)

    def _load_base_model(self):
        return keras.applications.mobilenetv2.MobileNetV2(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return keras.applications.mobilenetv2.preprocess_input


class FineVGG16(FineModel):
    description = 'Custom fine model'
    name = 'vgg16'
    output_shape = (224, 224)
    depths = [20, 15, 11, 7, 4, 0]

    def _load_base_model(self):
        return keras.applications.vgg16.VGG16(include_top=False,
                                              pooling=self.pooling,
                                              weights='imagenet',
                                              input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return keras.applications.vgg16.preprocess_input


class FineVGG19(FineModel):
    description = 'Custom fine model'
    name = 'vgg19'
    output_shape = (224, 224)

    def _load_base_model(self):
        return keras.applications.vgg19.VGG19(include_top=False,
                                              pooling=self.pooling,
                                              weights='imagenet',
                                              input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return keras.applications.vgg19.preprocess_input


class FineResNet50(FineModel):
    description = 'Custom fine model'
    name = 'resnet50'
    output_shape = (224, 224)
    depths = [176, 143, 81, 39, 7, 0]  # depreciated format

    def _load_base_model(self):
        return keras.applications.resnet50.ResNet50(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return keras.applications.resnet50.preprocess_input


class FineInceptionV3(FineModel):
    description = 'Custom fine model'
    name = 'inception_v3'
    output_shape = (299, 299)
    depths = [312, 229, 87, 18, 11, 1, 0]

    def _load_base_model(self):
        return keras.applications.inception_v3.InceptionV3(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return keras.applications.inception_v3.preprocess_input


class FineInceptionResNetV2(FineModel):
    description = 'Custom fine model'
    name = 'inception_resnet_v2'
    output_shape = (299, 299)
    depths = [781, 595, 267, 18, 11, 1, 0]

    def _load_base_model(self):
        return keras.applications.inception_resnet_v2.InceptionResNetV2(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return keras.applications.inception_resnet_v2.preprocess_input


class FineDenseNet121(FineModel):
    description = 'Custom fine model'
    name = 'densenet_121'
    output_shape = (224, 224)
    depths = [428, 309, 137, 49, 7, 1, 0]

    def _load_base_model(self):
        return keras.applications.densenet.DenseNet121(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return keras.applications.densenet.preprocess_input


class FineNASNetMobile(FineModel):
    description = 'Custom fine model'
    name = 'nasnet_mobile'
    output_shape = (224, 224)
    depths = [770, 533, 296, 53, 5, 1, 0]

    def _load_base_model(self):
        return keras.applications.nasnet.NASNetMobile(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return keras.applications.nasnet.preprocess_input


class FineNASNetLarge(FineModel):
    description = 'Custom fine model'
    name = 'nasnet_large'
    output_shape = (224, 224)

    def _load_base_model(self):
        return keras.applications.nasnet.NASNetLarge(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return keras.applications.nasnet.preprocess_input


class FineResNet50V2(FineModel):
    description = 'Custom fine model'
    name = 'resnet50_v2'
    output_shape = (224, 224)
    depths = [191, 142, 74, 28, 5, 1, 0]

    def _load_base_model(self):
        return ka.resnet_v2.ResNet50V2(include_top=False,
                                       pooling=self.pooling,
                                       weights='imagenet',
                                       input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return ka.resnet_v2.preprocess_input


class FineMobileNetA25(FineModel):
    description = 'Custom fine model'
    name = 'mobilenet_a25'
    output_shape = (224, 224)
    depths = [88, 74, 37, 24, 11, 1, 0]

    def _load_base_model(self):
        return ka.mobilenet.MobileNet(alpha=0.25,
                                      include_top=False,
                                      pooling=self.pooling,
                                      weights='imagenet',
                                      input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return ka.mobilenet.preprocess_input


class FineMobileNetV2A35(FineModel):
    description = 'Custom fine model'
    name = 'mobilenet_v2_a35'
    output_shape = (224, 224)
    depths = [156, 117, 55, 28, 10, 1, 0]

    def _load_base_model(self):
        return ka.mobilenet_v2.MobileNetV2(alpha=0.35,
                                           include_top=False,
                                           pooling=self.pooling,
                                           weights='imagenet',
                                           input_shape=self.input_shape)

    def _get_preprocess_func(self):
        return ka.mobilenet_v2.preprocess_input


class BaselineModel(FineModel):
    description = 'Baseline CNN model'
    name = 'generic_baseline_model'
    output_shape = (224, 224)
    depths = [0]  # no layerwise fine-tuning

    def set_depth(self, *args, **kwargs):
        warnings.warn('You should not call set_depth() on BaselineModels')
        return super(BaselineModel, self).set_depth(*args, **kwargs)

    def _get_preprocess_func(self):
        return ka.imagenet_utils.preprocess_input


class BaselineModelV1(BaselineModel):
    name = 'baseline_model_v1'

    def _load_base_model(self):
        model = Sequential()
        model.add(Conv2D(32, (3, 3), input_shape=self.input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(32, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(64, (3, 3)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        if self.pooling == 'avg':
            model.add(GlobalAveragePooling2D())
        elif self.pooling == 'max':
            model.add(GlobalMaxPooling2D())

        return model
