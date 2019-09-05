import abc
from abc import abstractmethod
import datetime
import glob
import os
import shutil
from typing import List
import warnings

import keras
from keras import optimizers
from keras.layers import Dense, Dropout, Flatten
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator

import core
import cr_interface as cri
import cr_analysis as cra
import keras_apps as ka


DEFAULT_POOLING = 'avg'

class FineModel(metaclass=abc.ABCMeta):
    description = 'Default fine model'
    name = 'DefaultFineModel'
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

    @staticmethod
    def get_input_shape(image_size):
        """
        Deprecated: saved as member variable

        Get input shape of conv-nets based on keras backend settings

        Returns
        tuple(n1, n2, n3)
        """
        warnings.warn('deprecated', DeprecationWarning)

        if keras.backend.image_data_format() == 'channels_first':
            return (3, ) + image_size
        else:
            return image_size + (3, )

    def _get_weight_path(self, key, directory, makedirs=True):
        if directory == None:
            path = os.path.join(core.BASE_DIR, '.fine_model_weights')
        else:
            path = directory
        path = os.path.join(path, type(self).name)
        if key:
            path = os.path.join(path, key + '.hd5')
        else:
            path = os.path.join(path, 'default.hd5')

        if makedirs:
            os.makedirs(os.path.dirname(path), exist_ok=True)

        return path

    def save_weights(self, key=None, directory=None, verbose=1):
        if self.model is None:
            warnings.warn('saving weights for unloaded model')
        model = self.get_model()
        path = self._get_weight_path(key, directory)
        if verbose:
            print('Saving weights to {}...'.format(path), end='')
        model.save_weights(path)
        if verbose:
            print('complete!')

    def load_weights(self, key=None, directory=None, verbose=1):
        model = self.get_model()
        path = self._get_weight_path(key, directory)
        if verbose:
            print('Loading weights from {}...'.format(path), end='')
        model.load_weights(path)
        if verbose:
            print('complete!')

    def get_weight_keys(self, directory=None, makedirs=True):
        if directory is None:
            path = os.path.join(core.BASE_DIR, '.fine_model_weights')
        else:
            path = directory
        path = os.path.join(path, type(self).name)
        paths = [
            os.path.splitext(os.path.basename(d))[0]
            for d in glob.glob(os.path.join(path, '*'))
        ]
        paths.sort()
        return paths

    def load_weights_by_index(self, index, directory=None):
        key = self.get_weight_keys(directory)[index]
        self.load_weights(key, directory)

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

    def get_name(self):
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
                             save_to_key=None,
                             verbose_short_name=None,
                             params={},
                             use_multiprocessing=False,
                             workers=4,
                             description=''):
        """
        Genereates a cra.Result based on predictions against test_collection.

        When save_to_key is not None, the results are saved to
            <model_name>/<save_to_key>/cr_result.json
        
        You can explore these results via cra.select_result
        """
        model = self.get_model()
        test_gen = self.get_test_generator(test_collection)
        test_gen.reset()
        if (verbose):
            print('Generating predictions for {}'.format(
                self.get_name()).center(80, '-'))
        predictions = model.predict_generator(
            test_gen,
            steps=len(test_gen),
            use_multiprocessing=use_multiprocessing,
            workers=workers,
            verbose=1)
        filenames = test_gen.filenames
        cr_codes = cri.extract_cr_codes(filenames)

        if verbose_short_name is None:
            short_name = self.get_name()
            dt = datetime.datetime.now()
            short_name += ' analyzed on {}'.format(
                dt.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            short_name = verbose_short_name

        result = cra.Result.from_predictions(predictions, cr_codes, params,
                                             short_name, description)

        if save_to_key:
            result.to_json([self.get_name(), save_to_key])

        return result

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
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

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
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

    def _get_preprocess_func(self):
        return keras.applications.mobilenet.preprocess_input


class FineMobileNetV2(FineModel):
    description = 'Custom fine model'
    name = 'mobilenetv2'
    output_shape = (224, 224)

    def _load_base_model(self):
        return keras.applications.mobilenetv2.MobileNetV2(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

    def _get_preprocess_func(self):
        return keras.applications.mobilenetv2.preprocess_input


class FineVGG16(FineModel):
    description = 'Custom fine model'
    name = 'vgg16'
    output_shape = (224, 224)
    depths = [20, 15, 11, 7, 4, 0]

    def _load_base_model(self):
        return keras.applications.vgg16.VGG16(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

    def _get_preprocess_func(self):
        return keras.applications.vgg16.preprocess_input


class FineVGG19(FineModel):
    description = 'Custom fine model'
    name = 'vgg19'
    output_shape = (224, 224)

    def _load_base_model(self):
        return keras.applications.vgg19.VGG19(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

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
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

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
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

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
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

    def _get_preprocess_func(self):
        return keras.applications.inception_resnet_v2.preprocess_input


class FineDenseNet121(FineModel):
    description = 'Custom fine model'
    name = 'densenet121'
    output_shape = (224, 224)
    depths = [428, 309, 137, 49, 7, 1, 0]

    def _load_base_model(self):
        return keras.applications.densenet.DenseNet121(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

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
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

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
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

    def _get_preprocess_func(self):
        return keras.applications.nasnet.preprocess_input


class FineResNet50V2(FineModel):
    description = 'Custom fine model'
    name = 'resnet50v2'
    output_shape = (224, 224)
    depths = [191, 142, 74, 28, 5, 1, 0]

    def _load_base_model(self):
        return ka.resnet_v2.ResNet50V2(
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

    def _get_preprocess_func(self):
        return ka.resnet_v2.preprocess_input


class FineMobileNetA25(FineModel):
    description = 'Custom fine model'
    name = 'mobileneta25'
    output_shape = (224, 224)
    depths = [88, 74, 37, 24, 11, 1, 0]

    def _load_base_model(self):
        return ka.mobilenet.MobileNet(
            alpha=0.25,
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

    def _get_preprocess_func(self):
        return ka.mobilenet.preprocess_input


class FineMobileNetV2A35(FineModel):
    description = 'Custom fine model'
    name = 'mobilenetv2a35'
    output_shape = (224, 224)
    depths = [156, 117, 55, 28, 10, 1, 0]

    def _load_base_model(self):
        return ka.mobilenet_v2.MobileNetV2(
            alpha=0.35,
            include_top=False,
            pooling=self.pooling,
            weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape),
        )

    def _get_preprocess_func(self):
        return ka.mobilenet_v2.preprocess_input
