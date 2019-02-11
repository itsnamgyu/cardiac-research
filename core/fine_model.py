import os
import abc
import warnings
from abc import abstractmethod
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
import keras_apps as ka

import core


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

    @staticmethod
    def get_input_shape(image_size):
        '''
        Deprecated: saved as member variable

        Get input shape of conv-nets based on keras backend settings

        Returns
        tuple(n1, n2, n3)
        '''
        warnings.warn("deprecated", DeprecationWarning)

        if keras.backend.image_data_format() == 'channels_first':
            return (3,) + image_size
        else:
            return image_size + (3,)

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

    def compile_model(self, lr=1e-4, decay=1e-6, optimizer=None):
        if optimizer is None:
            optimizer = optimizers.SGD(
                lr=lr, decay=decay, momentum=0.9, nesterov=True)
        self.model.compile(
            loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

    def get_image_data_generator(self, augment=True):
        transform_parameters = {
            'zx': 0.6,
            'zy': 0.6,
        }
        zoom_gen = ImageDataGenerator()

        def zoom(x): return self._get_preprocess_func()(
            zoom_gen.apply_transform(x, transform_parameters))

        if augment:
            augment_kwargs = dict(
                rotation_range=45,
                fill_mode='nearest'
            )
        else:
            augment_kwargs = dict()

        return ImageDataGenerator(
            **augment_kwargs,
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

    @classmethod
    def get_list(cls):
        '''
        # don't return
            FineVGG19,
            FineMobileNet,
            FineMobileNetV2,
            FineResNet50,
            FineNASNetLarge,
        '''
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
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
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
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
        )

    def _get_preprocess_func(self):
        return keras.applications.mobilenet.preprocess_input


class FineMobileNetV2(FineModel):
    description = 'Custom fine model'
    name = 'mobilenetv2'
    output_shape = (224, 224)

    def _load_base_model(self):
        return keras.applications.mobilenetv2.MobileNetV2(
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
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
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
        )

    def _get_preprocess_func(self):
        return keras.applications.vgg16.preprocess_input


class FineVGG19(FineModel):
    description = 'Custom fine model'
    name = 'vgg19'
    output_shape = (224, 224)

    def _load_base_model(self):
        return keras.applications.vgg19.VGG19(
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
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
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
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
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
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
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
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
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
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
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
        )

    def _get_preprocess_func(self):
        return keras.applications.nasnet.preprocess_input


class FineNASNetLarge(FineModel):
    description = 'Custom fine model'
    name = 'nasnet_large'
    output_shape = (224, 224)

    def _load_base_model(self):
        return keras.applications.nasnet.NASNetLarge(
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
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
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
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
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
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
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
        )

    def _get_preprocess_func(self):
        return ka.mobilenet_v2.preprocess_input
