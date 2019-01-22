import abc
import warnings
from abc import abstractmethod
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, Dropout


class FineModel(metaclass=abc.ABCMeta):
    description = 'Default fine model'
    name = 'FineModel'
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

    def compile_model(self, lr=1e-4, decay=1e-6):
        optimizer = optimizers.SGD(
            lr=lr, decay=decay, momentum=0.9, nesterov=True)
        optimizer = optimizers.Adam()
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
            self.model = Model(inputs=self.model.input, outputs=top(self.model.output))
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
        return [
            FineXception,
            FineMobileNet,
            FineMobileNetV2,
            FineVGG16,
            FineVGG19,
            FineResNet50,
            FineInceptionV3,
            FineInceptionResNetV2,
            FineDenseNet121,
            FineNASNetMobile,
            FineNASNetLarge,
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
    output_shape  = (299, 299)

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
    output_shape  = (224, 224)
    depths = [ 88, 74, 37, 24, 11, 0 ]

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
    output_shape  = (224, 224)

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
    output_shape  = (224, 224)

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
    output_shape  = (224, 224)

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
    output_shape  = (224, 224)
    depths = [ 176, 143, 81, 39, 7, 0 ]


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
    output_shape  = (299, 299)

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
    output_shape  = (299, 299)

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
    output_shape  = (224, 224)

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
    output_shape  = (224, 224)

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
    output_shape  = (224, 224)

    def _load_base_model(self):
        return keras.applications.nasnet.NASNetLarge(
            include_top=False, pooling='avg', weights='imagenet',
            input_shape=FineModel.get_input_shape(self.__class__.output_shape)
        )

    def _get_preprocess_func(self):
        return keras.applications.nasnet.preprocess_input
