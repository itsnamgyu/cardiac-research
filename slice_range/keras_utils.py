from keras.applications import vgg16, vgg19, inception_v3, resnet50, mobilenet, mobilenet_v2, inception_resnet_v2, xception, nasnet
import keras
from keras import optimizers
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
import warnings

import cr_interface as cri
import numpy as np


class Application:
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

    def __init__(self, model_func, preprocess_input_func, image_size, name, codename):
        self.model_func = model_func
        self.preprocess_input = preprocess_input_func
        self.image_size = image_size
        self.name = name
        self.codename = codename
        self.model = None
        if keras.backend.image_data_format() == 'channels_first':
            self.input_shape = (3,) + image_size
        else:
            self.input_shape = image_size + (3,)

    def get_model(self):
        if self.model is None:
            print('loading {} model'.format(self.name))
            self.model = self.model_func(
                weights='imagenet',
                include_top=False,
                input_shape=Application.get_input_shape(self.image_size))
        return self.model

    def free_model(self):
        self.model = None

    def get_image_data_generator(self, augment=True):
        transform_parameters = {
            'zx': 0.6,
            'zy': 0.6,
        }
        zoom_gen = ImageDataGenerator()

        def zoom(x): return self.preprocess_input(
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

    def load_top_model(self, index=0, **kwargs):
        '''
        Convenience functions for fine-tuning experiments
        '''
        if index == 0:
            def compile_model(model, lr):
                optimizer = optimizers.SGD(
                    lr=lr, decay=1e-6, momentum=0.9, nesterov=True)
                model.compile(
                    loss='categorical_crossentropy',
                    optimizer=optimizer,
                    metrics=['accuracy'])
            model = Sequential()
            model.add(Flatten(input_shape=self.get_model().output_shape[1:]))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(3, activation='softmax'))
            model.name = '{}_top'.format(self.get_model().name)
            compile_model(model, kwargs['lr'])
            return model
        else:
            raise NotImplementedError(
                'top_model not specified for index {}'.format(index))


def get_cr_codes_from_iterator(iterator, multiplier=1):
    '''
    Get cr_code of images generated from image. Does not consider shuffling.
    '''
    filenames = np.tile(np.array(iterator.filenames), multiplier)
    cr_codes = map(lambda f: cri.extract_cr_code(f), filenames)

    return cr_codes


def run_for_all_apps(f, title='', verbose=1):
    '''
    f: def f(app: ku.Application) -> ?
    '''
    results = []
    for app in apps.values():
        if verbose:
            title_part = '[{}] '.format(title) if title else ''
            print('Running {}on {}'.format(
                title_part, app.name).center(100, '-'))
        results.append(f(app))
    return results


applications = [
    Application(mobilenet.MobileNet, mobilenet.preprocess_input,
                (224, 224), 'mobilenet', 'MOB'),
    Application(mobilenet_v2.MobileNetV2, mobilenet_v2.preprocess_input,
                (224, 224), 'mobilenetv2', 'MOB2'),
    Application(inception_resnet_v2.InceptionResNetV2, inception_resnet_v2.preprocess_input,
                (299, 299), 'inceptionresnetv2', 'INCRES2'),
    Application(inception_v3.InceptionV3, inception_v3.preprocess_input,
                (299, 299), 'inceptionv3', 'INC3'),
    Application(nasnet.NASNetLarge, nasnet.preprocess_input,
                (224, 224), 'nasnet', 'NAS'),
    Application(resnet50.ResNet50, resnet50.preprocess_input,
                (224, 224), 'resnet50', 'RES'),
    Application(vgg16.VGG16, vgg16.preprocess_input,
                (224, 224), 'vgg16', 'VGG16'),
    Application(vgg19.VGG19, vgg19.preprocess_input,
                (224, 224), 'vgg19', 'VGG19'),
    Application(xception.Xception, xception.preprocess_input,
                (299, 299), 'xception', 'XC'),
]

application_dict = {}
for application in applications:
    application_dict[application.name] = application

apps = application_dict
applications = apps  # legacy
