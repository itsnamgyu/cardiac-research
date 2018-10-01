from keras_applications import vgg16, vgg19, inception_v3, resnet50, mobilenet, mobilenet_v2, inception_resnet_v2, xception, densenet, nasnet
import keras

class Application:
    @staticmethod
    def get_input_shape(image_size):
        '''
        Get input shape of conv-nets based on keras backend settings

        Returns
        tuple(n1, n2, n3)
        '''

        if keras.backend.image_data_format() == 'channels_first':
            return (3,) + image_size 
        else:
            return image_size + (3,)
        
    def __init__(self, model_func, preprocess_input_func, image_size, name, codename):
        self.model_func = model_func
        self.preprocess_input = preprocess_input_func
        self.image_size= image_size
        self.name = name
        self.codename = codename
        self.model = None

    def get_model(self):
        if self.model == None:
            print('loading {} model'.format(self.name))
            self.model = self.model_func(
                weights='imagenet',
                include_top=False,
                input_shape=Application.get_input_shape(self.image_size))
        return self.model
    
    def free_model(self):
        self.model = None

def get_cr_codes_from_iterator(iterator, multiplier=1):
    '''
    Get cr_code of images generated from image. Does not consider shuffling.
    '''
    filenames = np.tile(np.array(iterator.filenames), multiplier)
    cr_codes = map(lambda f: cri.extract_cr_code(f), filenames)

    return cr_codes


applications = [
    Application(mobilenet.MobileNet, mobilenet.preprocess_input,
        (224, 224), 'mobilenet', 'MOB'),
    Application(mobilenet_v2.MobileNetV2, mobilenet_v2.preprocess_input,
        (224, 224), 'mobilenetv2', 'MOB2'),
    Application(inception_resnet_v2.InceptionResNetV2, inception_resnet_v2.preprocess_input,
        (299, 299), 'inceptionresnetv2', 'INCRES2'),
    Application(inception_v3.InceptionV3, inception_v3.preprocess_input,
        (299, 299), 'inceptionv3', 'INC3'),
    Application(nasnet.NASNet, nasnet.preprocess_input,
        (224, 224), 'nasnet', 'NAS'),
    Application(resnet50.ResNet50, resnet50.preprocess_input,
        (224, 224), 'resnet50', 'RES'),
    Application(vgg16.VGG16, vgg16.preprocess_input,
        (224, 224), 'vgg16', 'VGG16'),
    Application(vgg19.VGG19, vgg19.preprocess_input,
        (224, 224), 'vgg19', 'VGG19'),
    Application(xception.Xception, xception.preprocess_input,
        (299, 299), 'xception','XC'),
]

application_dict = {}
for application in applications:
    application_dict[application.name] = application

applications = application_dict
