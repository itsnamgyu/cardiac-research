from keras_applications import vgg16, vgg19, inception_v3, resnet50, mobilenet, mobilenet_v2, inception_resnet_v2, xception, densenet, nasnet

class Application:
    @staticmethod
    def get_input_shape(image_shape):
        '''
        Get input shape of conv-nets based on keras backend settings

        Returns
        tuple(n1, n2, n3)
        '''

        if keras.backend.image_data_format() == 'channels_first':
            return (3,) + image_shape 
        else:
            return image_shape + (3,)
        
    def __init__(self, application_func, image_shape, name, codename):
        self.application_func = application_func
        self.image_shape = image_shape
        self.name = name
        self.codename = codename
        self.application = None

    def get_application(self):
        if self.application == None:
            print('loading {} application'.format(self.name))
            self.application = self.application_func(
                weights='imagenet',
                include_top=False,
                input_shape=Application.get_input_shape(self.image_shape))
        return self.application
    
    def free_application(self):
        self.application = None


applications = [
    Application(mobilenet.MobileNet, (224, 224), 'mobilenet', 'MOB'),
    Application(mobilenet_v2.MobileNetV2, (224, 224), 'mobilenetv2', 'MOB2'),
    Application(inception_resnet_v2.InceptionResNetV2, (299, 299), 'inceptionresnetv2', 'INCRES2'),
    Application(inception_v3.InceptionV3, (299, 299), 'inceptionv3', 'INC3'),
    Application(nasnet.NASNet, (224, 224), 'nasnet', 'NAS'),
    Application(resnet50.ResNet50, (224, 224), 'resnet50', 'RES'),
    Application(vgg16.VGG16, (224, 224), 'vgg16', 'VGG16'),
    Application(vgg19.VGG19, (224, 244), 'vgg19', 'VGG19'),
    Application(xception.Xception, (299, 299), 'xception',' XC'),
]

application_dict = {}
for application in applications:
    application_dict[application.name] = application

applications = application_dict
