import cr_interface as cri
from core.fine_model import FineModel
from functions import optimize_full_model

train = cri.CrCollection.load().filter_by(dataset_index=0).tri_label().labeled()
test = cri.CrCollection.load().filter_by(dataset_index=1).tri_label().labeled()

models = FineModel.get_dict()
models.keys()
#dict_keys(['xception', 'mobileneta25', 'mobilenetv2a35', 'vgg16', 'resnet50v2',
#'inception_v3','inception_resnet_v2', 'densenet121', 'nasnet_mobile'])

fm = models['mobileneta25']()
optimize_full_model(train, test, fm)

