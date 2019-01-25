import cr_interface as cri
from core.fine_model import FineModel
from functions import optimize_all_models

train = cri.CrCollection.load().filter_by(dataset_index=0).tri_label().labeled()
test = cri.CrCollection.load().filter_by(dataset_index=1).tri_label().labeled()

optimize_all_models(train, test)

