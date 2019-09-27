import matplotlib.cm as cm
import numpy as np

from .fine_model import FineModel
import cr_interface as cri
from vis.visualization import visualize_cam, overlay


def generate_gradcam(fm: FineModel,
                     image: np.ndarray,
                     penultimate_layer_idx: int = None,
                     color=True,
                     class_index: int = None):
    """
    Generate gradcam image for given image, classification and class. If `class_index`
    is unspecified, we run the image through the classifier and use the predicted class.

    Penultimate layer does not need to be manually specified for our current models.
    """
    if not class_index:
        class_index = fm.predict(image)

    preprocess = fm._get_preprocess_func()
    processed = preprocess(image)

    gradcam = visualize_cam(fm.get_model(),
                            fm.get_depths()[0],
                            filter_indices=class_index,
                            seed_input=processed,
                            penultimate_layer_idx=penultimate_layer_idx,
                            backprop_modifier=None)
    if color:
        gradcam = np.uint8(cm.jet(gradcam)[..., :3] * 255)

    return gradcam


def overlay_gradcam(fm: FineModel,
                    image: np.ndarray,
                    penultimate_layer_idx: int = None,
                    class_index: int = None):
    """
    Penultimate layer does not need to be manually specified for our current models.
    """
    gradcam = generate_gradcam(fm,
                               image,
                               penultimate_layer_idx,
                               color=True,
                               class_index=class_index)
    return overlay(gradcam, image)
