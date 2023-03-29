from PIL import Image
import numpy as np


def display_img(img_tensor):
    """Displays image.

    Displays image represented by an instance of torch.Tensor. This instance
    is specified by the argument img_tensor.

    Args:
        img_tensor: Instance of torch.tensor which represents an image. Entries
        of tensor are between 0 and 1.
    """

    img_narray = img_tensor.to("cpu").numpy() * 255
    img_narray = img_narray.astype(np.uint8)
    img_narray = img_narray.transpose((1, 2, 0))
    img = Image.fromarray(img_narray)
    img.show()


def save_img(img_tensor, img_path):
    """Saves image.

    Saves image represented by an instance of torch.Tensor to the path specified
    by argument img_path. The instance torch.Tensor is specified by the argument
    img_tensor.

    Args:
        img_tensor: Instance of torch.tensor which represents an image. Entries
        of tensor are between 0 and 1.
    """

    img_narray = img_tensor.to("cpu").numpy() * 255
    img_narray = img_narray.astype(np.uint8)
    img_narray = img_narray.transpose((1, 2, 0))
    img = Image.fromarray(img_narray)
    img.save(img_path)
