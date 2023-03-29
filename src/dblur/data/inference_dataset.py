from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
from dblur.utils.dataset_utils import get_blur_img_paths


class InferenceDataset(Dataset):
    """
    Class representing a dataset for image deblurring inference, i.e. consists of only blur images.

    InferenceDataset is subclassed from torch.utils.data.Dataset. Using an 
    instance of torch.utils.data.Dataloader, we can wrap an iterable around 
    InferenceDataset to enable easy access to a particular blurred image.

    Attributes: 
        blur_img_dir: path to directory containing blur images.
        blur_img_paths: list of blur image paths in the directory blur_img_dir.
    """

    def __init__(self, blur_img_dir):
        """Constructor for InferenceDataset class.

        Args:
            blur_img_dir: path to directory containing blur images.  
        """

        self.blur_img_paths = get_blur_img_paths(blur_img_dir)

    def __len__(self):
        """Returns the number of blur images in the directory."""

        return len(self.blur_img_paths)

    def __getitem__(self, idx):
        """Get blur image with index idx.

        Loads the blur image as instances of torch.Tensor specified by the path
        blur_img_paths[idx] given an argument idx. Then normalized blur image is
        returned.

        Args:
            idx: specifies the index of blur_img_paths that is loaded.

        Returns:
            Normalized blur image.
        """

        blur_img = (read_image(self.blur_img_paths[idx]).type(torch.float32)) / 255
        blur_img = torch.clamp(blur_img, 0, 1)
        blur_img -= 0.5

        return blur_img
