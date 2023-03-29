from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
from dblur.utils.dataset_utils import get_img_paths


class ImageDataset(Dataset):
    """
    Class representing a dataset for image deblurring training and testing.

    ImageDataset is subclassed from torch.utils.data.Dataset. Using an instance 
    of torch.utils.data.Dataloader, we can wrap an iterable around ImageDataset
    to enable easy access to a particular sample. The sample consists of a sharp 
    image and its corresponding blur image. 

    Attributes: 
        img_dir: path to directory containing sharp and blur images.
        transform: a single transform or a list of transforms specified by an 
            instance of torchvision.transforms.Compose that are applied to each 
            sample.
        sharp_img_paths: list of sharp image paths in the directory 
            img_dir/sharp
        blur_img_paths: list of corresponding blur image paths in the directory
            img_dir/blur
    """

    def __init__(self, img_dir, transform=None):
        """Constructor for ImageDataset class.

        Args:
            img_dir: path to directory containing sharp and blur images.
                image in a sample.  
            transform: a single transform or a list of transforms specified by an 
                instance of torchvision.transforms.Compose that are applied to each 
                sample.
        """

        self.img_dir = img_dir
        self.transform = transform
        self.sharp_img_paths, self.blur_img_paths = get_img_paths(self.img_dir)

    def __len__(self):
        """Returns the number of sharp/blur images in the directory."""

        return len(self.sharp_img_paths)

    def __getitem__(self, idx):
        """Get sharp and corresponding blur image with index idx.

        Loads the sharp and blur image as instances of torch.Tensor specified
        by the paths sharp_img_paths[idx] and blur_img_paths[idx] given an
        argument idx. Having transformed and normalized the blur and sharp image,
        they are returned.

        Args:
            idx: specifies the index of sharp_img_paths and blur_image_paths that
                is loaded.
          
        Returns:
            Transformed and normalized blur and sharp image.
        """

        sharp_img = (read_image(self.sharp_img_paths[idx]).type(torch.float32)) / 255
        blur_img = (read_image(self.blur_img_paths[idx]).type(torch.float32)) / 255
        sample = {"sharp": sharp_img, "blur": blur_img}

        if self.transform:
            sample = self.transform(sample)

        sample['sharp'] = torch.clamp(sample['sharp'], 0, 1)
        sample['blur'] = torch.clamp(sample['blur'], 0, 1)
        sample['sharp'] -= 0.5
        sample['blur'] -= 0.5
        return sample["blur"], sample["sharp"]
