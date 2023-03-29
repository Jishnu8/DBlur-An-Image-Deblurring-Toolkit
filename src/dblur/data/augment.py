import random
from torchvision import transforms
from torchvision.transforms import functional as TF
import numpy as np


class RandomCrop:
    """Randomly crops the sharp and blurred image in a sample. 

    Attributes:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        """Constructor for RandomCrop class

        Args:
            output_size (tuple or int): Desired output size. If int, square crop
                is made.
        """

        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        """Randomly crops the sharp and blurred image in a sample. 

        Args:
            sample: A dict consisting of two keys 'sharp' and 'blur'. The 
                values of the keys are the sharp and blur image respectively 
                represented by instances of torch.Tensor.

        Returns:
            Sample with the randomly cropped sharp and blurred image.
        """

        sharp_img, blur_img = sample['sharp'], sample['blur']
        h, w = sharp_img.shape[:2]
        new_h, new_w = self.output_size

        # Random crop
        i, j, h, w = transforms.RandomCrop.get_params(sharp_img, output_size=self.output_size)
        sharp_cropped_img = TF.crop(sharp_img, i, j, h, w)
        blur_cropped_img = TF.crop(blur_img, i, j, h, w)

        return {'sharp': sharp_cropped_img, 'blur': blur_cropped_img}


class RotationTransform:
    """Rotates the sharp and blur image in a sample by one of the given angles.

    Attributes:
        angles: List of angles in degrees.
    """

    def __init__(self, angles):
        """Constructor for RotationTransform class.

        Args:
            angles: List of angles in degrees.
        """

        self.angles = angles

    def __call__(self, sample):
        """Rotates that sharp and blur image in a sample by one of the given angles.

        Args:
            sample: A dict consisting of two keys 'sharp' and 'blur'. The 
                values of the keys are the sharp and blur image respectively 
                represented by instances of torch.Tensor.

        Returns:
            Sample with the rotated sharp and blurred image.
        """

        sharp_img, blur_img = sample['sharp'], sample['blur']
        angle = random.choice(self.angles)

        sharp_rot_img = TF.rotate(sharp_img, angle)
        blur_rot_img = TF.rotate(blur_img, angle)
        return {'sharp': sharp_rot_img, 'blur': blur_rot_img}


class RandomHorizontalFlip:
    """Flips the sharp and blur image in a sample horizontally with a specified probability.

    Attributes:
       prob: Probability of flipping sharp and blur image in a sample
           horizontally.  
    """

    def __init__(self, prob=0.5):
        """Constructor for RandomHorizontalFlip class.

        Args:
            prob: Probability of flipping sharp and blur image in a sample
                horizontally. 
        """
        self.prob = prob

    def __call__(self, sample):
        """Flips the sharp and blur image in a sample horizontally with a specified probability.

        Args:
            sample: A dict consisting of two keys 'sharp' and 'blur'. The 
                values of the keys are the sharp and blur image respectively 
                represented by instances of torch.Tensor.

        Returns:
            Sample with either the unchanged or horizontally flipped sharp and 
            blur image.
        """

        sharp_img, blur_img = sample['sharp'], sample['blur']
        if random.random() < self.prob:
            sharp_img = TF.hflip(sharp_img)
            blur_img = TF.hflip(blur_img)

        return {'sharp': sharp_img, 'blur': blur_img}


class RandomVerticalFlip:
    """Flips the sharp and blur image in a sample vertically with a specified probability.

    Attributes:
       prob: Probability of flipping sharp and blur image in a sample
       vertically. 
    """

    def __init__(self, prob=0.5):
        """Constructor for RandomVerticalFlip class.

        Args:
            prob: Probability of flipping sharp and blur image in a sample
                vertically. 
        """

        self.prob = prob

    def __call__(self, sample):
        """Flips the sharp and blur image in a sample vertically with a specified probability.

        Args:
            sample: A dict consisting of two keys 'sharp' and 'blur'. The 
                values of the keys are the sharp and blur image respectively 
                represented by instances of torch.Tensor.

        Returns:
            Sample with either the unchanged or vertically flipped sharp and 
            blur image.
        """

        sharp_img, blur_img = sample['sharp'], sample['blur']
        if random.random() < self.prob:
            sharp_img = TF.vflip(sharp_img)
            blur_img = TF.vflip(blur_img)

        return {'sharp': sharp_img, 'blur': blur_img}


class RandomSwapColorChannels:
    """Randomly swaps the color channels of the sharp and blur image in a sample.
    """

    def __call__(self, sample):
        """Randomly swaps the color channels of the sharp and blur image in a sample.

        Args:
            sample: A dict consisting of two keys 'sharp' and 'blur'. The 
                values of the keys are the sharp and blur image respectively 
                represented by instances of torch.Tensor.

        Returns:
            Sample with sharp and blur image whose color channels have been 
            randomly swapped. 
        """

        sharp_img, blur_img = sample['sharp'], sample['blur']
        channels_shuffled = np.random.permutation(3)
        sharp_img = sharp_img[channels_shuffled, :, :]
        blur_img = blur_img[channels_shuffled, :, :]
        return {'sharp': sharp_img, 'blur': blur_img}


class RandomSaturation:
    """Randomly adjust saturation of the sharp and blur image in sample with a specified probability.

    Attributes:
        saturation_factor_range: Float between 0 and 1 that decides the range
            to adjust the saturation. 0 always gives the original image. 1 
            specifies a range between a black and white image to an image whose
            saturation is increased by a factor of 2.
        prob: Probability of adjusting saturation of the sharp and blur 
            image in a sample. 
    """

    def __init__(self, saturation_factor_range=0.2, prob=0.5):
        """Constructor for RandomSaturation class.

        Args:
            saturation_factor_range: Float between 0 and 1 that decides the range
                to adjust the saturation. 0 always gives the original image. 1 
                specifies a range between a black and white image to an image whose
                saturation is increased by a factor of 2.
            prob: Probability of adjusting saturation of the sharp and blur 
                image in a sample.  
        """

        self.saturation_factor_range = saturation_factor_range
        self.prob = prob

    def __call__(self, sample):
        """Randomly adjust saturation of the sharp and blur image in sample with a specified probability.

        Args:
            sample: A dict consisting of two keys 'sharp' and 'blur'. The 
                values of the keys are the sharp and blur image respectively 
                represented by instances of torch.Tensor.

        Returns:
            Sample with the transformed sharp and blur image. 
        """

        sharp_img, blur_img = sample['sharp'], sample['blur']
        if random.random() < self.prob:
            saturation_factor = 1 + np.random.uniform(-self.saturation_factor_range, self.saturation_factor_range)
            sharp_img = TF.adjust_saturation(sharp_img, saturation_factor)
            blur_img = TF.adjust_saturation(blur_img, saturation_factor)

        return {'sharp': sharp_img, 'blur': blur_img}


class AddRandomNoise:
    """Adds gaussian noise to sharp and blur image in a sample.

    Attributes:
        mu: Mean of guassian noise.
        sigma: Standard deviation of guassian noise.
    """

    def __init__(self, mu=0, sigma=2 / 255):
        """Constructor for AddRandomNoise class.

        Args:
            mu: Mean of guassian noise.
            sigma: Standard deviation of guassian noise .
        """

        self.mu = mu
        self.sigma = sigma

    def __call__(self, sample):
        """Adds gaussian noise to sharp and blur image in a sample.

        Args:
            sample: A dict consisting of two keys 'sharp' and 'blur'. The 
                values of the keys are the sharp and blur image respectively 
                represented by instances of torch.Tensor.

        Returns:
            Sample with the transformed sharp and blur image. 
        """

        sharp_img, blur_img = sample['sharp'], sample['blur']
        noise = np.random.normal(self.mu, self.sigma, sharp_img.shape)
        sharp_img += noise
        blur_img += noise

        return {'sharp': sharp_img, 'blur': blur_img}


class RandomCutout:
    """Randomly masks out a number of square regions from the sharp and blur image in a sample.

    Attributes:
        sizes: List of sizes of the square regions that are masked out from the 
            sharp and blur image. An element is picked randomly from the list
            for a given sample. 
        num_of_squares: number of square regions that are masked out. 
        prob: Probability of masking out square regions from the sharp and blur 
            image in a sample. 
    """

    def __init__(self, sizes=[16], num_of_squares=2, prob=0.5):
        """Constructor for RandomCutout class.

        Args:
            sizes: List of sizes of the square regions that are masked out from the 
                sharp and blur image. An element is picked randomly from the list
                for a given sample. 
            num_of_squares: number of square regions that are masked out. 
            prob: Probability of masking out square regions from the sharp and blur 
                image in a sample.  
        """

        self.sizes = sizes
        self.num_of_squares = num_of_squares
        self.prob = prob

    def __call__(self, sample):
        """Randomly masks out a number of square regions from the sharp and blur image in a sample.

        Args:
            sample: A dict consisting of two keys 'sharp' and 'blur'. The 
                values of the keys are the sharp and blur image respectively 
                represented by instances of torch.Tensor.

        Returns:
            Sample with the transformed sharp and blur image. 
        """

        sharp_img, blur_img = sample['sharp'], sample['blur']
        h, w = sharp_img.shape[:2]
        if random.random() < self.prob:
            for i in range(self.num_of_squares):
                random_size_index = np.random.randint(len(self.sizes))
                output_size = (self.sizes[random_size_index], self.sizes[random_size_index])

                # Random square selection
                i, j, h, w = transforms.RandomCrop.get_params(sharp_img, output_size=output_size)
                sharp_img[:, i:i + h, j:j + w] = 0
                blur_img[:, i:i + h, j:j + w] = 0

        return {'sharp': sharp_img, 'blur': blur_img}


class RandomBlend:
    """ Adds one of the given pixel values to the sharp and blur image in a sample with a specified probability.

    Attributes:
        pixel_values: list of fixed pixel values which could be added to the 
            sharp and blur image in a sample. Pixel values should be between 0
            to 255.
        prob: Probability of adding a fixed pixel value to the sharp and blur 
            image in a sample. 
    """

    def __init__(self, pixel_values=[1, 2, 3, 4, 5], prob=0.5):
        """Constructor for RandomBlend class.

        Args:
            pixel_values: list of fixed pixel values which could be added to the 
                sharp and blur image in a sample. Pixel values should be between 0
                to 255.
            prob: Probability of adding a fixed pixel value to the sharp and blur 
                image in a sample. 
        """

        self.pixel_values = pixel_values
        self.prob = prob

    def __call__(self, sample):
        """Adds one of the given pixel values to the sharp and blur image in asample with a specified probability.

        Args:
            sample: A dict consisting of two keys 'sharp' and 'blur'. The 
                values of the keys are the sharp and blur image respectively 
                represented by instances of torch.Tensor.

        Returns:
            Sample with the transformed sharp and blur image.  
        """

        pixel_values = [i / 255 for i in self.pixel_values]
        sharp_img, blur_img = sample['sharp'], sample['blur']

        if random.random() < self.prob:
            random_pixel_value_index = np.random.randint(len(pixel_values))
            random_pixel_value = pixel_values[random_pixel_value_index]
            sharp_img += random_pixel_value
            blur_img += random_pixel_value

        return {'sharp': sharp_img, 'blur': blur_img}


class RandomBrightness:
    """Randomly adjust brightness of the sharp and blur image in sample with a specified probability.

    Attributes:
        prob: Probability of adjusting brightness of the sharp and blur 
            image in a sample. 
        brightness_factor_range: Float between 0 and 1 that decides the range
            to adjust the brightness. 0 always gives the original image. 1 
            specifies a range between a black image to a white image.

    """

    def __init__(self, brightness_factor_range=0.2, prob=0.5):
        """Constructor for RandomBrightness class.

        Args:
            prob: Probability of adjusting brightness of the sharp and blur 
            image in a sample.  
            brightness_factor_range: Float between 0 and 1 that decides the 
                range to adjust the brightness. 0 always gives the original 
                image. 1 specifies a range between a black image to a white 
                image.
        """

        self.brightness_factor_range = brightness_factor_range
        self.prob = prob

    def __call__(self, sample):
        """Randomly adjust saturation of the sharp and blur image in sample with a specified probability.

        Args:
            sample: A dict consisting of two keys 'sharp' and 'blur'. The 
                values of the keys are the sharp and blur image respectively 
                represented by instances of torch.Tensor.

        Returns:
            Sample with the transformed sharp and blur image.  
        """

        brightness_factor = 1 + np.random.uniform(-self.brightness_factor_range, self.brightness_factor_range)
        sharp_img, blur_img = sample['sharp'], sample['blur']

        if random.random() < self.prob:
            sharp_img = TF.adjust_brightness(sharp_img, brightness_factor)
            blur_img = TF.adjust_brightness(blur_img, brightness_factor)

        return {'sharp': sharp_img, 'blur': blur_img}


class RandomContrast:
    """Randomly adjust contrast of the sharp and blur image in sample with a specified probability.

    Attributes:
        prob: Probability of adjusting constrast of the sharp and blur 
            image in a sample. 
        contrast_factor_range: Float between 0 and 1 that decides the range
            to adjust the constrast. 0 always gives the original image. 1 
            specifies a range between a solid gray image and an image whose 
            constrast has been increased by a factor of 2. 

    """

    def __init__(self, contrast_factor_range=0.2, prob=0.5):
        """Constructor for RandomContrast class.

        Args:
            prob: Probability of adjusting brightness of the sharp and blur 
            image in a sample.  
            contrast_factor_range: Float between 0 and 1 that decides the range
            to adjust the constrast. 0 always gives the original image. 1 
            specifies a range between a solid gray image and an image whose 
            constrast has been increased by a factor of 2. 
        """

        self.contrast_factor_range = contrast_factor_range
        self.prob = prob

    def __call__(self, sample):
        """Randomly adjust constrast of the sharp and blur image in sample with a specified probability.

        Args:
            sample: A dict consisting of two keys 'sharp' and 'blur'. The 
                values of the keys are the sharp and blur image respectively 
                represented by instances of torch.Tensor.

        Returns:
            Sample with the transformed sharp and blur image. 
        """

        contrast_factor = 1 + np.random.uniform(-self.contrast_factor_range, self.contrast_factor_range)
        sharp_img, blur_img = sample['sharp'], sample['blur']

        if random.random() < self.prob:
            sharp_img = TF.adjust_contrast(sharp_img, contrast_factor)
            blur_img = TF.adjust_contrast(blur_img, contrast_factor)

        return {'sharp': sharp_img, 'blur': blur_img}


class RandomHue:
    """Randomly adjust hue of the sharp and blur image in sample with a specified probability.

    Attributes:
        prob: Probability of adjusting hue of the sharp and blur 
            image in a sample. 
        hue_factor_range: Float between 0 and 1 that decides the range
            to adjust the hue.

    """

    def __init__(self, hue_factor_range=0.2, prob=0.5):
        """Constructor for RandomHue class.

        Args:
            prob: Probability of adjusting brightness of the sharp and blur 
            image in a sample.  
            hue_factor_range: Float between 0 and 1 that decides the range
                to adjust the hue.
        """

        self.hue_factor_range = hue_factor_range
        self.prob = prob

    def __call__(self, sample):
        """Randomly adjust hue of the sharp and blur image in sample with a specified probability.

        Args:
            sample: A dict consisting of two keys 'sharp' and 'blur'. The 
                values of the keys are the sharp and blur image respectively 
                represented by instances of torch.Tensor.

        Returns:
            Sample with the transformed sharp and blur image. 
        """

        hue_factor = np.random.uniform(-self.hue_factor_range, self.hue_factor_range)
        sharp_img, blur_img = sample['sharp'], sample['blur']

        if random.random() < self.prob:
            sharp_img = TF.adjust_hue(sharp_img, hue_factor)
            blur_img = TF.adjust_hue(blur_img, hue_factor)

        return {'sharp': sharp_img, 'blur': blur_img}
