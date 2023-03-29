import os


def get_img_paths(img_dir):
    """Returns list of sharp image paths and corresponding blur image paths.

    Returns a list of sharp image paths and a list of blur image paths in directory
    specified by img_dir. sharp_img_paths[i] and blur_img_paths[i] specifiy the
    path of the sharp and blurred version of the same image respectively.

    Args:
        img_dir: Directory containing two subdirectories: 'sharp' and 'blur'.
            'sharp' contains sharp images and 'blur' contains the corresponding
            blur images with the same name.
  
    Returns:
        A list of sharp image paths and a list of corresponding blur image paths in
        directory specified by argument img_dir
    """

    sharp_img_dir = os.path.join(img_dir, "sharp")
    blur_img_dir = os.path.join(img_dir, "blur")

    if not os.path.exists(sharp_img_dir):
        raise Exception("img_dir specified as the argument must contain a subdirectory 'sharp'")
    if not os.path.exists(blur_img_dir):
        raise Exception("img_dir specified as the argument must contain a subdirectory 'blur'")

    sharp_imgs = os.listdir(sharp_img_dir)

    sharp_img_paths = [os.path.join(sharp_img_dir, sharp_img) for sharp_img in sharp_imgs]
    blur_img_paths = [os.path.join(blur_img_dir, blur_img) for blur_img in sharp_imgs]

    return sharp_img_paths, blur_img_paths


def get_blur_img_paths(img_dir):
    """
    Returns a list of blur image paths in directory specified by img_dir.

    Args:
        img_dir: Directory containing blur images.
    
    Returns:
        A list of blur image paths in directory specified by img_dir 
    """

    blur_imgs = os.listdir(img_dir)
    blur_imgs_paths = [os.path.join(img_dir, blur_img) for blur_img in blur_imgs]
    return blur_imgs_paths
