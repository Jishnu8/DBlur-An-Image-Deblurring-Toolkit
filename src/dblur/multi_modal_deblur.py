from piq import brisque
import torch
import os
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from torchvision.io import read_image
from dblur.data.inference_dataset import InferenceDataset
from dblur.testers.stack_dmphn import StackDMPHNTester
from dblur.testers.mscnn import MSCNNTester
from dblur.testers.srn import SRNTester
from dblur.testers.textdbn import TextDBNTester
from dblur.testers.base_tester import BaseTester
from dblur.utils.img_utils import save_img, display_img


def multi_modal_deblur_imgs(models, model_names, model_paths, blur_img_dir, sharp_img_dir, is_checkpoint, batch_size=2,
                            window_slicing=False, window_size=256, overlap_size=0):
    """Deblurs images given multiple pretrained models by averaging outputs.

    Deblurs all images in a specified directory given multiple pretrained
    models. For a given image, the outputs of all the models given are averaged
    to give us the final deblurred image. This takes advantage of the strengths
    and weaknesses of each model.

    Args:
        models: list of instances of different models (e.g Restormer, MSCNN, etc.)
        model_names: list of the model names.
        model_paths: list of the paths of the checkpoints (model, optimizer and
            lr_scheduler) or just the list of the paths of the models. 
        blur_img_dir: directory which contains all the blur images. 
        sharp_img_dir: directory in which all the deblurred images shall be
            stored.
        batch_size: batch size in which to perform deblurring of images. 
        is_checkpoint: list specifying if path specified in model_path[i] is a 
            checkpoint (model, optimizer and lr_scheduler) or just a model. 
        batch_size: batch size in which images are deblurred. 
        window_slicing: if True, image is sliced into overlapping windows of
            size window_size and overlap overlap_size. Deblurring is then 
            performed in each window and merged back appropriately. This 
            takes into account the fact that training of deblurring models
            are usually performed on small image patches.
        window_size: size of the window in pixels that the image is sliced 
            into.
        overlap_size: size of overlap between two windows in pixels.  
    """

    if len(models) != len(model_names) or len(models) != len(model_paths) or len(models) != len(is_checkpoint):
        raise Exception("Length of models, model_names, model_paths and is_checkpoint have to be the same.")

    for i in range(len(model_names)):
        if (model_names[i] != "TextDBN" and model_names[i] != "Restormer" and model_names[i] != "NAFNet" and
                model_names[i] != "FNAFNet"
                and model_names[i] != "StackDMPHN" and model_names[i] != "SRN" and model_names[i] != "MSCNN"):
            raise Exception(model_names[i], "is either not valid or not supported. Currently the following models are \
                            supported: TextDBN, MSCNN, Restormer, StackDMPHN, SRN, NAFNet, FNAFNet")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(len(models)):
        if is_checkpoint[i]:
            checkpoint = torch.load(model_paths[i])
            models[i].load_state_dict(checkpoint['model_state_dict'])
        else:
            models[i].load_state_dict(torch.load(model_paths[i]))

        models[i].eval()
        models[i] = models[i].to(device)

    dataset = InferenceDataset(blur_img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    blur_img_paths = os.listdir(blur_img_dir)
    sharp_img_paths = [os.path.join(sharp_img_dir, blur_img) for blur_img in blur_img_paths]

    if window_slicing:
        testers = []
        for k in range(len(model_names)):
            if model_names[k] == "StackDMPHN":
                testers.append(StackDMPHNTester())
            elif model_names[k] == "MSCNN":
                testers.append(MSCNNTester())
            elif model_names[k] == "SRN":
                testers.append(SRNTester())
            else:
                testers.append(BaseTester())

    count = 0
    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = torch.zeros(X.shape).to(device)
            for i in range(len(models)):
                if (model_names[i] == "TextDBN" or model_names[i] == "Restormer" or model_names[i] == "NAFNet" or
                        model_names[i] == "FNAFNet"):
                    if not window_slicing:
                        pred += models[i](X)
                    else:
                        pred += testers[i].sliding_window_deblur(models[i], X, window_size=window_size,
                                                                 overlap_size=overlap_size)
                elif model_names[i] == "StackDMPHN" or model_names[i] == "SRN":
                    if not window_slicing:
                        out = models[i](X)
                        pred += out[len(out) - 1]
                    else:
                        out = testers[i].sliding_window_deblur(models[i], X, window_size=window_size,
                                                               overlap_size=overlap_size)
                        pred += out[len(out) - 1]
                elif model_names[i] == "MSCNN":
                    if not window_slicing:
                        upscale_factor = model_names[i].upscale_factor
                        B1 = X + 0.5
                        height, width = B1.shape[2], B1.shape[3]
                        resize_transform1 = Resize(
                            (int(height / upscale_factor ** 1), int(width / upscale_factor ** 1)))
                        resize_transform2 = Resize(
                            (int(height / upscale_factor ** 2), int(width / upscale_factor ** 2)))
                        B2 = resize_transform1(B1)
                        B3 = resize_transform2(B1)
                        X = [(B1 - 0.5).float(), (B2 - 0.5).float(), (B3 - 0.5).float()]
                        X[0], X[1], X[2] = X[0].to(device), X[1].to(device), X[2].to(device)

                        out = models[i](X)
                        pred += out[0]
                    else:
                        pred += testers[i].sliding_window_deblur(models[i], X, window_size=window_size,
                                                                 overlap_size=overlap_size)

            denormalized_pred = pred / len(models) + 0.5
            denormalized_pred = torch.clamp(denormalized_pred, 0, 1)

            for i in range(denormalized_pred.shape[0]):
                denormalized_pred_img = denormalized_pred[i]
                torch.clamp(denormalized_pred_img, 0, 1)
                save_img(denormalized_pred_img, sharp_img_paths[count])
                count += 1

    print("Successfully saved deblurred images in: ", sharp_img_dir)


def multi_modal_deblur(models, model_names, model_paths, blur_img_path, sharp_img_path, is_checkpoint,
                       window_slicing=False, window_size=256, overlap_size=0):
    """Deblurs image given multiple pretrained models by averaging outputs.

    Deblurs a single blurry image specified by a path given multiple
    pretrained models. For the given image, the outputs of all the models given 
    are averaged to give us the final deblurred image. This takes advantage of 
    the strengths and weaknesses of each model.

    Args:
        models: list of instances of different models (e.g Restormer, MSCNN, etc.)
        model_names: list of the model names.
        model_paths: list of the paths of the checkpoints (model, optimizer and
            lr_scheduler) or just the list of the paths of the models. 
        blur_img_path: path of the blurry image. 
        sharp_img_path: path in which all the deblurred image shall be stored.
        batch_size: batch size in which to perform deblurring of images. 
        is_checkpoint: list specifying if path specified in model_path[i] is a 
            checkpoint (model, optimizer and lr_scheduler) or just a model.
        window_slicing: if True, image is sliced into overlapping windows of
            size window_size and overlap overlap_size. Deblurring is then 
            performed in each window and merged back appropriately. This 
            takes into account the fact that training of deblurring models
            are usually performed on small image patches.
        window_size: size of the window in pixels that the image is sliced 
            into.
        overlap_size: size of overlap between two windows in pixels.  
    """

    if len(models) != len(model_names) or len(models) != len(model_paths) or len(models) != len(is_checkpoint):
        raise Exception("Length of models, model_names, model_paths and is_checkpoint have to be the same.")

    for i in range(len(model_names)):
        if (model_names[i] != "TextDBN" and model_names[i] != "Restormer" and model_names[i] != "NAFNet" and
                model_names[i] != "FNAFNet"
                and model_names[i] != "StackDMPHN" and model_names[i] != "SRN" and model_names[i] != "MSCNN"):
            raise Exception(model_names[i], "is either not valid or not supported. Currently the following models are \
                            supported: TextDBN, MSCNN, Restormer, StackDMPHN, SRN, NAFNet, FNAFNet")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(len(models)):
        if is_checkpoint[i]:
            checkpoint = torch.load(model_paths[i])
            models[i].load_state_dict(checkpoint['model_state_dict'])
        else:
            models[i].load_state_dict(torch.load(model_paths[i]))

        models[i].eval()
        models[i] = models[i].to(device)

    if window_slicing:
        testers = []
        for k in range(len(model_names)):
            if model_names[k] == "StackDMPHN":
                testers.append(StackDMPHNTester())
            elif model_names[k] == "MSCNN":
                testers.append(MSCNNTester())
            elif model_names[k] == "SRN":
                testers.append(SRNTester())
            else:
                testers.append(BaseTester())

    blur_img = ((read_image(blur_img_path).type(torch.float32)) / 255).to(device)
    blur_img -= 0.5
    blur_img = blur_img.unsqueeze(0)

    with torch.no_grad():
        pred = torch.zeros(blur_img.shape).to(device)
        for i in range(len(models)):
            if (model_names[i] == "TextDBN" or model_names[i] == "Restormer" or model_names[i] == "NAFNet" or
                    model_names[i] == "FNAFNet"):
                if not window_slicing:
                    pred += models[i](blur_img)
                else:
                    pred += testers[i].sliding_window_deblur(models[i], blur_img, window_size=window_size,
                                                             overlap_size=overlap_size)
            elif model_names[i] == "StackDMPHN" or model_names[i] == "SRN":
                if not window_slicing:
                    out = models[i](blur_img)
                    pred += out[len(out) - 1]
                else:
                    out = testers[i].sliding_window_deblur(models[i], blur_img, window_size=window_size,
                                                           overlap_size=overlap_size)
                    pred += out[len(out) - 1]
            elif model_names[i] == "MSCNN":
                if not window_slicing:
                    upscale_factor = model_names[i].upscale_factor
                    B1 = blur_img + 0.5
                    height, width = B1.shape[2], B1.shape[3]
                    resize_transform1 = Resize(
                        (int(height / upscale_factor ** 1), int(width / upscale_factor ** 1)))
                    resize_transform2 = Resize(
                        (int(height / upscale_factor ** 2), int(width / upscale_factor ** 2)))
                    B2 = resize_transform1(B1)
                    B3 = resize_transform2(B1)
                    X = [(B1 - 0.5).float(), (B2 - 0.5).float(), (B3 - 0.5).float()]
                    X[0], X[1], X[2] = X[0].to(device), X[1].to(device), X[2].to(device)

                    out = models[i](X)
                    pred += out[0]
                else:
                    pred += testers[i].sliding_window_deblur(models[i], blur_img, window_size=window_size,
                                                             overlap_size=overlap_size)

        denormalized_pred = pred / len(models) + 0.5
        denormalized_pred = torch.clamp(denormalized_pred, 0, 1)
        denormalized_pred = denormalized_pred.squeeze(0)

    save_img(denormalized_pred, sharp_img_path)
    display_img(denormalized_pred)


def multi_modal_deblur_imgs_by_brisque(models, model_names, model_paths, blur_img_dir, sharp_img_dir, is_checkpoint,
                                       batch_size=2,
                                       window_slicing=False, window_size=256, overlap_size=0):
    """Deblurs images given multiple pretrained models using the brisque index.

    Deblurs all images in a specified directory given multiple pretrained
    models. For a given image, the outputs of all the models are evaluated using
    the Brisque Index. Then the output with the highest brisque score is 
    selected. This takes advantage of the strengths and weaknesses of each model.

    Args:
        models: list of instances of different models (e.g Restormer, MSCNN, etc.)
        model_names: list of the model names.
        model_paths: list of the paths of the checkpoints (model, optimizer and
            lr_scheduler) or just the list of the paths of the models. 
        blur_img_dir: directory which contains all the blur images. 
        sharp_img_dir: directory in which all the deblurred images shall be
            stored.
        batch_size: batch size in which to perform deblurring of images. 
        is_checkpoint: list specifying if path specified in model_path[i] is a 
            checkpoint (model, optimizer and lr_scheduler) or just a model. 
        batch_size: batch size in which images are deblurred. 
        window_slicing: if True, image is sliced into overlapping windows of
            size window_size and overlap overlap_size. Deblurring is then 
            performed in each window and merged back appropriately. This 
            takes into account the fact that training of deblurring models
            are usually performed on small image patches.
        window_size: size of the window in pixels that the image is sliced 
            into.
        overlap_size: size of overlap between two windows in pixels.  
    """

    if len(models) != len(model_names) or len(models) != len(model_paths) or len(models) != len(is_checkpoint):
        raise Exception("Length of models, model_names, model_paths and is_checkpoint have to be the same.")

    for i in range(len(model_names)):
        if (model_names[i] != "TextDBN" and model_names[i] != "Restormer" and model_names[i] != "NAFNet" and
                model_names[i] != "FNAFNet"
                and model_names[i] != "StackDMPHN" and model_names[i] != "SRN" and model_names[i] != "MSCNN"):
            raise Exception(model_names[i], "is either not valid or not supported. Currently the following models are \
                            supported: TextDBN, MSCNN, Restormer, StackDMPHN, SRN, NAFNet, FNAFNet")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(len(models)):
        if is_checkpoint[i]:
            checkpoint = torch.load(model_paths[i])
            models[i].load_state_dict(checkpoint['model_state_dict'])
        else:
            models[i].load_state_dict(torch.load(model_paths[i]))

        models[i].eval()
        models[i] = models[i].to(device)

    dataset = InferenceDataset(blur_img_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    blur_img_paths = os.listdir(blur_img_dir)
    sharp_img_paths = [os.path.join(sharp_img_dir, blur_img) for blur_img in blur_img_paths]

    if window_slicing:
        testers = []
        for k in range(len(model_names)):
            if model_names[k] == "StackDMPHN":
                testers.append(StackDMPHNTester())
            elif model_names[k] == "MSCNN":
                testers.append(MSCNNTester())
            elif model_names[k] == "SRN":
                testers.append(SRNTester())
            else:
                testers.append(BaseTester())

    count = 0
    best_brisque_index = 100
    temp_pred = 0

    with torch.no_grad():
        for X in dataloader:
            X = X.to(device)
            pred = torch.zeros(X.shape).to(device)
            for i in range(len(models)):
                if (model_names[i] == "TextDBN" or model_names[i] == "Restormer" or model_names[i] == "NAFNet" or
                        model_names[i] == "FNAFNet"):
                    if not window_slicing:
                        temp_pred = models[i](X)
                    else:
                        temp_pred = testers[i].sliding_window_deblur(models[i], X, window_size=window_size,
                                                                     overlap_size=overlap_size)
                elif model_names[i] == "StackDMPHN" or model_names[i] == "SRN":
                    if not window_slicing:
                        out = models[i](X)
                        temp_pred = out[len(out) - 1]
                    else:
                        out = testers[i].sliding_window_deblur(models[i], X, window_size=window_size,
                                                               overlap_size=overlap_size)
                        temp_pred = out[len(out) - 1]
                elif model_names[i] == "MSCNN":
                    if not window_slicing:
                        upscale_factor = model_names[i].upscale_factor
                        B1 = X + 0.5
                        height, width = B1.shape[2], B1.shape[3]
                        resize_transform1 = Resize(
                            (int(height / upscale_factor ** 1), int(width / upscale_factor ** 1)))
                        resize_transform2 = Resize(
                            (int(height / upscale_factor ** 2), int(width / upscale_factor ** 2)))
                        B2 = resize_transform1(B1)
                        B3 = resize_transform2(B1)
                        X = [(B1 - 0.5).float(), (B2 - 0.5).float(), (B3 - 0.5).float()]
                        X[0], X[1], X[2] = X[0].to(device), X[1].to(device), X[2].to(device)

                        out = models[i](X)
                        temp_pred = out[0]
                    else:
                        temp_pred = testers[i].sliding_window_deblur(models[i], X, window_size=window_size,
                                                                     overlap_size=overlap_size)

                temp_pred = temp_pred + 0.5
                temp_pred = torch.clamp(temp_pred, 0, 1)
                temp_brisque_index: torch.Tensor = brisque(temp_pred, data_range=1.)

                if temp_brisque_index < best_brisque_index:
                    pred = temp_pred
                    best_brisque_index = temp_brisque_index

            denormalized_pred = pred

            for i in range(denormalized_pred.shape[0]):
                denormalized_pred_img = denormalized_pred[i]
                save_img(denormalized_pred_img, sharp_img_paths[count])
                count += 1

    print("Successfully saved deblurred images in: ", sharp_img_dir)


def multi_modal_deblur_by_brisque(models, model_names, model_paths, blur_img_path, sharp_img_path, is_checkpoint,
                                  window_slicing=False, window_size=256, overlap_size=0):
    """Deblurs images given multiple pretrained models using the brisque index.

    Deblurs a single blurry image specified by a path given multiple
    pretrained models. For the given image, the outputs of all the models are 
    evaluated using the Brisque Index. Then the output with the highest brisque 
    score is selected. This takes advantage of the strengths and weaknesses of 
    each model.

    Args:
        models: list of instances of different models (e.g Restormer, MSCNN, etc.)
        model_names: list of the model names.
        model_paths: list of the paths of the checkpoints (model, optimizer and
            lr_scheduler) or just the list of the paths of the models. 
        blur_img_path: path of the blurry image. 
        sharp_img_path: path in which all the deblurred image shall be stored.
        is_checkpoint: list specifying if path specified in model_path[i] is a 
            checkpoint (model, optimizer and lr_scheduler) or just a model.
        window_slicing: if True, image is sliced into overlapping windows of
            size window_size and overlap overlap_size. Deblurring is then 
            performed in each window and merged back appropriately. This 
            takes into account the fact that training of deblurring models
            are usually performed on small image patches.
        window_size: size of the window in pixels that the image is sliced 
            into.
        overlap_size: size of overlap between two windows in pixels.  
    """

    if len(models) != len(model_names) or len(models) != len(model_paths) or len(models) != len(is_checkpoint):
        raise Exception("Length of models, model_names, model_paths and is_checkpoint have to be the same.")

    for i in range(len(model_names)):
        if (model_names[i] != "TextDBN" and model_names[i] != "Restormer" and model_names[i] != "NAFNet" and
                model_names[i] != "FNAFNet"
                and model_names[i] != "StackDMPHN" and model_names[i] != "SRN" and model_names[i] != "MSCNN"):
            raise Exception(model_names[i], "is either not valid or not supported. Currently the following models are \
                            supported: TextDBN, MSCNN, Restormer, StackDMPHN, SRN, NAFNet, FNAFNet")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(len(models)):
        if is_checkpoint[i]:
            checkpoint = torch.load(model_paths[i])
            models[i].load_state_dict(checkpoint['model_state_dict'])
        else:
            models[i].load_state_dict(torch.load(model_paths[i]))

        models[i].eval()
        models[i] = models[i].to(device)

    if window_slicing:
        testers = []
        for k in range(len(model_names)):
            if model_names[k] == "StackDMPHN":
                testers.append(StackDMPHNTester())
            elif model_names[k] == "MSCNN":
                testers.append(MSCNNTester())
            elif model_names[k] == "SRN":
                testers.append(SRNTester())
            else:
                testers.append(BaseTester())

    blur_img = ((read_image(blur_img_path).type(torch.float32)) / 255).to(device)
    blur_img -= 0.5
    blur_img = blur_img.unsqueeze(0)
    best_brisque_index = 100
    temp_pred = 0

    with torch.no_grad():
        pred = torch.zeros(blur_img.shape).to(device)
        for i in range(len(models)):
            if (model_names[i] == "TextDBN" or model_names[i] == "Restormer" or model_names[i] == "NAFNet" or
                    model_names[i] == "FNAFNet"):
                if not window_slicing:
                    temp_pred = models[i](blur_img)
                else:
                    temp_pred = testers[i].sliding_window_deblur(models[i], blur_img, window_size=window_size,
                                                                 overlap_size=overlap_size)

                print("TextDBN\n\n")
            elif model_names[i] == "StackDMPHN" or model_names[i] == "SRN":
                if not window_slicing:
                    out = models[i](blur_img)
                    temp_pred = out[len(out) - 1]
                else:
                    out = testers[i].sliding_window_deblur(models[i], blur_img, window_size=window_size,
                                                           overlap_size=overlap_size)
                    temp_pred = out[len(out) - 1]
            elif model_names[i] == "MSCNN":
                if not window_slicing:
                    upscale_factor = model_names[i].upscale_factor
                    B1 = blur_img + 0.5
                    height, width = B1.shape[2], B1.shape[3]
                    resize_transform1 = Resize(
                        (int(height / upscale_factor ** 1), int(width / upscale_factor ** 1)))
                    resize_transform2 = Resize(
                        (int(height / upscale_factor ** 2), int(width / upscale_factor ** 2)))
                    B2 = resize_transform1(B1)
                    B3 = resize_transform2(B1)
                    X = [(B1 - 0.5).float(), (B2 - 0.5).float(), (B3 - 0.5).float()]
                    X[0], X[1], X[2] = X[0].to(device), X[1].to(device), X[2].to(device)

                    out = models[i](X)
                    temp_pred = out[0]
                else:
                    temp_pred = testers[i].sliding_window_deblur(models[i], blur_img, window_size=window_size,
                                                                 overlap_size=overlap_size)

            temp_pred = temp_pred + 0.5
            temp_pred = torch.clamp(temp_pred, 0, 1)
            temp_brisque_index: torch.Tensor = brisque(temp_pred, data_range=1.)
            if temp_brisque_index < best_brisque_index:
                pred = temp_pred
                best_brisque_index = temp_brisque_index

        denormalized_pred = pred
        denormalized_pred = denormalized_pred.squeeze(0)

    save_img(denormalized_pred, sharp_img_path)
    display_img(denormalized_pred)
