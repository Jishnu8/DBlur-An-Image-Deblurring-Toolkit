from dblur.trainers.mscnn import MSCNNTrainer
from dblur.testers.mscnn import MSCNNTester


def train(checkpoint_save_name, train_img_dir, val_img_dir=None, batch_size=2, epochs=2000,
          save_checkpoint_freq=100, val_freq=100):
    """Trains MSCNN model.

    Trains a MSCNN model on the given training dataset with the default
    parameters mentioned in the paper "Deep Multi-scale Convolutional Neural 
    Network for Dynamic Scene Deblurring".

    Args:
        train_img_dir: directory of training data
        val_img_dir: directory of validation data
        checkpoint_save_name: name of checkpoint (includes model, optimizer, 
            lr_scheduler) to be saved at a given frequency. 
        batch_size: batch size for training. Default value set to 2 as in the 
            paper.
        epochs: number of epochs for training.
        save_checkpoint_freq: frequency at which checkpoint is saved.
        val_freq: frequency at which validation is performed. 
    """

    mscnn_trainer = MSCNNTrainer()

    train_dataloader = mscnn_trainer.get_train_dataloader(train_img_dir, batch_size=batch_size)
    if val_img_dir is not None:
        val_dataloader = mscnn_trainer.get_val_dataloader(val_img_dir, batch_size=batch_size)
    else:
        val_dataloader = None

    model = mscnn_trainer.get_model()
    discriminator = mscnn_trainer.get_discriminator()
    model_optimizer = mscnn_trainer.get_model_optimizer(model.parameters())
    discriminator_optimizer = mscnn_trainer.get_discriminator_optimizer(discriminator.parameters())
    lr_scheduler = mscnn_trainer.get_lr_scheduler(model_optimizer)
    model_loss_func = mscnn_trainer.get_model_loss()
    discriminator_loss_func = mscnn_trainer.get_discriminator_loss()
    mscnn_trainer.train(model=model,
                        discriminator=discriminator,
                        train_dataloader=train_dataloader,
                        val_dataloader=val_dataloader,
                        model_optimizer=model_optimizer,
                        discriminator_optimizer=discriminator_optimizer,
                        model_loss_func=model_loss_func,
                        discriminator_loss_func=discriminator_loss_func,
                        save_checkpoint_freq=save_checkpoint_freq,
                        logs_folder='runs',
                        checkpoint_save_name=checkpoint_save_name,
                        val_freq=val_freq,
                        write_logs=True,
                        lr_scheduler=lr_scheduler,
                        scheduler_step_every_batch=True,
                        epochs=epochs)


def test(model_path, test_img_dir, is_checkpoint=True, batch_size=2):
    """Tests MSCNN model.

    Tests a MSCNN model on the given testing dataset with the default
    parameters mentioned in the paper "Deep Multi-scale Convolutional Neural 
    Network for Dynamic Scene Deblurring".
    
    Args:
        model_path: path of the checkpoint (model, optimizer and lr_scheduler) 
            or just the model.
        test_img_dir: directory of testing data.
        is_checkpoint: specifies if path specified in model_path is a 
            checkpoint (model, optimizer and lr_scheduler) or just a model.
        batch_size: batch size for testing.
    """

    mscnn_tester = MSCNNTester()
    test_dataloader = mscnn_tester.get_test_dataloader(test_img_dir, batch_size=batch_size)
    model = mscnn_tester.get_model()
    loss_func = mscnn_tester.get_loss()
    mscnn_tester.test(model,
                      model_path,
                      test_dataloader,
                      loss_func,
                      is_checkpoint=is_checkpoint,
                      window_slicing=False)


def deblur_imgs(model_path, blur_img_dir, sharp_img_dir, is_checkpoint=True, batch_size=32):
    """Deblurs images in directory using MSCNN model.

    Deblurs images using a MSCNN model on the given directory with the default
    parameters mentioned in the paper "Deep Multi-scale Convolutional Neural
    Network for Dynamic Scene Deblurring". Please ensure that the checkpoint
    provided corresponds to the model architecture specified with the default
    parameters.

    Args:
        model_path: path of the checkpoint (model, optimizer and lr_scheduler)
            or just the model.
        blur_img_dir: directory containing the blurred images.
        sharp_img_dir: directory in which the deblurred images shall be stored.
        is_checkpoint: specifies if path specified in model_path is a
            checkpoint (model, optimizer and lr_scheduler) or just a model.
        batch_size: batch size for deblurring images.
    """

    mscnn_tester = MSCNNTester()
    model = mscnn_tester.get_model()
    mscnn_tester.deblur_imgs(model,
                             model_path,
                             blur_img_dir,
                             sharp_img_dir,
                             is_checkpoint=is_checkpoint,
                             batch_size=batch_size,
                             window_slicing=False)


def deblur_single_img(model_path, blur_img_path, sharp_img_path, is_checkpoint=True):
    """Deblurs image using MSCNN model.

    Deblurs image using a MSCNN model ith the default parameters
    mentioned in the paper "Deep Multi-scale Convolutional Neural Network for
    Dynamic Scene Deblurring". Please ensure that the checkpoint provided corresponds
    to the model architecture specified with the default parameters.

    Args:
        model_path: path of the checkpoint (model, optimizer and lr_scheduler)
            or just the model.
        blur_img_path: path of the blurred images.
        sharp_img_path: path in which the deblurred image shall be stored.
        is_checkpoint: specifies if path specified in model_path is a
            checkpoint (model, optimizer and lr_scheduler) or just a model.
    """

    mscnn_tester = MSCNNTester()
    model = mscnn_tester.get_model()
    mscnn_tester.deblur_single_img(model,
                                   model_path,
                                   blur_img_path,
                                   sharp_img_path,
                                   is_checkpoint=is_checkpoint,
                                   window_slicing=False)
