
CKPT_DIR = "addxshift07"

class DefaultConfig(object):
    data_dir = 'DataSets\\CityScape'
    train_img_dir = 'DataSets\\CityScape\\train'
    train_annot_dir = 'DataSets\\CityScape\\trainannot'
    valid_img_dir='DataSets\\CityScape\\val_350f'
    valid_annot_dir='DataSets\\CityScape\\val_350f_annot'

    train_with_ckpt = False
    logdir = "checkpoints/"+CKPT_DIR
    ckpt_name = "checkpoints/"+CKPT_DIR+"/ckpt"
    model_path = "checkpoints/"+CKPT_DIR+"/Model.pth"
    
    ckpt_path = "checkpoints/CKPT/6.pth"

    batch_size = 12
    val_batch_size = 25

    dataloader_num_worker = 0
    class_num = 21

    learning_rate = 4e-4
    max_epoch = 220

    crop = True
    crop_rate = 0.8

    rand_ext = True
    ext_range = [0, 0, 0, 0.7, 0, 0]
    ext_param = [0, 0, 0, 0, 0, 0]

    rand_f = True
    f = 350
    f_range = [200, 400]

    fish_size = [640, 640]

    mask_radius = 100
