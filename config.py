

class DefaultConfig(object):
    train_img_dir = 'DataSets\\CityScape\\train'
    train_annot_dir = 'DataSets\\CityScape\\trainannot'
    valid_img_dir='DataSets\\CityScape\\val_350f'
    valid_annot_dir='DataSets\\CityScape\\val_350f_annot'

    train_with_ckpt = False
    logdir = "checkpoints/350F_RandShift"
    ckpt_name = "checkpoints/350F_RandShift/ckpt_RandShift"
    model_path = "checkpoints/350F_RandShift/Model.pth"
    ckpt_path = "checkpoints/ckpt_focalloss_weight/ckpt_test_168.pth"

    batch_size = 12
    val_batch_size = 25

    dataloader_num_worker = 0
    class_num = 21

    learning_rate = 4e-4
    max_epoch = 200

    crop = True
    crop_rate = 0.8

    rand_ext = True
    ext_range = [5, 5, 5, 0.3, 0.1, 0.4]
    ext_param = [0, 0, 0, 0, 0, 0]

    rand_f = False
    f = 350
    f_range = [200, 400]

    fish_size = [640, 640]


