import os

class Config:
    data_root = '/workspace/aesthetic_cropping/dataset/'
    predefined_pkl = os.path.join(data_root, 'pdefined_anchors.pkl') # download from https://github.com/luwr1022/listwise-view-ranking/blob/master/pdefined_anchors.pkl
    FLMS_folder = os.path.join(data_root, 'FLMS')
    GAIC_folder = os.path.join(data_root, 'GAICD')

    image_size = (256,256)
    backbone = 'vgg16'

    # training
    gpu_id = 0
    num_workers = 4
    batch_size  = 1
    keep_aspect_ratio = True
    data_augmentation = True

    max_epoch = 50
    lr = 1e-4
    lr_decay = 0.1
    lr_decay_epoch = [max_epoch + 1]
    weight_decay = 1e-4
    eval_freq = 1
    save_freq = max_epoch+1
    display_freq = 100

    prefix = 'CGS'
    exp_root = os.path.join(os.getcwd(), './experiments/')
    exp_name = prefix
    exp_path = os.path.join(exp_root, prefix)
    while os.path.exists(exp_path):
        index = os.path.basename(exp_path).split(prefix)[-1].split('repeat')[-1]
        try:
            index = int(index) + 1
        except:
            index = 1
        exp_name = prefix + ('_repeat{}'.format(index))
        exp_path = os.path.join(exp_root, exp_name)
    # print('Experiment name {} \n'.format(os.path.basename(exp_path)))
    checkpoint_dir = os.path.join(exp_path, 'checkpoints')
    log_dir = os.path.join(exp_path, 'logs')

    def create_path(self):
        print('Create experiment directory: ', self.exp_path)
        os.makedirs(self.exp_path)
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)

cfg = Config()

if __name__ == '__main__':
    cfg = Config()