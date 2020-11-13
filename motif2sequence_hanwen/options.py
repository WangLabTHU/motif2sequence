class train_opt():

    def __init__(self, input_nc=5, output_nc=5):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = 64
        self.ndf = 64
        self.epoch = 10
        self.lr_policy = 'linear'
        self.netG = 'unet_128'
        self.netD = 'basic'
        self.gan_mode = 'vanilla'
        self.isTrain = True
        self.gpu_ids = '0'
        self.checkpoints_dir = 'checkpoints/'
        self.name = 'model_check'
        self.preprocess=''
        self.lr = 0.0002
        self.beta1 = 0.5
        self.lr_decay_iters = 40
        self.lmd_l1 = 100
        self.continue_train = False
        self.verbose = True
        self.log_dir = 'log_info.txt'
        self.n_epochs_decay = 100
        self.epoch_count = 1
        self.n_epochs = 100


class test_opt():
    def __init__(self, input_nc=3, output_nc=3):
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.lmd_l1 = 0.6
        self.ngf = 64
        self.ndf = 128
        self.epoch = 100
        self.lr_policy = 'step'
        self.netG = 'resnet_9blocks'
        self.netD = 'basic'
        self.gan_mode = 'lsgan'
        self.isTrain = False
        self.gpu_ids = None
        self.checkpoints_dir = 'checkpoints/'
        self.name = 'model_check'
        self.preprocess=''
        self.lr = 0.0002
        self.beta1 = 0.5
        self.lr_decay_iters = 20
        self.continue_train = False
        self.verbose = True