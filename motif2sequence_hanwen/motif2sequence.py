from basemodel import BaseModel
import sequenceModels
import torch

class P2P_model(BaseModel):

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:
            self.model_names = ['G']
        self.visual_names = ['realA', 'fakeB', 'realB']
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        self.netG = sequenceModels.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, gpu_ids=opt.gpu_ids)
        if self.isTrain:
            self.netD = sequenceModels.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD, gpu_ids=opt.gpu_ids)
        if self.isTrain:
            self.criterionGAN = sequenceModels.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
        from IPython import embed; embed()

    def set_input(self, input):
        self.realA = torch.tensor(input['A']).to(self.device)
        self.realB = torch.tensor(input['B']).to(self.device)
        #self.image_paths = input['A_paths']

    def forward(self):
        self.fakeB = self.netG(self.realA)

    def backward_D(self):
        fakeAB = torch.cat((self.realA, self.fakeB), 1)
        predFake = self.netD(fakeAB.detach())
        self.loss_D_fake = self.criterionGAN(predFake, False)
        realAB = torch.cat((self.realA, self.realB), 1)
        predReal = self.netD(realAB)
        self.loss_D_real = self.criterionGAN(predReal, True)
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        fakeAB = torch.cat((self.realA, self.fakeB), 1)
        predFake = self.netD(fakeAB)
        self.loss_G_GAN = self.criterionGAN(predFake, True)
        self.loss_G_L1 = self.criterionL1(self.fakeB, self.realB) * self.opt.lmd_l1
        self.loss_G = self.loss_G_L1 + self.loss_G_GAN
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()
        self.set_requires_grad(self.netD, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        self.set_requires_grad(self.netD, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def compute_G_loss(self):
        self.test()
        fakeAB = torch.cat((self.realA, self.fakeB), 1)
        predFake = self.netD(fakeAB)
        self.loss_G_GAN_test = self.criterionGAN(predFake, True)
        self.loss_G_L1_test = self.criterionL1(self.fakeB, self.realB)
        self.loss_G_test = self.loss_G_L1_test + self.loss_G_GAN_test








