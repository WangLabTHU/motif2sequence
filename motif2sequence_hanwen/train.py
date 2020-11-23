from basemodel import BaseModel
import sequenceModels
import torch
from motif2sequence import P2P_model
from dataset import CMP_dataset
from options import train_opt, test_opt
from torch.utils.data import DataLoader
from tensor2seq import save_sequence, tensor2seq



def log_info(dir, train_loss):
    with open(dir, 'a') as f:
        print('epoch: {} training loss G: {}\n'.format(epoch, train_loss))
        f.write('epoch: {} training loss G: {}\n'.format(epoch, train_loss))


opt_train = train_opt()
opt_test = test_opt()
model = P2P_model(opt_train)
CMP_train = CMP_dataset(opt_train)
CMP_test = CMP_dataset(opt_test)
dataset_train = DataLoader(dataset=CMP_train, batch_size=300, shuffle=True)
dataset_test = DataLoader(dataset=CMP_test, batch_size=1, shuffle=False)
model.setup(opt_train)


for epoch in range(opt_train.epoch_count, opt_train.n_epochs + opt_train.n_epochs_decay + 1):
    model.update_learning_rate()
    num = 0
    train_loss = 0
    for i, data in enumerate(dataset_train):
        model.set_input(data)
        model.optimize_parameters()
        train_loss += model.loss_G
        num += 1
    train_loss /= num
    test_loss = 0
    num = 0
    log_info(opt_train.log_dir, train_loss)
tensorSeq = []
for i, data in enumerate(dataset_test):
    model.set_input(data)
    model.test()
    tensorSeq.append(model.get_current_visuals())
    save_sequence(tensorSeq)