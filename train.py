import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.linalg import norm
from skimage.util import random_noise

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from autoattack import AutoAttack
#from tqdm.notebook import tqdm
from tqdm import tqdm

parser = argparse.ArgumentParser(description="Train the model with SOAR")
parser.add_argument(
    "checkpoint_dir",
    type=str,
    help="directory to store the checkpoint data"
)
parser.add_argument(
    "checkpoint_freq",
    type=int,
    help="save a checkpoint after how many epochs"
)
parser.add_argument(
    "xi",
    type=float,default=1e-6,
    help="step size for the second-order term"
)
parser.add_argument(
    "learning_rate",
    type=float,default=1e-1,
    help="learning rate for the model"
)
parser.add_argument(
    "steps_fd",
    type=int,default=10,
    help="steps of approximation for finite difference"
)
args=parser.parse_args()

learning_rate = args.learning_rate 
epsilon = 8/255
k = 7
alpha = 2/255
file_name = 'pgd_adversarial_training'
xi = args.xi
STEPS = args.steps_fd
N_EPOCHS = 200
checkpoint_dir = args.checkpoint_dir
checkpoint_freq = args.checkpoint_freq
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
validation_dataset,test_dataset = torch.utils.data.random_split(torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test),[5000,5000],generator=torch.Generator().manual_seed(42))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=100, num_workers=4)

class LinfPGDAttack(object):
  def __init__(self, model):
    self.model = model

  def perturb(self, x_natural, y):
    def get_alpha_2(alpha_1, d):
      return alpha_1 / torch.max(torch.abs(d)) 
        
    x = x_natural.detach()        
    for i in range(k):
      # First order term
      x.requires_grad_()
      with torch.enable_grad():
        logits = self.model(x)
        loss = F.cross_entropy(logits, y)
        grad = torch.autograd.grad(loss, [x])[0]
      # Second order term
      d = torch.Tensor(random_noise(x.to('cpu').detach().numpy(),mode='gaussian')).to(device)
      d = d / norm(d)
      for _ in range(STEPS):
        x_pert = x + xi * d/norm(d) # torch.zeros_like(x).uniform_(-epsilon, epsilon) +
        x_pert.requires_grad_()
        logits_new = self.model(x_pert)
        loss = F.cross_entropy(logits_new,y)
        Hd = (torch.autograd.grad(loss,[x_pert])[0]-grad)/xi
        print('norm_diff_norm',norm(d-Hd/norm(Hd)))
        d = Hd/norm(Hd)
      print(f'attack step {i}, grad_x norm{norm(grad)}, Hd {norm(Hd)}')
      alpha_2 = get_alpha_2(alpha,d)
      x = x + alpha * torch.sign(grad.detach()) + alpha_2 * d
      # projection
      x = torch.min(torch.max(x, x_natural - epsilon), x_natural + epsilon)
      x = torch.clamp(x, 0, 1)
    return x


net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
state = {
        'net': net.state_dict()
    }

adversary = LinfPGDAttack(net)
adv_AA = AutoAttack(net,norm='Linf',eps=epsilon,version='standard')
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0002)
scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=learning_rate, max_lr=0.1)


def get_loss_and_correct(net, batch, criterion, device, adversary=adversary,is_vali=False):
  # Implement forward pass and loss calculation for one batch.
  # Remember to move the batch to device.
  # 
  # Return a tuple:
  # - loss for the batch (Tensor)
  # - number of correctly classified examples in the batch (Tensor)
  pass
  #for batch_idx, (data,target) in enumerate(train_dataloader):
  data, target = batch
  data, target = data.to(device), target.to(device)
  if not is_vali:
    adv = adversary.perturb(data, target)    
  else:
    adv = adversary.run_standard_evaluation(data,target,bs=len(target))
  adv_outputs = net(adv)
  loss = criterion(adv_outputs, target)

  _, predicted = adv_outputs.max(1)

  #total += target.size(0)
  correct = predicted.eq(target).sum()
  return loss, correct  #(pred.argmax(1) == target).type(torch.float).sum()



def step(loss,optimizer):
  optimizer.zero_grad()
  loss.backward()
  optimizer.step();
  #scheduler.step()   

def adjust_learning_rate(optimizer):
  for param_group in optimizer.param_groups:
    param_group['lr'] = learning_rate


net.train()

train_losses = []
train_accuracies = []
validation_losses = []
validation_accuracies = []

pbar = tqdm(range(N_EPOCHS))

best_val_accuracy = 0;
n_no_improve = 0;
es_step_tolerance = 5;

for i in pbar:
  total_train_loss = 0.0
  total_train_correct = 0.0
  total_validation_loss = 0.0
  total_validation_correct = 0.0

  net.train()

  for batch in tqdm(train_loader):#, leave=False):
    loss, correct = get_loss_and_correct(net, batch, criterion, device)
    step(loss, optimizer)
    total_train_loss += loss.item()
    total_train_correct += correct.item()
  
  scheduler.step()
  #with torch.no_grad():
  for batch in validation_loader:
    loss, correct = get_loss_and_correct(net, batch, criterion, device)#, adversary=adv_AA, is_vali=True)
    total_validation_loss += loss.item()
    total_validation_correct += correct.item()
  
  mean_train_loss = total_train_loss / len(train_dataset)
  train_accuracy = total_train_correct / len(train_dataset)

  mean_validation_loss = total_validation_loss / len(validation_dataset)
  validation_accuracy = total_validation_correct / len(validation_dataset)

  train_losses.append(mean_train_loss)
  validation_losses.append(mean_validation_loss)

  pbar.set_postfix({'train_loss': mean_train_loss, 'validation_loss': mean_validation_loss, 'train_accuracy': train_accuracy, 'validation_accuracy': validation_accuracy})


  if validation_accuracy > best_val_accuracy:
    best_val_accuracy = validation_accuracy;
    best_idx = i
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state, './' + checkpoint_dir + '/best_' + file_name + '.pt')
    n_no_improve = 0;
  else:
    n_no_improve += 1;
  
  if i % checkpoint_freq == 0 and i > 0:
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state, './'+ checkpoint_dir + '/' + str(i) + '_' + file_name + '.pt')
  if best_val_accuracy > 0.75:
    print("Early stopped at epoch ",i+1)
    break
  if n_no_improve > es_step_tolerance:
    if best_val_accuracy > 0.75:
      print("Early stopped at epoch ",i+1)
      break
print('best model is from epoch', best_idx)
