import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

from models import *
#from advertorch.attacks import LinfPGDAttack
from autoattack import AutoAttack
from tqdm import tqdm

file_name = 'pgd_adversarial_training'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=4)

net = ResNet18()
net = net.to(device)
net = torch.nn.DataParallel(net)
cudnn.benchmark = True
checkpoint = torch.load('./checkpoint/' + file_name)
#print('model loaded successfully.')
#print('device:',device)
net.load_state_dict(checkpoint['net'])

#adversary = LinfPGDAttack(net, loss_fn=nn.CrossEntropyLoss(), eps=0.0314, nb_iter=7, eps_iter=0.00784, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
adversary = AutoAttack(net,norm='Linf',eps=8/255,version='standard')
criterion = nn.CrossEntropyLoss()

def test():
    print('\n[ Test Start ]')
    net.eval()
    benign_loss = 0
    adv_loss = 0
    benign_correct = 0
    adv_correct = 0
    #print('net.eval() works fine')
    total = 0
    for batch_idx, (inputs, targets) in tqdm(enumerate(test_loader)):
        #print('---------')
        #print('batch',batch_idx)
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)

        outputs = net(inputs)
        loss = criterion(outputs, targets)
        benign_loss += loss.item()

        _, predicted = outputs.max(1)
        benign_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('\nCurrent batch:', str(batch_idx))
            print('Current benign test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current benign test loss:', loss.item())

        batch_size = len(targets)
        #adv = adversary.perturb(inputs, targets)
        x_adv = adversary.run_standard_evaluation(inputs,targets,bs=batch_size)
        adv_outputs = net(x_adv)
        loss = criterion(adv_outputs, targets)
        adv_loss += loss.item()

        _, predicted = adv_outputs.max(1)
        adv_correct += predicted.eq(targets).sum().item()

        if batch_idx % 10 == 0:
            print('Current adversarial test accuracy:', str(predicted.eq(targets).sum().item() / targets.size(0)))
            print('Current adversarial test loss:', loss.item())

    print('\nTotal benign test accuarcy:', 100. * benign_correct / total)
    print('Total adversarial test Accuarcy:', 100. * adv_correct / total)
    print('Total benign test loss:', benign_loss)
    print('Total adversarial test loss:', adv_loss)

test()

