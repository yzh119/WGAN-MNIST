
# This code follows the instruction of https://github.com/martinarjovsky/WassersteinGAN/blob/master/main.py
# Notice how to freeze G and D respectively.
# When training D, let G be volatile, and set all parameters in D to False.
# When training G, set all parameters of D to False.
# The reason doing so is that when training D, we don't need any gradient information 
# concerning G, but it is not the case when we train G, in which scenario we have to use 
# the gradient back-propagated by D.
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import training_monitor.logger as tmlog
from torch.autograd import Variable
import torch.utils.data
from torchvision import datasets, transforms
import torchvision.utils as vutils
from model import *

logger = tmlog.Logger('LOG')

n_epochs = 25
init_Diters = 5
lrD = 5e-4
lrG = 5e-4
clamp_lower = -0.01
clamp_upper = 0.01
batch_size = 64
ngpu = 2
nz = 50
nc = 1
ngf = 64
ndf = 64
isize = 28

kwargs = {'num_workers': 1, 'pin_memory': True}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
batch_size=batch_size, shuffle=True, **kwargs)

G = Generator(nz, nc, ngf, isize, ngpu)
D = Discriminator(nz, nc, ndf, isize, ngpu)
G.cuda()
D.cuda()

optimizerD = optim.RMSprop(D.parameters(), lr = lrD)
optimizerG = optim.RMSprop(G.parameters(), lr = lrG)

input = torch.FloatTensor(batch_size, nc, isize, isize)
noise = torch.FloatTensor(batch_size, nz, 1, 1)
fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
one = torch.FloatTensor([1])
mone = torch.FloatTensor([-1])
input.cuda()
noise.cuda()
fixed_noise.cuda()
one.cuda()
mone.cuda()


def set_require_grads(net, val=True):
    for p in net.parameters():
        p.requires_grad = val

# The reason we use clamp_ rather than clamp here is that
# p.clamp will return a new Variable but it will not change p itself.
# p.clamp_ is an in-place version and it will clamp p.
def clamp_grads(net, lower, upper):
    for p in net.parameters():
        p.data.clamp_(lower, upper)

# The main loop is a little bit different from the vanilla NN training pipeline for the reason
# that we need to train D several times as we train G once.

g_iters = 0
for epoch in xrange(n_epochs):
    i = 0
    data_iter = iter(train_loader) # Bulid a new iterator in each epoch.
    
    while i < len(data_iter):
        if g_iters < 25 or g_iters % 500 == 0:
            D_iters = 100
        else:
            D_iters = init_Diters
        
        # Train D
        set_require_grads(D, True)
        j = 0
        while i < len(data_iter) and j < D_iters:
            j += 1
            clamp_grads(D, clamp_lower, clamp_upper)
            i, (real_cpu, _) = i + 1, data_iter.next()
            
            D.zero_grad()
            # 
            input.resize_as_(real_cpu).copy_(real_cpu)
            inputv = Variable(input)
            err_real = D(inputv)
            err_real.backward(one)
            
            noise.normal_(0, 1)
            noisev = Variable(noise, volatile=True)
            fake = G(noisev)
            err_fake = D(Variable(fake.data))
            err_fake.backward(mone)
            err_d = err_real - err_fake
            optimizerD.step()
        
        # Train G        
        set_require_grads(D, False)
        G.zero_grad()
        
        noise.normal_(0, 1)
        noisev = Variable(noise)
        fake = G(noisev)
        err_g = D(fake)
        err_g.backward(one)
        g_iters += 1
        optimizerG.step()
        
        logger.add_scalar('lossd', err_d.data[0], g_iters)
        logger.add_scalar('lossg', err_g.data[0], g_iters)
        logger.add_scalar('lossreal', err_real.data[0], g_iters)
        logger.add_scalar('lossfake', err_fake.data[0], g_iters)
        
        print('epoch: {}/{}, batches: {}/{}, Loss_D: {}, Loss_G: {}, Loss_D_real: {}, Loss_D_fake: {}'.format(
            epoch, n_epochs, i, len(train_loader), 
            err_d.data[0], err_g.data[0], err_real.data[0], err_fake.data[0]))
        
        # To dump fake samples
        if g_iters % 500 == 0:
            real_cpu = real_cpu.mul(0.5).add(0.5)
            vutils.save_image(real_cpu, 'img/real_samples.png')
            fake = G(Variable(fixed_noise, volatile=True))
            fake.data = fake.data.mul(0.5).add(0.5)
            vutils.save_image(fake.data, 'img/fake_samples_{}.png'.format(g_iters))
    
    torch.save(G.state_dict(), 'params/generator.params')
    torch.save(D.state_dict(), 'params/discriminator.params')            