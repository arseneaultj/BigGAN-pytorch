import os
import torch
from torch.autograd import Variable
from torch.nn import init
import imageio


def make_folder(path, version):
        if not os.path.exists(os.path.join(path, version)):
            os.makedirs(os.path.join(path, version))


def tensor2var(x, grad=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=grad)


def var2tensor(x):
    return x.data.cpu()


def var2numpy(x):
    return x.data.cpu().numpy()


def denorm(x):
    out = (x + 1) / 2
    return out.clamp_(0, 1)


def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data)
        init.constant_(m.bias.data, 0.0)


def generate_gif(path):
    filenames = []
    for (_, _, files) in os.walk(path):
        filenames.extend(files)
        break
    with imageio.get_writer('samples3.gif', mode='I', duration=0.3) as writer:
        for idx, filename in enumerate(filenames):
            if idx % 10 == 0:
                image = imageio.imread(path + filename)
                writer.append_data(image)