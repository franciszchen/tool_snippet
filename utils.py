import os 
import time 
import random
import json 
import numpy as np 
from collections import OrderedDict

import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt

import torch 
from scipy.special import softmax as scipy_softmax

############ record curve #####################
def init_dict(keys):
    d = {}
    for key in keys:
        d[key] = []
    return d

############ record curve #####################
def save_dict(info_dict, theme, save_dir):
    
    with open(os.path.join(save_dir, 'infodict-{}.json'.format(theme)), 'w') as f:
        f.write(json.dumps(info_dict))

def read_dict(filename):
    with open(filename, 'r') as f:
        info_dict = json.load(f)
    return info_dict

############ record curve #####################
def curve_save(x, y, tag, yaxis, theme, save_dir):
    color = ['r', 'b', 'g', 'c', 'orange', 'lightsteelblue', 'cornflowerblue', 'indianred']
    fig = plt.figure()
    # ax = plt.subplot()
    plt.grid(linestyle='-', linewidth=0.5, alpha=0.5)
    if isinstance(tag, list):
        for i, (y_term, tag_term) in enumerate(zip(y, tag)):
            plt.plot(x, y_term, color[i], label=tag_term, alpha=0.7)
    else:
        plt.plot(x, y, color[0], label=tag, alpha=0.7)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel(yaxis, fontsize=12)
    plt.title('curve-{}'.format(theme), fontsize=14)
    plt.legend()

    fig.savefig(os.path.join(save_dir,'curve-{}.png'.format(theme)), dpi=300)
    plt.close('all') ####

"""
info_keys_norm = [ 
    'norm_epochs',
    'norm_centroids',
]
info_dicts_norm = {
    'Server': init_dict(keys=info_keys_norm),  
    'A': init_dict(keys=info_keys_norm), 
    'B': init_dict(keys=info_keys_norm), 
    'C': init_dict(keys=info_keys_norm), 
    'D': init_dict(keys=info_keys_norm), 
}
curve_save(
            x=info_dicts_norm[datasets[0]]['norm_epochs'], 
            y=[
                info_dicts_norm['Server']['norm_centroids'], 
                info_dicts_norm[datasets[0]]['norm_centroids'], 
                info_dicts_norm[datasets[1]]['norm_centroids'], 
                info_dicts_norm[datasets[2]]['norm_centroids'], 
                info_dicts_norm[datasets[3]]['norm_centroids'], 
                ], 
            tag=['Server', 'client_A', 'client_B', 'client_C', 'client_D'],
            yaxis='Norm centroids', 
            theme='Norm-centroids-all-client', 
            save_dir=log_path
        )
    
"""

#################################################

def time_mark():
    time_now = int(time.time())
    time_local = time.localtime(time_now)

    dt = time.strftime('%Y%m%d-%H%M%S', time_local)
    return(dt)

def print_cz(str, f=None):
    if f is not None:
        print(str, file=f)
        if random.randint(0, 20) < 3:
            f.flush()
    print(str)

def expand_user(path):
    return os.path.abspath(os.path.expanduser(path))

def update_lr(lr, epoch, lr_step=20, lr_gamma=0.5):
    """Sets the learning rate to the initial LR decayed by 0.5 every 20 epochs"""
    lr = lr * (lr_gamma ** (epoch // lr_step)) 
    return lr

def update_lr_multistep(args, lr_current, step, lr_steps=None, lr_gamma=0.1):
    if step in lr_steps:
        lr_current = lr_current*lr_gamma
        print_cz("step {} in lr_steps {}".format(step, lr_steps), f=args.logfile)
        print_cz("lr change to {}".format(lr_current), f=args.logfile)
    return lr_current

def adjust_learning_rate(optimizer, lr, epoch, lr_step=40, lr_gamma=0.1):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = lr * (lr_gamma ** (epoch // lr_step)) # 
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

#########
def model_snapshot(model, new_file, old_file=None, save_dir='./', verbose=True, log_file=None):
    """
    :param model: network model to be saved
    :param new_file: new pth name
    :param old_file: old pth name
    :param verbose: more info or not
    :return: None
    """
    if os.path.exists(save_dir) is False:
        os.makedirs(expand_user(save_dir))
        print_cz(str='Make new dir:'+save_dir, f=log_file)
    if isinstance(model, torch.nn.DataParallel):
        model = model.module
    for file in os.listdir(save_dir):
        if old_file in file:
            if verbose:
                print_cz(str="Removing old model  {}".format(expand_user(save_dir + file)), f=log_file)
            os.remove(save_dir + file) # 先remove旧的pth，再存储新的
    if verbose:
        print_cz(str="Saving new model to {}".format(expand_user(save_dir + new_file)), f=log_file)
    torch.save(model, expand_user(save_dir + new_file))
    # torch.save(model.state_dict(),expand_user(save_dir + 'dict_'+new_file))

def remove_oldfile(dirname, file_keyword):
    for filename in os.listdir(dirname):
        if file_keyword in filename:
            os.remove(os.path.join(dirname, filename))
        
#########
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0# value = current value
        self.avg = 0
        self.sum = 0# weighted sum
        self.count = 0# total sample num

    def update(self, value, n=1):# n是加权数
        self.value = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    # output和target都是Tensor，分别是FloatTensor和LongTensor
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)#取maxk个预测值
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))#将target也进行扩展

    res = []
    # 遍历top1和top5 accuracy
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))# res是本batch内的accuracy
    return res

def returnCAM(feature, weight_softmax, class_idx):
    """check feature_conv>0, following relu
    """
    B, C, Z = feature.shape #bz, nc, h, w = feature_conv.shape #原本的case，bz=1，所以后续随意的reshape
    importance_classes = []
    for idx in class_idx:
        # 抹掉维度C
        importance = np.sum(
            (weight_softmax[idx]).reshape(1, -1, 1) * feature,
            axis=1,
            keepdims=False
            ) #bz*nc*z -> bz*z
        importance = scipy_softmax(importance, axis=-1) # 在z个Instance上进行归一化
        importance_classes.append(importance)

    importance_classes_npy = np.stack(importance_classes, axis=1) # bz*class*z
    # print('importance_classes_npy.shape:\t', importance_classes_npy.shape)
    # print('Z:\t', Z)
    importance_std = np.zeros((B, Z))
    for b in range(B):
        for z in range(Z):
            importance_std[b, z] = np.std(importance_classes_npy[b, :, z]) # std>=0
        # importance_std[b] = (importance_std[b] - np.min(importance_std[b]))/(np.max(importance_std[b]) - np.min(importance_std[b]))
        if np.sum(importance_std[b])>1e-10:
            importance_std[b] = importance_std[b] /np.sum(importance_std[b]) 
        else:
            print('std sum problem!:\t', np.sum(importance_std[b]), importance_std[b])
            importance_std[b] = np.ones((Z))/float(Z)
    # importance_std = softmax(importance_std, axis=-1)
    return importance_classes, importance_std

#######################
def seed_fix_all(seed_idx, logfile=None):
    # specific seed
    np.random.seed(seed_idx)
    torch.manual_seed(seed_idx)   
    torch.cuda.manual_seed(seed_idx)  
    torch.cuda.manual_seed_all(seed_idx) 
    #
    random.seed(seed_idx) # new
    torch.backends.cudnn.deterministic = True # new
    print_cz('seed fixed: {}'.format(seed_idx), f=logfile)

def prepare():
    import config 
    import json

    args = config.get_args()
    args.main_file = __file__
    #
    log_path = args.save_path + time_mark() \
        + '_{}'.format(args.theme) \
        + '_{}'.format(args.optim) \
        + '_lr{}'.format(args.lr) \
        + '_steps{}'.format(args.lr_multistep) \
        + '_gamma{}'.format(args.lr_gamma) \
        + '_seed{:d}'.format(args.seed) \
        + '_iters{:d}'.format(args.iters) \
        + '_wk{}'.format(args.wk_iters) 
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logfile = open(os.path.join(log_path,'log.txt'), 'a')
    with open(log_path + '/setting.json', 'w') as f:
        f.write(json.dumps(args.__dict__, indent=4))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.device.lower() == 'cuda' and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    # print args info
    config.args_info(args, logfile=logfile)
    
    args.logfile = logfile
    SAVE_PTH_NAME = 'model'

##################
def check_classifier_norm(
    args, 
    model
):
    for key in model.state_dict().keys():
        if "centroids_param" in key:
            param_centroids = model.state_dict()[key]
            norm_centroids = torch.mean(torch.abs(param_centroids), dim=-1, keepdim=False)
            # print_cz("* centroid norm:\t {}".format(norm_centroids), f=args.logfile)
            norm_centroids_sum = torch.sum(norm_centroids, dim=-1, keepdim=False)
            # print_cz("* centroid norm sum:\t {}".format(norm_centroids_sum), f=args.logfile)
    return norm_centroids_sum.item()

def save_source_code(
    args,
    log_path,
    logfile
):
    args.logfile = logfile
    SAVE_PTH_NAME = 'model'

    import shutil
    save_file_list =[
        __file__,
        'server_gnn.py'
    ]
    for filename in save_file_list:
        print_cz("log_path: {},  filename: {}".format(log_path, filename), f=args.logfile)
        filename = filename.split('/')[-1]
        print_cz("source: {}".format(os.path.join('/home/zchen72/code/noiseFL-v2/', filename)), f=args.logfile)
        print_cz("target: {}".format(os.path.join(log_path, filename)), f=args.logfile)
        shutil.copyfile(
            os.path.join('/home/zchen72/code/noiseFL-v2/', filename),
            os.path.join(log_path, filename)
        )