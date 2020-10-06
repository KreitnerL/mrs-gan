from torch.optim import lr_scheduler

def setup_for_no_TTUR(opt):
    if not opt.TTUR:
        opt.glr = opt.lr
        opt.dlr = opt.lr
        opt.n_epochs_gen = opt.n_epochs
        opt.n_epochs_dis = opt.n_epochs
        opt.n_epochs_gen_decay = opt.n_epochs_decay
        opt.n_epochs_dis_decay = opt.n_epochs_decay

def get_scheduler_G(optimizer, opt):
    return get_scheduler(optimizer, opt, opt.n_epochs_gen, opt.n_epochs_gen_decay)

def get_scheduler_D(optimizer, opt):
    return get_scheduler(optimizer, opt, opt.n_epochs_dis, opt.n_epochs_dis_decay)


def get_scheduler(optimizer, opt, n_epochs, n_epochs_decay):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        opt (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.n_epochs> epochs
    and linearly decay the rate to zero over the next <opt.n_epochs_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - n_epochs) / float(n_epochs_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler