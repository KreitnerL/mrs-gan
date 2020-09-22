# NOT USED

import torch
# from statistics import mean
import torch.jit
# from sklearn.metrics import mean_squared_error

__all__ = ['percent_error', 'pearson_correlation_coefficient', 'mean_squared_error',
           'mean_absolute_error']

# @torch.jit.script
def mean_squared_error(target, estimate):
    assert(len(target)==len(estimate))
    assert(target.dim()==estimate.dim())
    assert(target.device==estimate.device)

    return torch.sum((estimate - target)**2) / estimate.data.nelement()
    # return mean_squared_error(target, estimate)

# @torch.jit.script
def mean_absolute_error(target, estimate):
    assert(len(target)==len(estimate))
    assert(target.dim()==estimate.dim())
    assert(target.device==estimate.device)

    return torch.sum(estimate - target) / estimate.data.nelement()

# @torch.jit.script
def percent_error(target, estimate):
    '''
    Error > 0: understimating
    Error < 0: overestimating

    :param target:
    :param estimate:
    :return:
    '''
    assert(len(target)==len(estimate))
    assert(target.dim()==estimate.dim())
    assert(target.device==estimate.device)

    return torch.mean((target - estimate) / target)

# @torch.jit.export
def pearson_correlation_coefficient(target, estimate):
    '''
    Code adapted from: https://pythonprogramming.net/how-to-program-r-squared-machine-learning-tutorial/

    :param target:
    :param estimate:
    :return: R^2 value
    '''
    def best_fit_slope_and_intercept(xs, ys):
        # x, y = mean(xs), mean(ys)
        y = torch.mean(ys)
        x = torch.mean(xs)
        m = (((x * y) - torch.mean(xs * ys)) /
             ((x * x) - torch.mean(xs * xs)))
        b = y - m * x
        return m, b

    def squared_error(ys_orig, ys_line):
        # print('ys_orig.device: ',ys_orig.device)
        # print('ys_line.device: ',ys_line.device)
        return torch.sum((ys_line - ys_orig) ** 2)

    def coefficient_of_determination(ys_orig, ys_line):
        # print('ys_orig.shape: ',ys_orig.shape)
        y_mean_line = torch.tensor([torch.mean(ys_orig) for y in ys_orig]).to(ys_orig.device)
        squared_error_regr = squared_error(ys_orig, ys_line)
        squared_error_y_mean = squared_error(ys_orig, y_mean_line)
        return 1 - (squared_error_regr/squared_error_y_mean)

    assert(len(target)==len(estimate))
    assert(target.dim()==estimate.dim())
    assert(target.device==estimate.device)

    m, b = best_fit_slope_and_intercept(estimate, target)
    regression_line = torch.tensor([(m*x)+b for x in estimate]).to(target.device)

    return coefficient_of_determination(target, regression_line)


# def accuracy(target, estimate, normalize=True):
#     acc = []
#     assert(len(target)==len(estimate))
#     assert(target.dim()==estimate.dim())
#     assert(target.device==estimate.device)
#
#     if target.dim() > 1:
#         target = torch.squeeze(target)
#     # if y_pred.dim() > 1:
#         estimate = torch.squeeze(estimate).detach()
#
#     for i in range(len(target)):
#         acc.append((target[i] - estimate[i])/target[i])
#
#     if normalize==True:
#         score = torch.FloatTensor(acc).mean()
#     else:
#         score = (torch.FloatTensor(acc)>=0.5).nonzero().numel()
#
#     return score
