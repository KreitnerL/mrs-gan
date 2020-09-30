from __future__ import print_function
import numpy as np
from PIL import Image
import numpy as np
import os
import sys


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)

# def tensor2plot(spectra, *labels, **save):
#     if len(spectra)==1:
#         fig, ax = plt.subplots()
#         ax.plot(spectra,len(spectra))
#         ax.set(xlabel='PPM',ylabel='Intensity',title='MR Spectra')
#         # plt.show()
#         del fig
#     elif len(spectra)==2:               # Allow for plotting the real and synthetic spectra on the same axes
#         fig, ax = plt.subplots()
#         line1 = ax.plot(spectra[0],len(spectra[0]))
#         line1.set_label = labels[0]
#         line2 = ax.plot(spectra[1],len(spectra[1]))
#         line2.set_label = labels[1]
#         ax.set(xlabel='PPM',ylabel='Intensity',title='MR Spectra')
#         ax.legend(loc='best')
#         # plt.show()
#         del fig
#     return ax

# def diagnose_network(net, name='network'):
#     mean = 0.0
#     count = 0
#     for param in net.parameters():
#         if param.grad is not None:
#             mean += torch.mean(torch.abs(param.grad.data))
#             count += 1
#     if count > 0:
#         mean = mean / count
#     print(name)
#     print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

# def info(object, spacing=10, collapse=1):
#     """Print methods and doc strings.
#     Takes module, class, list, dictionary, or string."""
#     methodList = [e for e in dir(object) if isinstance(getattr(object, e), collections.Callable)]
#     processFunc = collapse and (lambda s: " ".join(s.split())) or (lambda s: s)
#     print( "\n".join(["%s %s" %
#                      (method.ljust(spacing),
#                       processFunc(str(getattr(object, method).__doc__)))
#                      for method in methodList]) )

# def varname(p):
#     for line in inspect.getframeinfo(inspect.currentframe().f_back)[3]:
#         m = re.search(r'\bvarname\s*\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*\)', line)
#         if m:
#             return m.group(1)

# def print_numpy(x, val=True, shp=False):
#     x = x.astype(np.float64)
#     if shp:
#         print('shape,', x.shape)
#     if val:
#         x = x.flatten()
#         print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
#             np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        print('path = ',path)
        oldmask = os.umask(0o000)
        os.makedirs(path, mode=0o755)
        os.umask(oldmask)


def progressbar(it, prefix="", size=60, file=sys.stdout):
    count = len(it)
    def show(j):
        x = int(size*j/count)
        file.write("%s[%s%s] %i/%i\r" % (prefix, "#"*x, "."*(size-x), j, count))
        file.flush()        
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    file.write("\n")
    file.flush()
