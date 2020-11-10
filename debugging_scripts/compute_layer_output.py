def conv_output(L_in, padding=0, dilation=1, kernelsize=1, stride=1):
    return int(((L_in + 2 * padding - dilation*(kernelsize-1)-1)/stride) +1)

print(conv_output(256, kernelsize=3, padding=1, stride=1))