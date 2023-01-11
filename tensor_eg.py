import torch
import numpy as np
from einops import rearrange, repeat

''' torch.roll(input, shifts, dims=None):
Roll the tensor 'input' along the given dimension
Elements that are shifted beyond the last position are re-introduced at the first position.
If dims is None, the tensor will be flattened before rolling and then restored to the original shape.
Args:
    - input (tensor)
    - shifts (int or tuple of ints) - The number of places by which the elements of the tensor are shifted. 
    If shifts is a tuple, dims must be a tuple of the same size, and each dimension will be rolled by the corresponding value
    - dims (int or tuple of ints) - Axis along which to roll
'''

x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8]).view(4, 2)
print('x before rolling:')
print(x)

x_rolled_1 = torch.roll(x, shifts=1)
print('x after rolling shifted 1:')
print(x_rolled_1)

x_rolled_2 = torch.roll(x, shifts=2)
print('x after rolling shifted 2:')
print(x_rolled_2)

x_roll_minus1a = torch.roll(x, shifts=-1, dims=1)
print('x after rolling shifted -1 dim 1:')
print(x_roll_minus1a)

x_roll_minus1b = torch.roll(x, shifts=-1, dims=0)
print('x after rolling shifted -1 dim 0:')
print(x_roll_minus1b)

x_roll = torch.roll(x, shifts=(2, 1), dims=(0, 1))
print('x after rolling (shifted 2 on dim 0 and shifted 1 on dim 1):')
print(x_roll)


''' torch.rearrange
A reader-friendly smart element reordering for multidimensional tensors.
This operation includes functionality of transpose (axes permutation), reshape(view), squeeze, unsqueeze, stack,
concatenate and other operations.
'''

# suppose we have a set of 32 images in "h w c" format (height - width - channel)
images = [np.random.randn(30, 40, 3) for _ in range(32)]

print('Image in a list of images shape: ')
print(images[0].shape)

# stack along first (batch) axis, output is a single array
print('stack along first (batch) axis, output is a single array')
print(rearrange(images, 'b h w c -> b h w c').shape)

# concatenate images along height (vertical axis), 960 = 32 * 30
print('concatenate images along height (vertical axis), 960 = 32 * 30')
print(rearrange(images, 'b h w c -> (b h) w c').shape)

# concatenate images along horizontal axis, 1280 = 32 * 40
print('concatenate images along horizontal axis, 1280 = 32 * 40')
print(rearrange(images, 'b h w c -> h (b w) c').shape)

# reorder axes to 'b c h w' format for deep learning
print('reorder axes to b c h w format for deep learning')
print(rearrange(images, 'b h w c -> b c h w').shape)

# flatten each image into a vector, 3600 = 30 * 40 * 3
print('flatten each image into a vector, 3600 = 30 * 40 * 3')
print(rearrange(images, 'b h w c -> b (c h w)').shape)

# split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2
print('split each image into 4 smaller (top-left, top-right, bottom-left, bottom-right), 128 = 32 * 2 * 2')
print(rearrange(images, 'b (h1 h) (w1 w) c -> (b h1 w1) h w c', h1=2, w1=2).shape)

# space to depth operation
print('space to depth operation')
print(rearrange(images, 'b (h h1) (w w1) c -> b h w (c h1 w1)', h1=2, w1=2).shape)

