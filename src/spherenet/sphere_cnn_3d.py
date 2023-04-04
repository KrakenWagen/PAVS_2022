import numpy as np
from functools import lru_cache
import torch
from torch import nn
from torch.nn.parameter import Parameter

from spherenet.sphere_cnn import cal_index

@lru_cache(None)
def cal_index_3d(t, h, w, img_t, img_r, img_c):
    '''
        Calculate Kernel Sampling Pattern
        only support 3x3x3 filter
        return 27 locations: (3, 3, 3, 3)
    '''

    idx_2d = cal_index(h,w,img_r, img_c) # (1, 3, 3, 2)
    idx_2d = np.stack([idx_2d, idx_2d, idx_2d], axis=0) # (3, 3, 3, 2)

    tt_shape = (3, 3, 1)
    tt_mat = np.stack([np.full(tt_shape, max(img_t-1, 0)),
                       np.full(tt_shape, img_t),
                       np.full(tt_shape, min(img_t+1, t-1))], axis=0)# (3, 3, 3, 1)

    res = np.concatenate([tt_mat, idx_2d], axis=3)
    return res # (3, 3, 3, 3)

@lru_cache(None)
def _gen_filters_coordinates_3d(t, h, w, stride):
    co = np.array([[[cal_index_3d(t, h, w, tt, i, j) for j in range(0, w, stride)] for i in range(0, h, stride)] for tt in range(0, t, stride)])
    return np.ascontiguousarray(co.transpose([6, 0, 1, 2, 3, 4, 5])) # ll, i, j, img_r, img_c
# ll, t, i, j, img_t, img_r, img_c


def gen_filters_coordinates_3d(t, h, w, stride=1):
    '''
    return np array of kernel lo (3, H/stride, W/stride, 3, 3)
    '''
    assert(isinstance(h, int) and isinstance(w, int) and isinstance(t, int))
    return _gen_filters_coordinates_3d(t, h, w, stride).copy()


def gen_grid_coordinates_3d(t, h, w, stride=1):
    coordinates = gen_filters_coordinates_3d(t, h, w, stride).copy()
    coordinates[0] = (coordinates[0] * 2 / t) - 1
    coordinates[1] = (coordinates[1] * 2 / h) - 1
    coordinates[2] = (coordinates[2] * 2 / w) - 1
    coordinates = coordinates[::-1]
    coordinates = coordinates.transpose(1, 4, 2, 5, 3, 6, 0)
    sz = coordinates.shape
    coordinates = coordinates.reshape(1, sz[0]*sz[1], sz[2]*sz[3], sz[4]*sz[5], sz[6])

    return coordinates.copy()

class SphereCoordinates3D():
    coordinates_dict = {}
    grids_array = []
    shapes_array = []

    @staticmethod
    def hasCoordinates(shape):
        return False
        return shape[-1] in SphereCoordinates3D.coordinates_dict and shape[-2] in SphereCoordinates3D.coordinates_dict[
            shape[-1]]

    @staticmethod
    def getCoordinates(shape):
        #if SphereCoordinates3D.hasCoordinates(shape):
            print("Getting coordinates", shape)
            idx = SphereCoordinates3D.coordinates_dict[shape[-1]][shape[-2]]
            return SphereCoordinates3D.shapes_array[idx], SphereCoordinates3D.grids_array[idx]
        #else:
        #    return None

    @staticmethod
    def setCoordinates(shape, grid_shape, grid):
        print("SETTING COORDINATES", shape)
        return None
        if shape[-1] not in SphereCoordinates3D.coordinates_dict:
            SphereCoordinates3D.coordinates_dict[shape[-1]] = {}
        SphereCoordinates3D.coordinates_dict[shape[-1]][shape[-2]] = len(SphereCoordinates3D.grids_array)
        SphereCoordinates3D.shapes_array.append(grid_shape)
        SphereCoordinates3D.grids_array.append(grid)

class SphereConv3D(nn.Module):
    '''  SphereConv3D
    Note that this layer only support 3x3x3 filter
    '''
    def __init__(self, in_c, out_c, stride=1, bias=True, mode='bilinear'):
        super(SphereConv3D, self).__init__()
        self.in_c = in_c
        self.out_c = out_c
        self.stride = stride
        self.mode = mode
        self.weight = Parameter(torch.Tensor(out_c, in_c, 3, 3, 3))
        if bias:
            self.bias = Parameter(torch.Tensor(out_c))
        else:
            self.register_parameter('bias', None)
        self.grid_shape = None
        self.grid = None

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            self.bias.data.zero_()

    def forward(self, x):
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[-3:]):
            if SphereCoordinates3D.hasCoordinates(x.shape):
                self.grid_shape, self.grid = SphereCoordinates3D.getCoordinates(x.shape)
            else:
                self.grid_shape = tuple(x.shape[-3:])
                coordinates = gen_grid_coordinates_3d(x.shape[-3], x.shape[-2], x.shape[-1], self.stride)
                with torch.no_grad():
                    self.grid = torch.FloatTensor(coordinates).to(x.device)
                    self.grid.requires_grad = True
                SphereCoordinates3D.setCoordinates(x.shape, self.grid_shape, self.grid)

        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1, 1).to(x.device)
        x = nn.functional.grid_sample(x, grid, mode=self.mode, align_corners=True)
        x = nn.functional.conv3d(x, self.weight, self.bias, stride=3, groups=1)
        return x

class SphereMaxPool3D(nn.Module):
    '''  SphereMaxPool3D
    Note that this layer only support 3x3x3 filter
    '''
    def __init__(self, stride=1, mode='bilinear'):
        super(SphereMaxPool3D, self).__init__()
        self.stride = stride
        self.mode = mode
        self.grid_shape = None
        self.grid = None
        self.pool = nn.MaxPool3d(kernel_size=3, stride=3)

    def forward(self, x):
        if self.grid_shape is None or self.grid_shape != tuple(x.shape[-3:]):
            if SphereCoordinates3D.hasCoordinates(x.shape):
                self.grid_shape, self.grid = SphereCoordinates3D.getCoordinates(x.shape)
            else:
                self.grid_shape = tuple(x.shape[-3:])
                coordinates = gen_grid_coordinates_3d(x.shape[-3], x.shape[-2], x.shape[-1], self.stride)
                with torch.no_grad():
                    self.grid = torch.FloatTensor(coordinates).to(x.device)
                    self.grid.requires_grad = True

                SphereCoordinates3D.setCoordinates(x.shape, self.grid_shape, self.grid)


        with torch.no_grad():
            grid = self.grid.repeat(x.shape[0], 1, 1, 1, 1).to(x.device)
        return self.pool(nn.functional.grid_sample(x, grid, mode=self.mode, align_corners=True))
