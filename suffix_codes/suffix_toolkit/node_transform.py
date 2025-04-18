import numpy as np
from scipy.ndimage import rotate, zoom

# 现有的变换类保持不变
class MinMaxNormalize3D:
    def __call__(self, img):
        for i in range(img.shape[0]):
            channel = img[i]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val != min_val:
                img[i] = (channel - min_val) / (max_val - min_val)
            else:
                img[i] = np.zeros_like(channel)
        return img

class RandomRotate3D:
    def __init__(self, max_angle=5):
        self.max_angle = max_angle
        self.angles = None

    def __call__(self, img):
        if self.angles is None:
            self.angles = np.random.uniform(-self.max_angle, self.max_angle, 3)
        img = rotate(img, angle=self.angles[0], axes=(1, 2), reshape=False, mode='nearest')
        img = rotate(img, angle=self.angles[1], axes=(0, 2), reshape=False, mode='nearest')
        img = rotate(img, angle=self.angles[2], axes=(0, 1), reshape=False, mode='nearest')
        return img

class RandomFlip3D:
    def __init__(self):
        self.flip_axes = None

    def __call__(self, img):
        if self.flip_axes is None:
            self.flip_axes = [np.random.rand() < 0.5 for _ in range(3)]
        for axis, flip in enumerate(self.flip_axes):
            if flip:
                img = np.flip(img, axis=axis).copy()
        return img

class RandomShift3D:
    def __init__(self, max_shift=5):
        self.max_shift = max_shift
        self.shifts = None

    def __call__(self, img):
        if self.shifts is None:
            self.shifts = np.random.randint(-self.max_shift, self.max_shift, 3)
        img = np.roll(img, self.shifts[0], axis=0)
        img = np.roll(img, self.shifts[1], axis=1)
        img = np.roll(img, self.shifts[2], axis=2)
        return img

class RandomZoom3D:
    def __init__(self, zoom_range=(0.9, 1.1)):
        self.zoom_range = zoom_range
        self.zoom_factor = None

    def __call__(self, img):
        if self.zoom_factor is None:
            self.zoom_factor = np.random.uniform(self.zoom_range[0], self.zoom_range[1])
        zoomed = np.zeros_like(img)
        for i in range(img.shape[0]):
            zoomed_slice = zoom(img[i], self.zoom_factor, mode='nearest')
            zoomed_slice = self._adjust_size(zoomed_slice, img.shape[1:])
            zoomed[i] = zoomed_slice
        return zoomed
    
    def _adjust_size(self, zoomed_slice, target_shape):
        for dim in range(len(target_shape)):
            if zoomed_slice.shape[dim] != target_shape[dim]:
                if zoomed_slice.shape[dim] > target_shape[dim]:
                    start = (zoomed_slice.shape[dim] - target_shape[dim]) // 2
                    end = start + target_shape[dim]
                    zoomed_slice = np.take(zoomed_slice, np.arange(start, end), axis=dim)
                else:
                    pad_width = [(0, 0)] * len(target_shape)
                    pad_width[dim] = ((target_shape[dim] - zoomed_slice.shape[dim]) + 1) // 2, (target_shape[dim] - zoomed_slice.shape[dim]) // 2
                    zoomed_slice = np.pad(zoomed_slice, pad_width, mode='constant', constant_values=0)
        return zoomed_slice

class AddNoise3D:
    def __init__(self, noise_std=0.01):
        self.noise_std = noise_std
        self.noise = None

    def __call__(self, img):
        if self.noise is None or self.noise.shape != img.shape:
            self.noise = np.random.normal(0, self.noise_std, img.shape)
        img = img + self.noise
        #img = np.clip(img, 0, 1)
        return img

class ZScoreNormalize3D:
    def __call__(self, img):
        for i in range(img.shape[0]):
            channel = img[i]
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            if std_val != 0:  # 防止除以零
                img[i] = (channel - mean_val) / std_val
            else:
                img[i] = np.zeros_like(channel)  # 如果标准差为零，返回零数组
        return img

# 新增屏蔽类：以一定概率将特征图置为零
class RandomMask3D:
    def __init__(self, mask_prob=0.2):
        self.mask_prob = mask_prob
        self.mask = None

    def __call__(self, img):
        if self.mask is None:
            self.mask = np.random.rand() < self.mask_prob
        if self.mask:
            return np.zeros_like(img)  # 以指定概率返回全零数组
        return img  # 否则返回原图
