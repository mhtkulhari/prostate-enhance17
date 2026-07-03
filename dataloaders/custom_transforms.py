import torch
import math
import numbers
import random
import numpy as np

from PIL import Image, ImageOps, ImageFilter
from scipy.ndimage.filters import gaussian_filter
from matplotlib.pyplot import imshow, imsave
from scipy.ndimage.interpolation import map_coordinates
import cv2
from scipy import ndimage
from torchvision import transforms
import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
from torch import nn


def _is_sequence_image(img):
    return isinstance(img, (list, tuple))


def _map_image(img, fn):
    if _is_sequence_image(img):
        return [fn(ch) for ch in img]
    return fn(img)


def _image_size(img):
    return img[0].size if _is_sequence_image(img) else img.size


def _stack_image(img):
    if _is_sequence_image(img):
        return np.stack([np.asarray(ch).astype(np.float32) for ch in img], axis=-1)
    arr = np.asarray(img).astype(np.float32)
    if arr.ndim == 2:
        arr = arr[..., np.newaxis]
    return arr


def _restore_image(arr, template):
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if _is_sequence_image(template):
        return [Image.fromarray(arr[..., i]) for i in range(arr.shape[-1])]
    if arr.shape[-1] == 1:
        return Image.fromarray(arr[..., 0])
    return Image.fromarray(arr)


def _ensure_uint8(img_np):
    img_np = np.asarray(img_np)
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
    return img_np


def _apply_clahe_gray(gray, clip_limit=2.0, tile_grid_size=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid_size, tile_grid_size))
    return clahe.apply(_ensure_uint8(gray))


def _normalize_to_uint8(channel):
    channel = channel.astype(np.float32)
    mn, mx = float(channel.min()), float(channel.max())
    if mx - mn < 1e-6:
        return np.zeros_like(channel, dtype=np.uint8)
    return np.clip((channel - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)


class RandomCLAHE(object):
    def __init__(self, p=0.5, clip_limit=2.0, tile_grid_size=8, blend_alpha=0.7):
        self.p = p
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.blend_alpha = blend_alpha

    def __call__(self, img):
        if random.random() > self.p:
            return img.copy() if hasattr(img, 'copy') else img

        def _one(ch):
            gray = _ensure_uint8(np.asarray(ch))
            out = _apply_clahe_gray(gray, self.clip_limit, self.tile_grid_size)
            if self.blend_alpha < 1.0:
                out = cv2.addWeighted(gray, 1.0 - self.blend_alpha, out, self.blend_alpha, 0)
            return Image.fromarray(out)

        return _map_image(img, _one)


class RandomGamma(object):
    def __init__(self, p=0.5, gamma_min=0.7, gamma_max=1.5):
        self.p = p
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max

    def __call__(self, img):
        if random.random() > self.p:
            return img.copy() if hasattr(img, 'copy') else img
        gamma = random.uniform(self.gamma_min, self.gamma_max)

        def _one(ch):
            arr = _ensure_uint8(np.asarray(ch))
            arr_f = arr.astype(np.float32) / 255.0
            out = np.power(arr_f, gamma) * 255.0
            return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

        return _map_image(img, _one)


class RandomBiasField(object):
    def __init__(self, p=0.3, strength=0.25, grid_size=4):
        self.p = p
        self.strength = strength
        self.grid_size = grid_size

    def __call__(self, img):
        if random.random() > self.p:
            return img.copy() if hasattr(img, 'copy') else img
        w, h = _image_size(img)
        low_res = np.random.uniform(
            1.0 - self.strength,
            1.0 + self.strength,
            (self.grid_size, self.grid_size)
        ).astype(np.float32)
        field = cv2.resize(low_res, (w, h), interpolation=cv2.INTER_CUBIC)
        field = gaussian_filter(field, sigma=max(1.0, min(h, w) * 0.03))
        field = field / (field.mean() + 1e-6)

        def _one(ch):
            arr = np.asarray(ch).astype(np.float32)
            out = arr * field
            return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))

        return _map_image(img, _one)


class RandomGaussianBlurMRI(object):
    def __init__(self, p=0.25, radius_min=0.2, radius_max=1.2):
        self.p = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        if random.random() > self.p:
            return img.copy() if hasattr(img, 'copy') else img
        radius = random.uniform(self.radius_min, self.radius_max)
        return _map_image(img, lambda ch: ch.filter(ImageFilter.GaussianBlur(radius=radius)))


def to_multilabel(pre_mask, classes = 2):
    mask = np.zeros((pre_mask.shape[0], pre_mask.shape[1], classes))
    mask[pre_mask == 1] = [0, 1]
    mask[pre_mask == 2] = [1, 1]
    return mask


class add_salt_pepper_noise():
    def __call__(self, sample):
        image = sample['image']
        X_imgs_copy = np.asarray(image).copy()

        salt_vs_pepper = 0.2
        amount = 0.004

        num_salt = np.ceil(amount * X_imgs_copy.size * salt_vs_pepper)
        num_pepper = np.ceil(amount * X_imgs_copy.size * (1.0 - salt_vs_pepper))

        seed = random.random()
        if seed > 0.75:
            # Add Salt noise
            coords = [np.random.randint(0, i - 1, int(num_salt)) for i in X_imgs_copy.shape]
            X_imgs_copy[coords[0], coords[1], :] = 1
        elif seed > 0.5:
            # Add Pepper noise
            coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in X_imgs_copy.shape]
            X_imgs_copy[coords[0], coords[1], :] = 0
        sample['image'] = X_imgs_copy
        return sample

class adjust_light():
    def __call__(self, sample):
        image = sample['image']
        seed = random.random()
        if seed > 0.5:
            gamma = random.random() * 3 + 0.5
            invGamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype(np.uint8)
            image = cv2.LUT(np.array(image).astype(np.uint8), table).astype(np.uint8)
            sample['image'] = image
        return sample

class Brightness():# new defined
    def __init__(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v

    def __call__(self, img):
        v = self.min_v + float(self.max_v-self.min_v)*random.random()
        return _map_image(img, lambda ch: PIL.ImageEnhance.Brightness(ch).enhance(v))

class Contrast():# new defined
    def __init__(self, min_v, max_v):
        self.min_v = min_v
        self.max_v = max_v

    def __call__(self, img):
        v = self.min_v + float(self.max_v-self.min_v)*random.random()
        return _map_image(img, lambda ch: PIL.ImageEnhance.Contrast(ch).enhance(v))

class GaussianBlur(object):
    """blur a single image on CPU"""
    def __init__(self, kernel_size, num_channels):
        self.num_channels = num_channels
        radias = kernel_size // 2
        kernel_size = radias * 2 + 1
        self.blur_h = nn.Conv2d(num_channels, num_channels, kernel_size=(kernel_size, 1),
                                stride=1, padding=0, bias=False, groups=num_channels)
        self.blur_v = nn.Conv2d(num_channels, num_channels, kernel_size=(1, kernel_size),
                                stride=1, padding=0, bias=False, groups=num_channels)
        self.k = kernel_size
        self.r = radias

        self.blur = nn.Sequential(
            nn.ReflectionPad2d(radias),
            self.blur_h,
            self.blur_v
        )

        self.pil_to_tensor = transforms.ToTensor()
        self.tensor_to_pil = transforms.ToPILImage()

    def __call__(self, img):
        img = self.pil_to_tensor(img).unsqueeze(0)

        sigma = np.random.uniform(0.1, 2.0)
        x = np.arange(-self.r, self.r + 1)
        x = np.exp(-np.power(x, 2) / (2 * sigma * sigma))
        x = x / x.sum()
        x = torch.from_numpy(x).view(1, -1).repeat(self.num_channels, 1)

        self.blur_h.weight.data.copy_(x.view(self.num_channels, 1, self.k, 1))
        self.blur_v.weight.data.copy_(x.view(self.num_channels, 1, 1, self.k))

        with torch.no_grad():
            img = self.blur(img)
            img = img.squeeze()

        img = self.tensor_to_pil(img)

        return img

class eraser():
    def __call__(self, sample, s_l=0.02, s_h=0.06, r_1=0.3, r_2=0.6, v_l=0, v_h=255, pixel_level=False):
        image = sample['image']
        img_h, img_w, img_c = image.shape


        if random.random() > 0.5:
            return sample

        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)

            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)

        image[top:top + h, left:left + w, :] = c
        sample['image'] = image
        return sample

class elastic_transform():
    """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
           Convolutional Neural Networks applied to Visual Document Analysis", in
           Proc. of the International Conference on Document Analysis and
           Recognition, 2003.
        """

    # def __init__(self):

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        w, h = _image_size(image)
        alpha = h * 2
        sigma = h * 0.08
        random_state = None
        seed = random.random()
        if seed > 0.5:
            image_np = _stack_image(image)
            label_np = np.asarray(label)
            image_channel = image_np.shape[-1]
            label_channel = len(label_np.shape)

            if random_state is None:
                random_state = np.random.RandomState(None)

            shape = image_np.shape[:2]
            dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
            dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

            x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
            indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))
            transformed_image = np.zeros_like(image_np)
            for i in range(image_channel):
                transformed_image[:, :, i] = map_coordinates(image_np[:, :, i], indices, order=1).reshape(shape)
            if label is not None:
                if label_channel == 3:
                    transformed_label = np.zeros_like(label_np)
                    for i in range(3):
                        transformed_label[:, :, i] = map_coordinates(label_np[:, :, i], indices, order=0, mode='nearest', prefilter=False).reshape(shape)
                elif label_channel == 2:
                    transformed_label = np.zeros_like(label_np)
                    transformed_label[:, :] = map_coordinates(label_np[:, :], indices, order=0, mode='nearest', prefilter=False).reshape(shape)
            else:
                transformed_label = None

            if label is not None:
                transformed_label = transformed_label.astype(np.uint8)
            sample['image'] = _restore_image(transformed_image, image)
            sample['label'] = transformed_label
        return sample

class cutout():

    def __init__(self):
        self.p=0.5
        self.size_min=0.02
        self.size_max=0.4
        self.ratio_1=0.3
        self.ratio_2=1/0.3
        self.value_min=0
        self.value_max=255
        self.pixel_level=True

    def __call__(self, sample):
        if random.random() < self.p:
            img, mask = sample['image'], sample['label']
            img = np.array(img)
            mask = np.array(mask)

            img_h, img_w = img.shape[0], img.shape[1]
            img_channel = len(img.shape)

            while True:
                size = np.random.uniform(self.size_min, self.size_max) * img_h * img_w
                ratio = np.random.uniform(self.ratio_1, self.ratio_2)
                erase_w = int(np.sqrt(size / ratio))
                erase_h = int(np.sqrt(size * ratio))
                x = np.random.randint(0, img_w)
                y = np.random.randint(0, img_h)

                if x + erase_w <= img_w and y + erase_h <= img_h:
                    break

            if self.pixel_level:
                if img_channel == 3:
                    value = np.random.uniform(self.value_min, self.value_max, (erase_h, erase_w, img.shape[2]))
                elif img_channel == 2:
                    value = np.random.uniform(self.value_min, self.value_max, (erase_h, erase_w)) 
            else:
                value = np.random.uniform(self.value_min, self.value_max)

            img[y:y + erase_h, x:x + erase_w] = value
            mask[y:y + erase_h, x:x + erase_w] = 255

            # img = Image.fromarray(img.astype(np.uint8))
            # mask = Image.fromarray(mask.astype(np.uint8))
            sample['image'] = Image.fromarray(img.astype(np.uint8))
            sample['label'] = mask

        return sample




class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size # h, w
        self.padding = padding

    def __call__(self, sample):
        img, mask = sample['image'], sample['label']
        w, h = _image_size(img)
        if self.padding > 0 or w < self.size[0] or h < self.size[1]:
            padding = np.maximum(self.padding,np.maximum((self.size[0]-w)//2+5,(self.size[1]-h)//2+5))
            img = _map_image(img, lambda ch: ImageOps.expand(ch, border=padding, fill=0))
            mask = ImageOps.expand(mask, border=padding, fill=255)

        w, h = _image_size(img)
        assert w == mask.width
        assert h == mask.height
        th, tw = self.size # target size
        if w == tw and h == th:
            return sample
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        img = _map_image(img, lambda ch: ch.crop((x1, y1, x1 + tw, y1 + th)))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        sample['image'] = img
        sample['label'] = mask
        return sample


class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        # assert img.width == mask.width
        # assert img.height == mask.height
        w, h = _image_size(img)
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        # y1 = int(round((h - th) / 2.))
        y1 = int(round((h - th) / 2.))
        img = _map_image(img, lambda ch: ch.crop((x1, y1, x1 + tw, y1 + th)))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))

        sample['image'] = img
        sample['label'] = mask
        return sample


class RandomFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = _map_image(img, lambda ch: ch.transpose(Image.FLIP_LEFT_RIGHT))
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = _map_image(img, lambda ch: ch.transpose(Image.FLIP_TOP_BOTTOM))
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        sample['image'] = img
        sample['label'] = mask
        return sample

class RandomHorizontalFlip(object):
    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        if random.random() < 0.5:
            img = _map_image(img, lambda ch: ch.transpose(Image.FLIP_LEFT_RIGHT))
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        sample['image'] = img
        sample['label'] = mask
        return sample


class FixedResize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']

        w, h = _image_size(img)
        assert w == mask.width
        assert h == mask.height
        img = _map_image(img, lambda ch: ch.resize(self.size, Image.BILINEAR))
        mask = mask.resize(self.size, Image.NEAREST)

        sample['image'] = img
        sample['label'] = mask
        sample['img_name'] = name
        return sample


class Scale(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w, h = _image_size(img)
        assert w == mask.width
        assert h == mask.height

        if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
            return sample
        oh, ow = self.size
        img = _map_image(img, lambda ch: ch.resize((ow, oh), Image.BILINEAR))
        mask = mask.resize((ow, oh), Image.NEAREST)

        sample['image'] = img
        sample['label'] = mask
        return sample


class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']
        w0, h0 = _image_size(img)
        assert w0 == mask.width
        assert h0 == mask.height
        for attempt in range(10):
            w_img, h_img = _image_size(img)
            area = w_img * h_img
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= w_img and h <= h_img:
                x1 = random.randint(0, w_img - w)
                y1 = random.randint(0, h_img - h)

                img = _map_image(img, lambda ch: ch.crop((x1, y1, x1 + w, y1 + h)))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))

                img = _map_image(img, lambda ch: ch.resize((self.size, self.size), Image.BILINEAR))
                mask = mask.resize((self.size, self.size), Image.NEAREST)

                sample['image'] = img
                sample['label'] = mask
                sample['img_name'] = name
                return sample

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        sample = crop(scale(sample))
        return sample


class RandomRotate(object):
    def __init__(self, size=512):
        self.degree = random.randint(1, 4) * 90
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        seed = random.random()
        if seed > 0.5:
            rotate_degree = self.degree
            img = _map_image(img, lambda ch: ch.rotate(rotate_degree, Image.BILINEAR, expand=0))
            mask = mask.rotate(rotate_degree, Image.NEAREST, expand=255)
            sample['image'] = img
            sample['label'] = mask
        return sample

class RandomScaleRotate(object):
    def __init__(self, size=512, left=-20, right=20, fillcolor=255):
        self.size = size
        self.left = left
        self.right = right
        self.fillcolor = fillcolor

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']

        seed = random.random()
        if seed > 0.5:
            rotate_degree = random.randint(self.left, self.right)
            img = _map_image(img, lambda ch: ch.rotate(rotate_degree, Image.BILINEAR))
            mask = mask.rotate(rotate_degree, Image.NEAREST, fillcolor=self.fillcolor)

            sample['image'] = img
            sample['label'] = mask
        return sample


class RandomScaleCrop(object):
    def __init__(self, size):
        self.size = size
        # self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w0, h0 = _image_size(img)
        assert w0 == mask.width
        assert h0 == mask.height

        seed = random.random()
        if seed > 0.5:
            w_img, h_img = _image_size(img)
            w = int(random.uniform(1, 1.5) * w_img)
            h = int(random.uniform(1, 1.5) * h_img)

            img = _map_image(img, lambda ch: ch.resize((w, h), Image.BILINEAR))
            mask = mask.resize((w, h), Image.NEAREST)
            sample['image'] = img
            sample['label'] = mask
        return self.crop(sample)


class LesionAwareScaleCrop(object):
    """Scale + crop transform that preserves tiny lesion foreground when present.

    For positive slices, the crop is centered near a foreground pixel from the
    lesion mask, with random jitter. If a foreground-preserving crop cannot be
    produced, it raises an error instead of silently falling back to random crop.

    For negative slices, random crop is expected and explicit.
    """
    def __init__(
        self,
        size,
        positive_crop_prob=1.0,
        center_jitter=0.35,
        scale_prob=0.5,
        scale_min=1.0,
        scale_max=1.5,
        mask_fill=0,
        image_fill=0,
        max_attempts=30,
        strict=True,
    ):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.positive_crop_prob = positive_crop_prob
        self.center_jitter = center_jitter
        self.scale_prob = scale_prob
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.mask_fill = mask_fill
        self.image_fill = image_fill
        self.max_attempts = max_attempts
        self.strict = strict

    def _pad_if_needed(self, img, mask):
        w, h = _image_size(img)
        th, tw = self.size
        pad_x = max(0, tw - w)
        pad_y = max(0, th - h)
        if pad_x == 0 and pad_y == 0:
            return img, mask
        left = pad_x // 2 + 5
        right = pad_x - pad_x // 2 + 5
        top = pad_y // 2 + 5
        bottom = pad_y - pad_y // 2 + 5
        border = (left, top, right, bottom)
        img = _map_image(img, lambda ch: ImageOps.expand(ch, border=border, fill=self.image_fill))
        mask = ImageOps.expand(mask, border=border, fill=self.mask_fill)
        return img, mask

    def _random_crop_coords(self, w, h):
        th, tw = self.size
        if w == tw and h == th:
            return 0, 0
        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        return x1, y1

    def _lesion_crop_coords(self, mask_np, w, h):
        th, tw = self.size
        ys, xs = np.where(mask_np > 0)
        if len(xs) == 0:
            return None
        for _ in range(self.max_attempts):
            pick = random.randrange(len(xs))
            cx = int(xs[pick] + random.uniform(-self.center_jitter, self.center_jitter) * tw)
            cy = int(ys[pick] + random.uniform(-self.center_jitter, self.center_jitter) * th)
            x1 = min(max(cx - tw // 2, 0), w - tw)
            y1 = min(max(cy - th // 2, 0), h - th)
            crop_mask = mask_np[y1:y1 + th, x1:x1 + tw]
            if np.any(crop_mask > 0):
                return x1, y1
        if self.strict:
            raise RuntimeError(
                'LesionAwareScaleCrop could not preserve a lesion after {} attempts. '
                'Check mask/image alignment or reduce crop size/augmentation strength.'.format(self.max_attempts)
            )
        return None

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        w0, h0 = _image_size(img)
        if w0 != mask.width or h0 != mask.height:
            raise ValueError('Image/mask size mismatch before crop: image={}, mask={}'.format((w0, h0), mask.size))

        if random.random() < self.scale_prob:
            w_img, h_img = _image_size(img)
            scale = random.uniform(self.scale_min, self.scale_max)
            w = max(1, int(scale * w_img))
            h = max(1, int(scale * h_img))
            img = _map_image(img, lambda ch: ch.resize((w, h), Image.BILINEAR))
            mask = mask.resize((w, h), Image.NEAREST)

        img, mask = self._pad_if_needed(img, mask)
        w, h = _image_size(img)
        th, tw = self.size
        if w < tw or h < th:
            raise RuntimeError('Padding failed: image size {} is smaller than crop size {}'.format((w, h), (tw, th)))
        if w == tw and h == th:
            sample['image'] = img
            sample['label'] = mask
            return sample

        mask_np = np.asarray(mask)
        has_lesion = np.any(mask_np > 0)
        coords = None
        if has_lesion and random.random() < self.positive_crop_prob:
            coords = self._lesion_crop_coords(mask_np, w, h)
        if coords is None:
            # Explicit negative-slice/random-crop path. Positive slices only reach
            # here when positive_crop_prob < 1.0 by user choice, not as a hidden fallback.
            coords = self._random_crop_coords(w, h)

        x1, y1 = coords
        img = _map_image(img, lambda ch: ch.crop((x1, y1, x1 + tw, y1 + th)))
        mask = mask.crop((x1, y1, x1 + tw, y1 + th))
        sample['image'] = img
        sample['label'] = mask
        return sample


class ResizeImg(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']
        w, h = _image_size(img)
        assert w == mask.width
        assert h == mask.height

        img = _map_image(img, lambda ch: ch.resize((self.size, self.size)))
        # mask = mask.resize((self.size, self.size))

        sample = {'image': img, 'label': mask, 'img_name': name}
        return sample


class Resize(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        name = sample['img_name']
        w, h = _image_size(img)
        assert w == mask.width
        assert h == mask.height

        img = _map_image(img, lambda ch: ch.resize((self.size, self.size)))
        mask = mask.resize((self.size, self.size))

        sample = {'image': img, 'label': mask, 'img_name': name}
        return sample


# class RandomScale(object):
#     def __init__(self, limit):
#         self.limit = limit
#
#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         assert img.width == mask.width
#         assert img.height == mask.height
#
#         scale = random.uniform(self.limit[0], self.limit[1])
#         w = int(scale * img.size[0])
#         h = int(scale * img.size[1])
#
#         img, mask = img.resize((w, h), Image.BILINEAR), mask.resize((w, h), Image.NEAREST)
#
#         return {'image': img, 'label': mask, 'img_name': sample['img_name']}


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img /= 255.0
        img -= self.mean
        img /= self.std

        return {'image': img,
                'label': mask,
                'img_name': sample['img_name']}


class GetBoundary(object):
    def __init__(self, width = 5):
        self.width = width
    def __call__(self, mask):
        cup = mask[:, :, 0]
        disc = mask[:, :, 1]
        dila_cup = ndimage.binary_dilation(cup, iterations=self.width).astype(cup.dtype)
        eros_cup = ndimage.binary_erosion(cup, iterations=self.width).astype(cup.dtype)
        dila_disc= ndimage.binary_dilation(disc, iterations=self.width).astype(disc.dtype)
        eros_disc= ndimage.binary_erosion(disc, iterations=self.width).astype(disc.dtype)
        cup = dila_cup + eros_cup
        disc = dila_disc + eros_disc
        cup[cup==2]=0
        disc[disc==2]=0
        size = mask.shape
        # boundary = np.zers(size[0:2])
        boundary = (cup + disc) > 0
        return boundary.astype(np.uint8)


class Normalize_tf(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
        self.mean = mean
        self.std = std
        self.get_boundary = GetBoundary()

    def __call__(self, sample):
        img = _stack_image(sample['image']).astype(np.float32)
        # __mask = np.array(sample['label']).astype(np.uint8)
        img /= 127.5
        img -= 1.0
        if 'strong_aug' in sample.keys():
            strong = _stack_image(sample['strong_aug']).astype(np.float32)
            strong /= 127.5
            strong -= 1.0
            sample['strong_aug'] = strong
        # _mask = np.zeros([__mask.shape[0], __mask.shape[1]])
        # _mask[__mask > 200] = 255
        # # index = np.where(__mask > 50 and __mask < 201)
        # _mask[(__mask > 50) & (__mask < 201)] = 128
        # _mask[(__mask > 50) & (__mask < 201)] = 128

        # __mask[_mask == 0] = 2
        # __mask[_mask == 255] = 0
        # __mask[_mask == 128] = 1

        # mask = to_multilabel(__mask)
        sample['image'] = img
        # sample['label'] = mask
        return sample


class LesionNormalize(object):
    """MRI-aware per-channel normalization and optional handcrafted channels."""
    def __init__(
        self,
        modalities=None,
        mode='minmax',
        t2w_clip=(0.5, 99.5),
        adc_clip=(1.0, 99.0),
        add_adc_sobel=False,
    ):
        self.modalities = [m.lower() for m in (modalities or ['t2w', 'adc'])]
        self.mode = mode
        self.t2w_clip = t2w_clip
        self.adc_clip = adc_clip
        self.add_adc_sobel = add_adc_sobel

    def _channel_norm(self, channel, modality):
        channel = channel.astype(np.float32)
        roi = channel[np.isfinite(channel)]
        body = roi[roi > 0]
        if body.size > 16:
            roi = body
        if roi.size == 0:
            return np.zeros_like(channel, dtype=np.float32)

        clip = self.adc_clip if 'adc' in modality else self.t2w_clip
        lo, hi = np.percentile(roi, clip)
        channel = np.clip(channel, lo, hi)

        if self.mode == 'legacy':
            return channel / 127.5 - 1.0
        if self.mode == 'zscore':
            norm_pixels = channel[channel > 0]
            if norm_pixels.size <= 16:
                norm_pixels = channel.reshape(-1)
            mu = float(norm_pixels.mean())
            sigma = float(norm_pixels.std())
            channel = (channel - mu) / (sigma + 1e-6)
            return np.clip(channel / 5.0, -1.0, 1.0)

        mn, mx = float(channel.min()), float(channel.max())
        if mx - mn < 1e-6:
            return np.zeros_like(channel, dtype=np.float32)
        return (channel - mn) / (mx - mn) * 2.0 - 1.0

    def _append_sobel(self, arr):
        if not self.add_adc_sobel:
            return arr
        adc_idx = 0
        for i, modality in enumerate(self.modalities):
            if 'adc' in modality:
                adc_idx = i
                break
        adc = arr[..., min(adc_idx, arr.shape[-1] - 1)].astype(np.float32)
        adc_01 = (adc + 1.0) * 0.5
        gx = cv2.Sobel(adc_01, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(adc_01, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.sqrt(gx * gx + gy * gy)
        if grad.max() > 1e-6:
            grad = grad / grad.max()
        grad = grad * 2.0 - 1.0
        return np.concatenate([arr, grad[..., np.newaxis].astype(np.float32)], axis=-1)

    def _normalize_image(self, img):
        arr = _stack_image(img).astype(np.float32)
        out = []
        for c in range(arr.shape[-1]):
            modality = self.modalities[c] if c < len(self.modalities) else 'image'
            out.append(self._channel_norm(arr[..., c], modality))
        out = np.stack(out, axis=-1).astype(np.float32)
        return self._append_sobel(out)

    def __call__(self, sample):
        sample['image'] = self._normalize_image(sample['image'])
        if 'strong_aug' in sample.keys():
            sample['strong_aug'] = self._normalize_image(sample['strong_aug'])
        return sample


class Normalize_cityscapes(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=(0., 0., 0.)):
        self.mean = mean

    def __call__(self, sample):
        img = np.array(sample['image']).astype(np.float32)
        mask = np.array(sample['label']).astype(np.float32)
        img -= self.mean
        img /= 255.0

        return {'image': img,
                'label': mask,
                'img_name': sample['img_name']}

def ToMultiLabel(dc):
    new_dc = np.zeros([3])
    for i in range(new_dc.shape[0]):
        if i == dc:
            new_dc[i] = 1
            return new_dc

def SoftLable(label):
    new_label = label.copy()
    label = list(label)
    index = label.index(1)
    new_label[index] = 0.8+random.random()*0.2
    accelarate = new_label[index]
    for i in range(len(label)):
        if i != index:
            if i == len(label) - 1:
                new_label[i] = 1 - accelarate
            else:
                new_label[i] = random.random()*(1-accelarate)
                accelarate += new_label[i]
    return new_label

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        if _is_sequence_image(sample['image']):
            sample['image'] = _stack_image(sample['image'])
        if len(np.array(sample['image']).shape) == 2:
            sample['image'] = np.expand_dims(np.array(sample['image']).astype(np.float32), 2)  # add channel dimension
        # if len(np.array(sample['label']).shape) == 2:
        #     sample['label'] = np.expand_dims(np.array(sample['label']).astype(np.float32), 2)  # add channel dimension
        img = np.array(sample['image']).astype(np.float32).transpose((2, 0, 1))
        map = np.array(sample['label']).astype(np.uint8)#.transpose((2, 0, 1))
        if 'strong_aug' in sample.keys():
            if _is_sequence_image(sample['strong_aug']):
                sample['strong_aug'] = _stack_image(sample['strong_aug'])
            if len(np.array(sample['strong_aug']).shape) == 2:
                sample['strong_aug'] = np.expand_dims(np.array(sample['strong_aug']).astype(np.float32), 2)  # add channel dimension
            strong = np.array(sample['strong_aug']).astype(np.float32).transpose((2, 0, 1))
            strong = torch.from_numpy(strong).float()
            sample['strong_aug'] = strong
        img = torch.from_numpy(img).float()
        map = torch.from_numpy(map).float()
        sample['image']=img
        sample['label']=map
        # domain_code = torch.from_numpy(SoftLable(ToMultiLabel(sample['dc']))).float()
        # sample['dc'] = domain_code
        return sample
