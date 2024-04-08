import numpy as np
import inspect


class Artist:

    def __init__(self, width, height, stride=128, mask_overlap=128, sd_width=512, sd_height=512):

        args, _, _, defaults = inspect.getargvalues(inspect.stack()[0][0])
        self.config = {a: v for a, v in defaults.items() if a in args and a not in ['self', 'image']}

        self.width = width
        self.height = height
        self.stride = stride
        self.mask_overlap = mask_overlap
        self.sd_width = sd_width
        self.sd_height = sd_height

        self.image = np.ones([self.height, self.width, 3])
        self.x = 0
        self.y = 0

    def reset_x(self):
        self.x = 0

    def reset_y(self):
        self.y = 0

    def sequence_x(self, step):
        num_steps = (self.width - self.sd_width) // step + (0 if (self.width - self.sd_width) % step == 0 else 1)
        for n in range(num_steps):
            self.x = n * step
            yield self.x

    def sequence_y(self, step):
        num_steps = (self.height - self.sd_height) // step + (0 if (self.height - self.sd_height) % step == 0 else 1)
        for n in range(num_steps):
            self.y = n * step
            yield self.y

    def get_patch(self, offset_x, offset_y, image=None):
        image = image if image is not None else self.image
        return image[self.y + offset_y:self.y + offset_y + self.sd_height,
                     self.x + offset_x:self.x + offset_x + self.sd_width, :]

    def safe_x(self, x, width):
        return x - max(0, self.x + x + width - self.width)

    def safe_y(self, y, height):
        return y - max(0, self.y + y + height - self.height)

    def get_mask(self, offset_x, offset_y, width, height):
        mask = np.zeros([self.sd_height, self.sd_width, 3])
        mask[offset_y:offset_y + height, offset_x:offset_x + width, :] = 1.0
        return mask

    def apply_patch(self, patch, offset_x, offset_y, image=None):
        image = image if image is not None else self.image
        image[self.y + offset_y:self.y + offset_y + patch.shape[0],
              self.x + offset_x:self.x + offset_x + patch.shape[1], :] = patch

    def build(self, painter, saver, verbose=True):
        raise NotImplementedError


class LineWiseArtist(Artist):

    def get_seed_patch(self):
        return self.get_patch(0, 0)

    def get_right_band_patch(self):
        return self.get_patch(self.safe_x(self.stride, self.sd_width), 0)

    def get_bottom_band_patch(self):
        return self.get_patch(0, self.safe_y(self.stride, self.sd_height))

    def get_right2_bottom1_corner_patch(self):
        return self.get_patch(self.safe_x(self.sd_width - 2 * self.stride, self.sd_width),
                              self.safe_y(self.stride, self.sd_height))

    def get_right1_bottom2_corner_patch(self):
        return self.get_patch(self.safe_x(self.stride, self.sd_width),
                              self.safe_y(self.sd_height - 2 * self.stride, self.sd_height))

    def get_seed_mask(self):
        return self.get_mask(0, 0, self.sd_width, self.sd_height)

    def get_right_band_mask(self):
        return self.get_mask(self.sd_width - self.stride - self.mask_overlap, 0, self.stride + self.mask_overlap,
                             self.sd_height)

    def get_bottom_band_mask(self):
        return self.get_mask(0, self.sd_height - self.stride - self.mask_overlap, self.sd_width,
                             self.stride + self.mask_overlap)

    def get_right2_bottom1_corner_mask(self):
        return self.get_mask(self.stride, self.sd_height - self.stride - self.mask_overlap, self.sd_width - self.stride,
                             self.stride + self.mask_overlap)

    def get_right1_bottom2_corner_mask(self):
        return self.get_mask(self.sd_width - self.stride - self.mask_overlap, self.stride, self.stride + self.mask_overlap,
                             self.sd_height - self.stride)

    def apply_seed_patch(self, patch):
        self.apply_patch(patch, 0, 0)

    def apply_right_band_patch(self, patch):
        self.apply_patch(patch[:, self.sd_width - self.stride - self.mask_overlap:, :],
                         self.safe_x(self.sd_width - self.mask_overlap, self.stride + self.mask_overlap), 0)

    def apply_bottom_band_patch(self, patch):
        self.apply_patch(patch[self.sd_height - self.stride - self.mask_overlap:, :, :], 0,
                         self.safe_y(self.sd_height - self.mask_overlap, self.stride + self.mask_overlap))

    def apply_right2_bottom1_corner_patch(self, patch):
        self.apply_patch(patch[self.sd_height - self.stride - self.mask_overlap:, self.stride:, :],
                         self.safe_x(self.sd_width - self.stride, self.sd_width - self.stride),
                         self.safe_y(self.sd_height - self.mask_overlap, self.stride + self.mask_overlap))

    def apply_right1_bottom2_corner_patch(self, patch):
        self.apply_patch(patch[self.stride:, self.sd_width - self.stride - self.mask_overlap:, :],
                         self.safe_x(self.sd_width - self.mask_overlap, self.stride + self.mask_overlap),
                         self.safe_y(self.sd_height - self.stride, self.sd_height - self.stride))

    def build_seed_image(self, painter, saver=None):

        patch = self.get_seed_patch()
        mask = self.get_seed_mask()
        result = painter.paint(patch, mask)
        self.apply_seed_patch(result)

        if saver is not None:
            saver.to_step(self.image)

    def build_right_band(self, painter, saver=None):

        patch = self.get_right_band_patch()
        mask = self.get_right_band_mask()
        result = painter.paint(patch, mask)
        self.apply_right_band_patch(result)

        if saver is not None:
            saver.to_step(self.image)

    def build_bottom_band(self, painter, saver=None):

        patch = self.get_bottom_band_patch()
        mask = self.get_bottom_band_mask()
        result = painter.paint(patch, mask)
        self.apply_bottom_band_patch(result)

        if saver is not None:
            saver.to_step(self.image)

    def build_right2_bottom1_corner(self, painter, saver=None):

        patch = self.get_right2_bottom1_corner_patch()
        mask = self.get_right2_bottom1_corner_mask()
        result = painter.paint(patch, mask)
        self.apply_right2_bottom1_corner_patch(result)

        if saver is not None:
            saver.to_step(self.image)

    def build_right1_bottom2_corner(self, painter, saver=None):

        patch = self.get_right1_bottom2_corner_patch()
        mask = self.get_right1_bottom2_corner_mask()
        result = painter.paint(patch, mask)
        self.apply_right1_bottom2_corner_patch(result)

        if saver is not None:
            saver.to_step(self.image)

    def build_topmost_row(self, painter, saver=None):
        self.build_seed_image(painter, saver)
        for _ in self.sequence_x(self.stride):
            self.build_right_band(painter, saver)

    def build_leftmost_column(self, painter, saver=None):
        self.build_seed_image(painter, saver)
        for _ in self.sequence_y(self.stride):
            self.build_bottom_band(painter, saver)

    def build_right_column(self, painter, saver=None):
        self.reset_y()
        self.build_right_band(painter, saver)
        for _ in self.sequence_y(2 * self.stride):
            self.build_right1_bottom2_corner(painter, saver)

    def build_bottom_row(self, painter, saver=None):
        self.reset_x()
        self.build_bottom_band(painter, saver)
        for _ in self.sequence_x(2 * self.stride):
            self.build_right2_bottom1_corner(painter, saver)


class LeftToRightArtist(LineWiseArtist):

    def build(self, painter, saver, verbose=True):

        saver.to_config({'run_id': saver.run_id, 'builder': self.config, 'painter': painter.config})

        self.reset_x()
        self.reset_y()

        self.build_leftmost_column(painter, saver if verbose else None)

        for _ in self.sequence_x(self.stride):
            self.build_right_column(painter, saver if verbose else None)

        saver.to_result(self.image)


class TopToBottomArtist(LineWiseArtist):

    def build(self, painter, saver, verbose=True):
        saver.to_config({'run_id': saver.run_id, 'builder': self.config, 'painter': painter.config})

        self.reset_x()
        self.reset_y()

        self.build_topmost_row(painter, saver if verbose else None)

        for _ in self.sequence_y(self.stride):
            self.build_bottom_row(painter, saver if verbose else None)

        saver.to_result(self.image)

        return painter.from_array(self.image)
