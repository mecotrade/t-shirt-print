import numpy as np
import os
import datetime
import json
import inspect

from torchvision.transforms import ToPILImage


class ImageSaver:

    def __init__(self, print_folder, filename='print.png', step_pattern='frame_%05d.jpg', keep_last_frame=True):
        self.run_id = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        self.print_folder = os.path.join(print_folder, self.run_id)
        self.step = 0
        os.makedirs(self.print_folder)
        self.filename = filename
        self.step_pattern = step_pattern
        self.keep_last_frame = keep_last_frame
        self.frames = []

        self.array_to_PIL = ToPILImage()

    def from_array(self, image):
        return self.array_to_PIL((image * 255.0).astype(np.uint8))

    def to_step(self, image):
        while self.keep_last_frame and self.frames:
            os.remove(self.frames.pop(0))
        self.step += 1
        frame_file = os.path.join(self.print_folder, self.step_pattern % self.step)
        self.from_array(image).save(frame_file)
        self.frames += [frame_file]

    def clean_up(self):
        while self.frames:
            os.remove(self.frames.pop(0))

    def to_result(self, image):
        self.from_array(image).save(os.path.join(self.print_folder, self.filename))

    def to_config(self, config):
        with open(os.path.join(self.print_folder, 'config.json'), 'w') as file:
            json.dump(config, file, indent=4)


class Painter:

    def __init__(self, pipe, prompt, negative_prompt=None, num_inference_steps=50, sd_height=512, sd_width=512,
                 guidance_scale=7.5, strength=1.):
        args, _, _, defaults = inspect.getargvalues(inspect.stack()[0][0])
        self.config = {a: v for a, v in defaults.items() if a in args and a not in ['self', 'pipe']}

        self.pipe = pipe
        self.prompt = prompt
        self.negative_prompt = negative_prompt
        self.num_inference_steps = num_inference_steps
        self.sd_height = sd_height
        self.sd_width = sd_width
        self.guidance_scale = guidance_scale
        self.strength = strength

        self.array_to_PIL = ToPILImage()

    def paint(self, image, mask):
        result = self.pipe(prompt=self.prompt,
                           image=self.from_array(image),
                           mask_image=self.from_array(mask),
                           num_inference_steps=self.num_inference_steps,
                           height=self.sd_height,
                           width=self.sd_width,
                           guidance_scale=self.guidance_scale,
                           strength=self.strength,
                           negative_prompt=self.negative_prompt)
        return self.to_array(result.images[0])

    def from_array(self, image):
        return self.array_to_PIL((image * 255.0).astype(np.uint8))

    def to_array(self, image):
        return np.array(image) / 255.0
