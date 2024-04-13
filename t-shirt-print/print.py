import os
import math
import torch
import argparse
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from RealESRGAN import RealESRGAN

from painters import ImageSaver, Painter
from artists import LeftToRightArtist, TopToBottomArtist


REAL_ESR_GAN_WEIGHTS = {
    2: 'realesrgan/RealESRGAN_x2.pth',
    4: 'realesrgan/RealESRGAN_x4.pth',
    8: 'realesrgan/RealESRGAN_x8.pth'
}

RESOLUTIONS = {
    'SVGA': (800, 600),
    'WSVGA': (1024, 600),
    'XGA': (1024, 768),
    'HD': (1280, 720),
    'WXGA': (1280, 768),
    'SXGA': (1280, 1024),
    'Full-HD': (1920, 1080),
    'Ultra-HD': (3840, 2160),
    '4K-UHD': (4096, 2160),
    'T-shirt': (4800, 3301)
}


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--prompt',
                        help='The prompt')
    parser.add_argument('-np', '--negative-prompt',
                        default='picture border, margin, image caption, image separator, '
                                'narrow band separating images, watermark, text across picture, '
                                'low resolution, blurry',
                        help='The prompt')
    parser.add_argument('-r', '--resolution', choices=RESOLUTIONS.keys(),
                        help='Image resolution')
    parser.add_argument('-s', '--shape', type=int, nargs=2,
                        help='Image shape: width height')
    parser.add_argument('-a', '--artist', choices=['top-to-bottom', 'left-to-right'],
                        default='top-to-bottom', help='Image upscale factor')
    parser.add_argument('-mi', '--model-id', default='stabilityai/stable-diffusion-2-inpainting',
                        help='Model id')
    parser.add_argument('-mt', '--model-type', choices=['stable-diffusion', 'stable-diffusion-xl'],
                        default='stable-diffusion', help='Model')
    parser.add_argument('-ms', '--mask-stride', default=128, type=int,
                        help='Mask stride in pixels while creating large image')
    parser.add_argument('-mo', '--mask-overlap', default=256, type=int,
                        help='Mask overlap in pixels while creating large image')
    parser.add_argument('-nis', '--num-inference-steps', default=50, type=int,
                        help='Mask overlap in pixels while creating large image')
    parser.add_argument('-gs', '--guidance-scale', default=7.5, type=float,
                        help='Guidance scale for inpainting')
    parser.add_argument('-is', '--inpainting-strength', default=1., type=float,
                        help='Inpainting strength')
    parser.add_argument('-u', '--upscale', choices=[2, 4, 8], type=int,
                        help='Image upscale factor')
    parser.add_argument('-t', '--token',
                        help='Huggingface token')
    parser.add_argument('-cf', '--cache-folder',
                        help='Huggingface cache folder')
    parser.add_argument('-pf', '--print-folder', default='prints',
                        help='Print folder')
    args = parser.parse_args()

    if args.resolution is not None:
        width, height = RESOLUTIONS[args.resolution]
    elif args.shape is not None:
        width, height = args.shape
    else:
        raise ValueError('Either resolution or shape must be set')

    huggingface_token = args.token or os.environ['HUGGINGFACE_AUTH_TOKEN']
    huggingface_cache = args.cache_folder or os.environ.get('HUGGINGFACE_HUB_CACHE', None) or 'huggingface'

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if args.model_type == 'stable-diffusion':
        inpainting = StableDiffusionInpaintPipeline.from_pretrained(args.model_id,
                                                                    token=huggingface_token,
                                                                    variant='fp16', torch_dtype=torch.float16,
                                                                    cache_dir=huggingface_cache)
    elif args.model_type == 'stable-diffusion-xl':
        inpainting = StableDiffusionXLInpaintPipeline.from_pretrained(args.model_id,
                                                                      token=huggingface_token,
                                                                      variant='fp16', torch_dtype=torch.float16,
                                                                      cache_dir=huggingface_cache)
    else:
        raise ValueError(f'Model type {args.model_type} not supported')

    inpainting.scheduler = DPMSolverMultistepScheduler.from_config(inpainting.scheduler.config)
    inpainting.to(device=device)

    if args.upscale:
        upscaler = RealESRGAN(device, scale=args.upscale)
        upscaler.load_weights(REAL_ESR_GAN_WEIGHTS[args.upscale], download=True)
        width = math.ceil(width / args.upscale)
        height = math.ceil(height / args.upscale)
    else:
        upscaler = None

    sd_width = min(width, 512)
    sd_height = min(height, 512)

    painter = Painter(inpainting,
                      prompt=args.prompt,
                      num_inference_steps=args.num_inference_steps,
                      sd_width=sd_width,
                      sd_height=sd_height,
                      strength=args.inpainting_strength,
                      negative_prompt=args.negative_prompt)
    saver = ImageSaver(args.print_folder)

    print(f'Image will be saved to {saver.print_folder}')

    if args.artist == 'top-to-bottom':
        artist = TopToBottomArtist(width, height, stride=args.mask_stride, mask_overlap=args.mask_overlap,
                                   sd_width=sd_width, sd_height=sd_height)
    elif args.artist == 'left-to-right':
        artist = LeftToRightArtist(width, height, stride=args.mask_shift, mask_overlap=args.mask_overlap,
                                   sd_width=sd_width, sd_height=sd_height)
    else:
        raise ValueError(f'Artist {args.artist} not supported')

    image = artist.build(painter, saver)

    if upscaler is not None:
        upscaler.predict(image).save(os.path.join(saver.print_folder, 'large.png'))

    saver.clean_up()
