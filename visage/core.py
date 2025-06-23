#!/usr/bin/env python3

import os
import logging
import subprocess as sp
from pathlib import Path 
from argparse import ArgumentParser 


def load_docker_image(image='astaileyyoung/visage'):
    command = [
        'docker',
        'images',
        '-q',
        image
    ]
    result = sp.run(command, capture_output=True, text=True)
    if result.stdout.strip():
        logging.info('Docker image already exists. Skipping download.')
        return 
    else:
        logging.info('Downloading docker image')
        pull_result = sp.run(['docker', 'pull', image])
        pull_result.check_returncode()
        logging.info(f'Docker image {image} pulled successfully.')


def load_models(image='astaileyyoung/visage', model_dir=None):
    if model_dir is None:
        model_dir = Path.home() / '.visage' / 'models'
    else:
        model_dir = Path(model_dir).absolute()
    model_dir.mkdir(exist_ok=True, parents=True)

    command = [
        'docker', 'run', '--rm',
        '--gpus', 'all',
        '-e', 'NVIDIA_DRIVER_CAPABILITIES=all',
        '-v', f'{model_dir}:/app/models',
        image,
        'python', '/app/scripts/prepare_models.py'
    ]
    sp.run(command, check=True)
    return model_dir


def run_docker_image(src, dst, image, frameskip, log_level, show, model_dir):
    src = Path(src).absolute().resolve()
    dst = Path(dst).absolute().resolve() if dst else None
    model_dir = Path(model_dir).absolute()
    detection_model_path = model_dir / 'yolov11m-face-dynamic.trt'
    recognition_model_path = model_dir / 'facenet-dynamic.trt'

    mount_point_src = src.parent
    mount_point_dst = dst.parent if dst else None
    command = [
        "docker",
        "run",
        "--rm",
        "--gpus",
        "all",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=all",
        "-v", f"{str(model_dir)}:/app/models",
        "-v", f"{str(mount_point_src)}:/app/{mount_point_src.parts[-1]}"
    ]
    if dst:
        command.extend([
            "-v",
            f"{str(mount_point_dst)}:/app/{mount_point_dst.parts[-1]}"
        ])
    command.extend([
        image,
        "/app/bin/visage",
        str(Path(mount_point_src.parts[-1]).joinpath(src.name))
    ])
    if dst:
        command.append(str(Path(mount_point_dst.parts[-1]).joinpath(dst.name)))
    else:
        command.append("dummy")
    
    command.append(str(frameskip))
    command.append(log_level)
    if show:
        command.append('-show')

    sp.run(command)


def run_visage(src, 
               dst,
               image,
               frameskip,
               log_level,
               show,
               model_dir=None):
    if not Path(src).exists():
        logging.error(f'{src} does not exist. Exiting.')
        exit()
    elif Path(src).suffix not in ('.mp4', '.mkv', '.m4v', '.avi', '.mov'):
        logging.warning(f'{Path(src).suffix} is not a valid file extension')

    if dst is not None and not Path(dst).parent.exists():
        Path(dst).parent.mkdir(parents=True)

    load_docker_image()
    model_dir = load_models(image=image, model_dir=model_dir)
    run_docker_image(src, dst, image, frameskip, log_level, show, model_dir)


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('src')
    ap.add_argument('--dst', default=None, type=str)
    ap.add_argument('--image', default='astaileyyoung/visage')
    ap.add_argument('--model_dir', default=None, type=str)
    ap.add_argument('--frameskip', default=1)
    ap.add_argument('--show', action='store_true')
    ap.add_argument('--log_level', default="info", type=str)
    args = ap.parse_args()

    levels = {
        "debug": 10,
        "info": 20,
        "warning": 30,
        "error": 40
    }

    log_level = levels[args.log_level]
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=log_level,
        datefmt='%y-%m-%d_%H:%M:%S'
    )
    run_visage(args.src,
               args.dst,
               args.image,
               args.frameskip,
               args.log_level,
               args.show,
               model_dir=args.model_dir)
