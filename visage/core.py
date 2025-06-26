#!/usr/bin/env python3

import os 
import csv
import uuid
import signal
import logging
import subprocess as sp
from pathlib import Path 
from argparse import ArgumentParser 


logger = logging.getLogger("visage")


def load_docker_image(image='astaileyyoung/visage'):
    command = [
        'docker',
        'images',
        '-q',
        image
    ]
    result = sp.run(command, capture_output=True, text=True)
    if result.stdout.strip():
        logger.debug('Docker image already exists. Skipping download.')
        return 
    else:
        logger.info('Downloading docker image')
        pull_result = sp.run(['docker', 'pull', image], stdout=sp.DEVNULL, stderr=sp.DEVNULL)
        pull_result.check_returncode()
        logger.info(f'Docker image {image} pulled successfully.')


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
    sp.run(command, check=True, stdout=sp.DEVNULL, stderr=sp.DEVNULL)
    return model_dir


def run_docker_image(src, dst, image, frameskip, log_level, show, model_dir):
    container_name = f"visage_{uuid.uuid4().hex[:8]}"

    src = Path(src).absolute().resolve()
    dst = Path(dst).absolute().resolve() if dst else None
    model_dir = Path(model_dir).absolute()

    mount_point_src = src.parent
    mount_point_dst = dst.parent if dst else None
    model_mount_point = f"{str(model_dir)}:/app/models"
    app_src = str(Path(mount_point_src.parts[-1]).joinpath(src.name)) 
    app_mount_src = f"{str(mount_point_src)}:/app/{mount_point_src.parts[-1]}"
    app_mount_dst = f"{str(mount_point_dst)}:/app/{mount_point_dst.parts[-1]}"
    command = [
        "docker",
        "run",
        "--rm",
        "--name", container_name,
        "--gpus",
        "all",
        "-e",
        "NVIDIA_DRIVER_CAPABILITIES=all",
        "-v", model_mount_point,
        "-v", app_mount_src
    ]
    if dst:
        command.extend([
            "-v",
            app_mount_dst
        ])
    command.extend([
        image,
        "/app/bin/visage",
        app_src
    ])
    if dst:
        command.append(str(Path(mount_point_dst.parts[-1]).joinpath(dst.name)))
    else:
        command.append("dummy")
    
    command.append(str(frameskip))
    command.append(log_level)
    if show:
        command.append('-show')

    try:
        proc = sp.Popen(command, preexec_fn=os.setsid)
        proc.wait()
    except KeyboardInterrupt:
        logger.info('Exiting docker.')
        sp.run(['docker', 'stop', '-t', '1', container_name], stdout=sp.DEVNULL, stderr=sp.DEVNULL, timeout=30)
        proc.terminate()
        proc.wait()
    
    return container_name


def run_visage(src, dst, image, frameskip, log_level, show, model_dir):  
    levels = {
        "debug": logging.DEBUG,
        "info": logging.INFO,
        "warning": logging.WARNING,
        "error": logging.ERROR,
        "critical": logging.CRITICAL
    }
    # Fallback to INFO if log_level is not recognized
    level = levels.get(str(log_level).lower(), logging.INFO)
    logger.setLevel(level)

    src = Path(src)
    dst = Path(dst) if dst else None
    if not src.exists():
        logger.error(f'{str(src)} does not exist. Exiting.')
        exit()
    elif src.suffix not in ('.mp4', '.mkv', '.m4v', '.avi', '.mov'):
        logger.warning(f'{src.suffix} is not a valid file extension')

    if dst is not None and not dst.parent.exists():
        dst.parent.mkdir(parents=True)

    load_docker_image()
    model_dir = load_models(image=image, model_dir=model_dir)
    container_name = run_docker_image(src, dst, image, frameskip, log_level, show, model_dir)
    data = []
    if dst.exists():
        with open(dst, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                data.append(row)
        dst.unlink()
    return data, container_name


def main():
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
    log_level = levels.get(args.log_level.lower(), 20)
    handler = logging.StreamHandler()
    handler.setLevel(log_level)
    formatter = logging.Formatter('[%(levelname)s]: %(message)s')
    handler.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(handler) 

    run_visage(src=args.src, 
               dst=args.dst,
               image=args.image,
               frameskip=args.frameskip,
               log_level=args.log_level,
               show=args.show,
               model_dir=args.model_dir
               )


if __name__ == '__main__':
    main()
