import subprocess as sp
from pathlib import Path


recognition_model_url = "https://huggingface.co/astaileyyoung/facenet-onnx/resolve/main/facenet.onnx"
detection_model_url = "https://huggingface.co/astaileyyoung/yolov11m-face-onnx/resolve/main/yolov11m-face-dynamic.onnx"
detection_trt = '/app/models/yolov11m-face-dynamic.trt'
recognition_trt = '/app/models/facenet-dynamic.trt'


def prepare_models(detection_model='/app/models/yolov11m-face-dynamic.onnx',
                recognition_model='/app/models/facenet.onnx'):
    if not Path('/app/models').exists():
        Path('/app/models').mkdir()

    if not Path(detection_model).exists():
        command = ["wget", detection_model_url]
        sp.run(command)
        command = ["mv", "yolov11m-face-dynamic.onnx", detection_model]
        sp.run(command)
    
    if not Path(recognition_model).exists():
        command = ["wget", recognition_model_url]
        sp.run(command)
        command = ["mv", "facenet.onnx", recognition_model]
        sp.run(command)

    if not Path(detection_trt).exists():
        command = [
            'trtexec',
            f'--onnx={detection_model}',
            '--minShapes=images:1x3x640x640',
            '--optShapes=images:1x3x640x640',
            '--maxShapes=images:128x3x640x640',
            f'--saveEngine={detection_trt}',
            '--int8'
        ]
        sp.run(command)
    
    if not Path(recognition_trt).exists():
        command = [
            'trtexec',
            f'--onnx={recognition_model}',
            '--minShapes=input:1x160x160x3',
            '--optShapes=input:1x160x160x3',
            '--maxShapes=input:128x160x160x3',
            f'--saveEngine={recognition_trt}'
        ]
        sp.run(command)


prepare_models()