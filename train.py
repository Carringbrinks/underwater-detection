import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('/home/scb123/yolov8/ultralytics/cfg/models/v8/yolov8s.yaml')
    # model = YOLO('/home/scb123/yolov8/ultralytics/cfg/models/v8/yolov8-mobilenetv3_large.yaml')
    # model = YOLO('/home/scb123/yolov8/ultralytics/cfg/models/v8/yolov8-edgenext.yaml')
    # model = YOLO('/home/scb123/yolov8/ultralytics/cfg/models/v8/yolov8-efficientb3.yaml')
    # model = YOLO('/home/scb123/yolov8/ultralytics/cfg/models/v8/yolov8-tiny_vit.yaml')
    # model = YOLO("/home/scb123/yolov8/ultralytics/cfg/models/v8/yolov8-ghostnet-160.yaml")
    model = YOLO("/home/scb123/yolov8/ultralytics/cfg/models/v8/yolov8-efficient_v2-b3.yaml")
    # model = YOLO(r"F:\PythonPro\DaiLian\yolov8-self\ultralytics\cfg\models\v8\yolov8-efficient_v2-b3-rfa.yaml")
    model = YOLO(r"F:\PythonPro\DaiLian\yolov8-self\ultralytics\cfg\models\v8\yolov8-efficient_v2-b3-sppf-rfa.yaml")
    # model = YOLO(r"F:\PythonPro\DaiLian\yolov8-self\ultralytics\cfg\models\v8\yolov8-efficient_v2-b3-all.yaml")
    # model.load('yolov8n.pt') # loading pretrain weights
    model.train(data='/home/scb123/yolov8/dataset/data.yaml',
                cache=False,
                imgsz=640,
                epochs=300,
                batch=16,
                close_mosaic=10,
                workers=4,
                device='1',
                optimizer='SGD', # using SGD
                lr0=0.01,
                # resume='', # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                project='runs/train',
                # name='exp_mobilenetv3',
                # name='exp_edgenext',
                name='exp-efficient_v2-b3-sppf-rfa',
                # name='exp_tiny_vit',
                seed=0,
                deterministic=True,
                patience=30,
                )