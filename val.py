import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # model = YOLO('/home/scb123/yolov8/runs/train/efficient_v2-b3/weights/best.pt')
    # model = YOLO('/home/scb123/yolov8/runs/train/efficient_v2-b3_16/weights/best.pt')
    # model = YOLO('/home/scb123/yolov8/runs/train/exp_efficinet_b3_500_4/weights/best.pt')
    # model = YOLO('/home/scb123/yolov8/runs/train/exp_efficinet_b32/weights/best.pt')
    # model = YOLO('/home/scb123/yolov8/runs/train/exp_efficinet_b32_16/weights/best.pt')
    # model = YOLO('/home/scb123/yolov8/runs/train/efficient_v2-b3_500_4/weights/best.pt')
    # model = YOLO('/home/scb123/yolov8/runs/train/exp_edgenext/weights/best.pt')
    # model = YOLO("/home/scb123/yolov8/runs/train/exp_mobilenetv3/weights/best.pt")
    # model = YOLO("/home/scb123/yolov8/runs/train/exp_ghostnet-160_500_4/weights/best.pt")
    model = YOLO("/home/scb123/yolov8/runs/train/efficient_v2-b3_16/weights/best.pt")
    model.val(data='dataset/data.yaml',
              split='test',
              batch=1,
              save_json=True, # if you need to cal coco metrice
              project='runs/val',
            #   device=1,
            #   name='exp-efficient_v2-b3',
            #   name='exp-efficient_v2-b3_16',
              # name='exp-efficinet_b3_500_4',
            #   name='exp-exp_efficinet_b32_16',
              # name='efficient_v2-b3_500_4',
            #   name='exp-exp_efficinet_b32',
            #   name='exp-exp_edgenext',
            #   name ="exp_mobilenetv3",
              # name = "exp_ghostnet-160_500_4",
              name = "exp_efficient_v2-b3_16",
              seed=0,
              deterministic=True,
              )