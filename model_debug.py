from ultralytics.nn.tasks import DetectionModel
import torch
from thop import profile


if __name__ == "__main__":
    # flag = torch.cuda.is_available()
    # if flag:
    #     print("CUDA可使用")
    # else:
    #     print("CUDA不可用")
    #
    # ngpu = 1
    # # Decide which device we want to run on
    # device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")
    # print("驱动为：", device)
    # print("GPU型号： ", torch.cuda.get_device_name(0))

    x = torch.randn(1, 3, 640, 640).cuda()
    model = DetectionModel(
        # r"/home/scb123/yolov8/ultralytics/cfg/models/v8/yolov8-edgenext.yaml", nc=2,
        # "/home/scb123/yolov8/ultralytics/cfg/models/v8/yolov8-mobilenetv3_large.yaml", nc=2
        #  "/home/scb123/yolov8/ultralytics/cfg/models/v8/yolov8-efficientb3.yaml", nc=2,
        # "/home/scb123/yolov8/ultralytics/cfg/models/v8/yolov8-tiny_vit.yaml", nc=2,
        # "/home/scb123/yolov8/ultralytics/cfg/models/v8/yolov8-ghostnet-160.yaml", nc=2,
        # r"F:\PythonPro\DaiLian\yolov8-self\ultralytics\cfg\models\v8\yolov8-efficient_v2-b3.yaml", nc=2,
        # r"F:\PythonPro\DaiLian\yolov8-self\ultralytics\cfg\models\v8\yolov8-efficient_v2-b3-rfa.yaml", nc=2,
        # r"F:\PythonPro\DaiLian\yolov8-self\ultralytics\cfg\models\v8\yolov8-efficient_v2-b3-asppf.yaml", nc=2,
        # r"F:\PythonPro\DaiLian\yolov8-self\ultralytics\cfg\models\v8\yolov8-efficient_v2-b3-dwrfa.yaml", nc=2,
        r"F:\PythonPro\DaiLian\yolov8-self\ultralytics\cfg\models\v8\yolov8-efficient_v2-b3-all.yaml", nc=2,
    ).cuda()
    # model = model.cuda()
    # model.eval()
    output = model(x)

    # print(output[0].shape)
    for y in output:
        print(y.shape)

    Flops, params = profile(model, inputs=(x,))  # macs
    print('Flops: % .4fG' % (Flops / 1000000000))  # 计算量
    print('params参数量: % .4fM' % (params / 1000000))  # 参数量：等价与上面的summary输出的Total params值
