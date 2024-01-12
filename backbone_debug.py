# from ultralytics.nn.models.efficientnet import efficientnet_b4, efficientnet_b3, efficientnetv2_rw_t, tf_efficientnetv2_b3
from ultralytics.nn.models.efficientnet_v2 import  tf_efficientnetv2_b3, tf_efficientnetv2_b3_rfa
from ultralytics.nn.models.RFAConv import  RFAConv
from ultralytics.nn.modules.block import  SPPF


from thop import profile
import torch

if __name__ == '__main__':
    x = torch.randn((2,3,640, 640)).cuda()
    model = tf_efficientnetv2_b3().cuda()
    # print(model.channel)
    y = model(x)
    # print(y.shape)
    # for y_ in y:
    #     print(y_.shape)

    Flops, params = profile(model, inputs=(x,))  # macs
    print('Flops: % .4fG' % (Flops / 1000000000))  # 计算量
    print('params参数量: % .4fM' % (params / 1000000))  # 参数量：等价与上面的summary输出的Total params值



