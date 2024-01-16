# from ultralytics.nn.models.efficientnet import efficientnet_b4, efficientnet_b3, efficientnetv2_rw_t, tf_efficientnetv2_b3
from ultralytics.nn.models.efficientnet_v2 import  tf_efficientnetv2_b3, tf_efficientnetv2_b3_rfa
from ultralytics.nn.models.RFAConv import  RFAConv, DWRFAConv
from ultralytics.nn.modules.block import  SPPF, ASPPF


from thop import profile
import torch

if __name__ == '__main__':
    x = torch.randn((2,232,32, 32)).cuda()
    # Flops: 0.6742
    # G
    # params参数量: 0.0008
    # M
    # model = DWRFAConv(3, 16, 3).cuda()
    # model = RFAConv(3, 16, 3).cuda()
    model = ASPPF(232,1,1).cuda()
    # model = SPPF(512,512).cuda()
    # print(model.channel)
    y = model(x)
    # print(y.shape)
    # for y_ in y:
    #     print(y_.shape)

    Flops, params = profile(model, inputs=(x,))  # macs
    print('Flops: % .4fG' % (Flops / 1000000000))  # 计算量
    print('params参数量: % .8fM' % (params / 1000000))  # 参数量：等价与上面的summary输出的Total params值



