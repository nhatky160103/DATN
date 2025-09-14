# from .iresnet import iresnet18, iresnet34, iresnet50, iresnet100, iresnet200
# from .iresnet_lite import iresnet18_lite, iresnet34_lite, iresnet50_lite, iresnet100_lite, iresnet200_lite

# def get_model(name, **kwargs):
#     # resnet
#     if name == "r18":
#         return iresnet18(False, **kwargs)
#     elif name == "r34":
#         return iresnet34(False, **kwargs)
#     elif name == "r50":
#         return iresnet50(False, **kwargs)
#     elif name == "r100":
#         return iresnet100(False, **kwargs)
#     elif name == "r200":
#         return iresnet200(False, **kwargs)
#     if name == "r18_lite":
#         return iresnet18_lite(False, **kwargs)
#     elif name == "r34_lite":
#         return iresnet34_lite(False, **kwargs)
#     elif name == "r50_lite":
#         return iresnet50_lite(False, **kwargs)
#     elif name == "r100_lite":
#         return iresnet100_lite(False, **kwargs)
#     elif name == "r200_lite":
#         return iresnet200_lite(False, **kwargs)
   
#     else:
#         raise ValueError()