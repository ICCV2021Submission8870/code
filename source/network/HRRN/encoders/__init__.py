from .resnet_enc import BasicBlock
from .res_shortcut_enc import ResShortCut_D

def _res_shortcut_D(block, layers, **kwargs):

    model = ResShortCut_D(block, layers, **kwargs)
    return model

def res_shortcut_encoder(**kwargs):
    return _res_shortcut_D(BasicBlock,[3, 4, 4, 2], **kwargs)


