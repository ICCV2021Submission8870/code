from .resnet_dec import BasicBlock
from .res_shortcut_dec import ResShortCut_D_Dec

def _res_shortcut_D_dec(block, layers, **kwargs):
    model = ResShortCut_D_Dec(block, layers, **kwargs)
    return model

def res_shortcut_decoder(**kwargs):
    return _res_shortcut_D_dec(BasicBlock,[2,3,3,2], **kwargs)
