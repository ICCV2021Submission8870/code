import torch.nn as nn
import torch
from network.HRRN import encoders,decoders

class Main_Net(nn.Module):

    def __init__(self):

        super(Main_Net, self).__init__()

        self.encoder = encoders.res_shortcut_encoder()
        self.decoder = decoders.res_shortcut_decoder()

    def forward(self,image,trimap):

        inp = torch.cat((image,trimap),1)
        embedding,mid_fea = self.encoder(inp)
        high_sal = self.decoder(embedding,mid_fea)
        return high_sal

def HRRN_Net( checkpoint_path,device_id ):

    model = Main_Net()
    device = torch.device("cuda:%d"%device_id if torch.cuda.is_available() else "cpu")
    print("you have loaded HRRN checkpoint file:", checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model



