import torch
import torch.nn as nn
import torch.nn.functional as F

from network.LRSCN.vgg import B2_VGG

class GCN(nn.Module):

    def __init__(self, c, out_c, k=(5,5) ):

        super(GCN, self).__init__()

        self.conv_l1 = nn.Conv2d(c, out_c, kernel_size=(k[0], 1), padding = [ int((k[0]-1)/2),0] )
        self.conv_l2 = nn.Conv2d(out_c, out_c, kernel_size=(1, k[0]), padding = [0,int((k[0]-1)/2)])

        self.conv_r1 = nn.Conv2d(c, out_c, kernel_size=(1,k[1]), padding = [0,int((k[1]-1)/2)] )
        self.conv_r2 = nn.Conv2d(out_c, out_c, kernel_size=(k[1], 1), padding = [int((k[1]-1)/2),0] )
        self.activation = nn.Sequential( nn.BatchNorm2d(out_c),nn.ReLU(inplace=True) )

    def forward(self, x):

        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r
        x = self.activation(x)
        return x

class sc_conv( nn.Module ):

    def __init__(self,in_channels):
        super(sc_conv, self).__init__()

        self.x1_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True) )
        
        self.x2_conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels,kernel_size=3,stride=1,padding=1,bias=False),
            nn.BatchNorm2d(num_features=in_channels),
            nn.ReLU(inplace=True) )

        self.gcn_kernel7 = GCN(in_channels,in_channels,(7,7))
        self.gcn_kernel11 = GCN(in_channels,in_channels,(11,11))
        self.gcn_kernel15 = GCN(in_channels,in_channels,(15,15))

    def forward(self,x,level='level2'):

        batch_size,channels,height,width = x.shape
        channels_per_group = channels // 2
        x = x.view(batch_size,2,channels_per_group,height,width)

        x1 = x[:,0,:,:,:].contiguous()
        x2 = x[:,1,:,:,:].contiguous()

        x2 = self.x2_conv(x2)

        x1_down = F.avg_pool2d(x1,kernel_size=(4,4),stride=4,padding=1)
        x1_down = self.x1_conv(x1_down)
        x1_up = F.interpolate(x1_down,size=x1.size()[2:], mode='bilinear', align_corners=False)
        x1 = x1_up + x1

        if level == 'level2' or level == 'level3':
            x1_k7 = self.gcn_kernel7(x1)
            x1_k11 = self.gcn_kernel11(x1)
            x1_k15 = self.gcn_kernel15(x1)
            x1 = torch.cat( (x1_k7,x1_k11,x1_k15),1 )
            return x1,x2

        if level == 'level4':
            x1_k7 = self.gcn_kernel7(x1)
            x1_k11 = self.gcn_kernel11(x1)
            x1 = torch.cat((x1_k7,x1_k11), 1)
            return x1,x2

        if level == 'level5':
            x1_k7 = self.gcn_kernel7(x1)
            return x1_k7,x2

class Main_Net(nn.Module):

    def __init__(self):

        super(Main_Net, self).__init__()

        self.vgg = B2_VGG()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.squeeze5 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3 = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2 = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.sc_conv2 = sc_conv( 256 // 2 )
        self.sc_conv3 = sc_conv( 512 // 2 )
        self.sc_conv4 = sc_conv( 512 // 2 )
        self.sc_conv5 = sc_conv( 512 // 2 )

        self.bridge5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.bridge4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.bridge3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.bridge2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.attention_conv4 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.attention_conv3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.attention_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.decoder5 = nn.Sequential(
            nn.Conv2d(in_channels=320, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.decoder4 = nn.Sequential(
            nn.Conv2d(in_channels=704, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.decoder3 = nn.Sequential(
            nn.Conv2d(in_channels=960, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )
        self.decoder2 = nn.Sequential(
            nn.Conv2d(in_channels=576, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True)
        )

        self.attention_activation = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(inplace=True) )
        
        self.supervision5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.supervision4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.supervision3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.supervision2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.supervision_tri = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self,x):

        x3_3,x4_3,x5_3,x6_3 = self.vgg(x)
        global_poo1 = self.avgpool(x6_3)
        global_poo1 = F.interpolate(global_poo1,x6_3.size()[2:], mode='bilinear', align_corners=False)
        x6_3 = global_poo1 + x6_3

        out2h_x1,out2h_x2 = self.sc_conv2(x3_3,level='level2' )
        out2h_x2 = self.squeeze2( out2h_x2 )
        out2h_x2 = self.bridge2(out2h_x2)

        out3h_x1,out3h_x2 = self.sc_conv3(x4_3,level='level3' )
        out3h_x2 = self.squeeze3( out3h_x2 )
        out3h_x2 = self.bridge3(out3h_x2)

        out4h_x1,out4h_x2 = self.sc_conv4(x5_3,level='level4' )
        out4h_x2 = self.squeeze4( out4h_x2 )
        out4h_x2 = self.bridge4( out4h_x2 )

        out5h_x1,out5h_x2 = self.sc_conv5(x6_3,level='level5' )
        out5h_x2 = self.squeeze5( out5h_x2 )

        decoder5 = self.bridge5(out5h_x2)
        decoder5 = self.decoder5( torch.cat( (decoder5,out5h_x1),1) )

        out4h_high = self.attention_conv4(out5h_x2)
        out4h_high = F.interpolate(out4h_high, size=out4h_x1.size()[2:],mode='bilinear', align_corners=False)
        out4h_down = F.interpolate(decoder5, size=out4h_x1.size()[2:],mode='bilinear', align_corners=False)
        decoder4 = torch.cat( (out4h_high,out4h_x1,out4h_x2,out4h_down),1 )
        decoder4 = self.decoder4(decoder4)

        out3h_high = self.attention_conv3(out5h_x2)
        out3h_high = F.interpolate(out3h_high, size=out3h_x1.size()[2:], mode='bilinear', align_corners=False)
        out3h_down = F.interpolate(decoder4, size=out3h_x1.size()[2:], mode='bilinear', align_corners=False)
        decoder3 = torch.cat((out3h_high,out3h_x1,out3h_x2,out3h_down), 1)
        decoder3 = self.decoder3(decoder3)

        out2h_high = self.attention_conv2(out5h_x2)
        out2h_high = F.interpolate(out2h_high,size=out2h_x1.size()[2:],mode='bilinear',align_corners=False)
        out2h_down = F.interpolate(decoder3, size=out2h_x1.size()[2:], mode='bilinear', align_corners=False)
        decoder2 = torch.cat((out2h_high, out2h_x1,out2h_x2,out2h_down),1)
        decoder2 = self.decoder2(decoder2)
        supervision2 = self.supervision2(decoder2)
        decoder2_attention = torch.sigmoid(supervision2) * decoder2 + decoder2

        decoder2_attention = self.attention_activation(decoder2_attention)
        supervision_trimap = self.supervision_tri(decoder2_attention)
        supervision_trimap = F.interpolate(supervision_trimap, size=x.size()[2:], mode='bilinear', align_corners=False)
        return supervision_trimap

def LRSCN_Net( checkpoint_path, device_id):

    model = Main_Net()
    device = torch.device( "cuda:%d"%device_id if torch.cuda.is_available() else "cpu" ) 

    print("you have loaded LRSCN checkpoint file:",checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    return model





