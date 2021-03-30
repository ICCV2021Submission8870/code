from network.LRSCN.En_De import LRSCN_Net
from network.HRRN.solver import HRRN_Net
from dataloader.data_loader import Val_Datasets
from test import Test
from torch.utils.data import DataLoader
import torch
import os


if __name__=='__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.set_device(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device_id = 0

    val_dataset = Val_Datasets()
    val_loader = DataLoader(dataset=val_dataset,batch_size=1,shuffle=False,num_workers=8)

    LRSCN = LRSCN_Net(checkpoint_path="LRSCN.pth", device_id=device_id).to(device)
    HRRN = HRRN_Net(checkpoint_path="HRRN.pth", device_id=device_id).to(device)

    tester = Test( lrscn = LRSCN,hrrn = HRRN, device = device )
    tester.test( val_loader = val_loader, val_dataset = val_dataset  )
