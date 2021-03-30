import torch
import os,cv2
import sys, time
import torch.nn.functional as F
import numpy as np
from process import process_high_res_im

class Test(object):

    def __init__(self,lrscn,hrrn,device):

        self.LRSCN = lrscn
        self.HRRN = hrrn
        self.device = device

    def test( self,val_loader,val_dataset ):

        device = self.device
        since = time.time()

        self.LRSCN.eval()
        self.HRRN.eval()

        bar_steps = len(val_loader)
        process_bar = ShowProcess( bar_steps )

        save_path = os.path.join("results")
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        

        for i,data in enumerate(val_loader,0):

            images_low,images_high = data
            images_low,images_high = images_low.to(device).float(),images_high.to(device).float()
            
            with torch.set_grad_enabled(False):

                trimap = self.LRSCN(images_low)
                trimap = torch.softmax(trimap,1)
                trimap = trimap.detach().cpu().numpy()
                trimap = trimap[0, :, :, :]
                trimap = np.transpose(trimap,(1,2,0))
                trimap = np.argmax(trimap,2)
                trimap[trimap == 0] = 0
                trimap[trimap == 1] = 255
                trimap[trimap == 2] = 150
                trimap = np.uint8(trimap)

                trimap = cv2.resize(trimap, dsize=(1024,1024), interpolation=cv2.INTER_LINEAR)

                h,w = trimap.shape
                trimap[trimap < 85] = 0
                trimap[trimap >= 170] = 2
                trimap[trimap >= 85] = 1
                trimap = torch.from_numpy(trimap).to(torch.long)
                trimap = F.one_hot(trimap,num_classes=3).permute(2,0,1).to(device).float()
                trimap = trimap.unsqueeze(0)

                alpha_pred = process_high_res_im(image=images_high,trimap=trimap,model=self.HRRN)
                
                test_pred = alpha_pred[0, 0, ...].data.cpu().numpy() * 255.0
                test_pred = np.round(test_pred)
                test_pred = test_pred.astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, val_dataset.examples[i]["label_name"]), test_pred)
                
            process_bar.show_process()
        process_bar.close()

class ShowProcess():

    i = 0
    max_steps = 0
    max_arrow = 50

    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.i = 0

    def show_process(self, i=None):
        if i is not None:
            self.i = i
        else:
            self.i += 1
        num_arrow = int(self.i * self.max_arrow / self.max_steps)
        num_line = self.max_arrow - num_arrow
        percent = self.i * 100.0 / self.max_steps
        process_bar = '[' + '>' * num_arrow + '-' * num_line + ']' \
                      + '%.2f' % percent + '%' + '\r'
        sys.stdout.write(process_bar)
        sys.stdout.flush()

    def close(self):
        print('')
        self.i = 0


