import torch
import torch.nn.functional as F

def process_high_res_im( image, trimap,model):

    max_L = 512
    _,_,h,w = image.shape

    combined = torch.zeros_like(image)[:,0,:,:].unsqueeze(1)
    combined_weight = torch.zeros_like(image)[:,0,:,:].unsqueeze(1)
   
    padding = 32
    stride = 512

    step_size = stride - padding * 2
    step_len = max_L

    used_start_idx = {}
    for x_idx in range((w) // step_size + 1):
        for y_idx in range((h) // step_size + 1):

            start_x = x_idx * step_size
            start_y = y_idx * step_size
            end_x = start_x + step_len
            end_y = start_y + step_len
            
            if end_y > h:
                end_y = h
                start_y = h - step_len
            if end_x > w:
                end_x = w
                start_x = w - step_len

            start_x = max(0, start_x)
            start_y = max(0, start_y)
            end_x = min(w, end_x)
            end_y = min(h, end_y)

            start_idx = start_y * w + start_x
            if start_idx in used_start_idx:
                continue
            else:
                used_start_idx[start_idx] = True

            image_part = image[:, :, start_y:end_y, start_x:end_x]
            trimap_part = trimap[:, :, start_y:end_y, start_x:end_x]
            
            alpha_pred = model(image_part,trimap_part)
           
            pred_sx = pred_sy = 0
            pred_ex = step_len
            pred_ey = step_len

            
            if start_x != 0:
                start_x += padding
                pred_sx += padding
            if start_y != 0:
                start_y += padding
                pred_sy += padding
            if end_x != w:
                end_x -= padding
                pred_ex -= padding
            if end_y != h:
                end_y -= padding
                pred_ey -= padding
            
            combined[:, :, start_y:end_y, start_x:end_x] += alpha_pred[:, :, pred_sy:pred_ey, pred_sx:pred_ex]

            del alpha_pred
            torch.cuda.empty_cache()
            combined_weight[:, :, start_y:end_y, start_x:end_x] += 1

    pred = combined / combined_weight
    trimap_argmax = trimap.argmax(dim=1, keepdim=True)
    pred[trimap_argmax == 2] = 1
    pred[trimap_argmax == 0] = 0
    return pred
