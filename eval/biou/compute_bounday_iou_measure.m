function boundary_iou = compute_bounday_iou_measure(pred,gt)
I = pred(:,:,1);
J = gt(:,:,1);
edge_I = double(edge(I,'canny'));

strel_dilate = strel('square',3);
edge_I = imdilate(edge_I,strel_dilate);
edge_J = double(edge(J,'canny'));
edge_J = imdilate(edge_J,strel_dilate);

inter = edge_I.*edge_J;
inter = sum(inter(:));
edge_I_s = edge_I.^2;
edge_J_s = edge_J.^2;
uinon = sum(edge_I_s(:))+sum(edge_J_s(:));
boundary_iou = 1-2*inter/uinon;
end

