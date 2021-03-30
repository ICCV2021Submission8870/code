Thresholds = 1:-1/255:0;



curSetName = 'davis';
salPath = 'write your saliency map path';      % sal path
gtPath = 'write your gt path';        % gt path
    
imgFiles = dir([salPath '*.png']);
imgNUM = length(imgFiles);

bmeasure=zeros(1,imgNUM);
tic;
for i = 1:imgNUM
    fprintf('%d on %s \n', i, curSetName);
    name =  imgFiles(i).name;
    pred = imread([salPath name(1:end-4) '.png']);
    %load gt
    gt = imread([gtPath name(1:end-4) '.png']);
    bmeasure(i) = compute_bounday_iou_measure(pred,gt);
    
end
toc;
b_measure = mean(bmeasure)