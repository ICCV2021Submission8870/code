clear all; close all; clc;

gt_path = 'write your gt path';
sal_path = 'write your saliency map path';


imgFiles = dir([sal_path '*.png'] );
imgNUM = length(imgFiles);

Smeasure=zeros(1,imgNUM);
Emeasure=zeros(1,imgNUM);
Fmeasure=zeros(1,imgNUM);
MAE=zeros(1,imgNUM);

tic;

for i = 1 : imgNUM
    
    fprintf( 'Evaluating: %d/%d\n', i, imgNUM )
    
    name = imgFiles(i).name
    gtname = name;
    
    gt = imread( [gt_path gtname] );
    
    if numel( size(gt) ) > 2
        gt = rgb2gray(gt);
    end
    
    if ~islogical( gt )
        gt = gt( :,:,1 ) > 128;
    end
    
    sal = imread( [sal_path name] );
    
    if size( sal,1 ) ~= size( gt,1) || size( sal,2 ) ~= size( gt,2 )
        sal = imresize( sal, size(gt) );
        imwrite( sal, [ sal_path name] );
        fprintf( ' Error occurs in the path: %s!!!\n', [sal_path name] );
    end
    
    sal = im2double( sal(:,:,1) );

    sal = reshape(mapminmax(sal(:)',0,1),size(sal));
    
    Smeasure(i) = StructureMeasure(sal,logical(gt));
    
    temp = Fmeasure_calu(sal,double(gt),size(gt)); 
    Fmeasure(i) = temp(3);
      
    MAE(i) = mean2(abs(double(logical(gt)) - sal));

    threshold =  2 * mean(sal(:)) ;
    if ( threshold > 1 )
        threshold = 1;
    end
    
    Bi_sal = zeros(size(sal));
    Bi_sal(sal>threshold)=1;
    Emeasure(i) = Enhancedmeasure(Bi_sal,gt);

end

toc;

Sm = mean2(Smeasure);
Fm = mean2(Fmeasure);
Em = mean2(Emeasure);
mae = mean2(MAE);

fprintf('Smeasure: %.3f; Emeasure %.3f; Fmeasure %.3f; MAE: %.3f.\n',Sm, Em, Fm, mae);
    

