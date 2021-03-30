function generate_mat(gt_path, salmap_path, save_gt_mat_path, save_sal_mat_path)

	gt_list = dir([gt_path '*.png']);
	salmap_list = dir( [salmap_path '*.png']);


	for i = 1:length(salmap_list)
        
		gt_img = imread(strcat(gt_path, gt_list(i).name));
		salmap_img = imread(strcat(salmap_path, salmap_list(i).name));
		
		gt_img = gt_img(:,:,1);
		gt_img = logical(gt_img);
		salmap_img = salmap_img(:,:,1);
		
		gt_img = double(gt_img);
		salmap_img = double(salmap_img);	
		
		imageLabelCell = {gt_img};
		sampleLabels = salmap_img;
		
		save(strcat(save_gt_mat_path, gt_list(i).name(1:length(gt_list(i).name)-4), '.mat'), 'imageLabelCell');
		save(strcat(save_sal_mat_path, salmap_list(i).name(1:length(salmap_list(i).name)-4), '.mat'), 'sampleLabels');
	end

end