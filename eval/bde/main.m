gt_path = 'write your gt path';
salmap_path = 'write your saliency map path';

save_gt_mat_path = 'gt_mat/';
save_sal_mat_path = 'sal_mat/';
mkdir( 'gt_mat' );
mkdir( 'sal_mat' );

generate_mat(gt_path, salmap_path, save_gt_mat_path, save_sal_mat_path);
image_segmentation_benchmarks(save_gt_mat_path, save_sal_mat_path);