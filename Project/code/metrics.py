import numpy as np
import cv2


def calculate_metrics():
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import mean_squared_error

    path_to_originals = [
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_1_input_3.jpg',
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_6_input_6.jpg',
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_7_input_2.jpg',
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_8_input_0.jpg',
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_9_input_1.jpg',
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_10_input_4.jpg',
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_11_input_7.jpg', 
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_15_input_5.jpg'
                    ]
    path_to_denoised = [
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_1_R=4_final_generated_sample_3.jpg',
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_6_R=4_final_generated_sample_6.jpg',
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_7_R=4_final_generated_sample_2.jpg',
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_8_R=4_final_generated_sample_0.jpg',
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_9_R=4_final_generated_sample_1.jpg',
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_10_R=4_final_generated_sample_4.jpg',
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_11_R=4_final_generated_sample_7.jpg', 
                        '/home/akash/Documents/Projects/csgm-mri-langevin/outputs/2024-05-01/12-34-29/file_brain_AXT2_200_2000019.h5|langevin|slide_idx_15_R=4_final_generated_sample_5.jpg'
                    ]
    psnr_results, mse_results, ssim_results = [], [], []
    for clean_image_path, noisy_image_path in zip(path_to_originals, path_to_denoised):
        print(clean_image_path, noisy_image_path)
        clean_image, noisy_image = cv2.imread(clean_image_path, 0), cv2.imread(noisy_image_path, 0)
        print(clean_image.shape, noisy_image.shape)
        psnr_value = cv2.PSNR(clean_image, noisy_image)
        mean_square_value = mean_squared_error(clean_image, noisy_image)
        
        ssim_value = ssim(clean_image, noisy_image, data_range = 255)
        print(f'{psnr_value = } {mean_square_value = } {ssim_value = }')
        psnr_results.append(psnr_value)
        mse_results.append(mean_square_value)
        ssim_results.append(ssim_value)
    
    print(f'Mean PSNR {np.mean(psnr_results)} Mean MSE {np.mean(mse_results)}, Mean SSIM {np.mean(ssim_results)}')

if __name__ == '__main__':
    calculate_metrics()
