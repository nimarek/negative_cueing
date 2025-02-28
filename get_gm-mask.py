from nilearn import datasets
from nilearn.image import math_img, resample_to_img

gm_mask = datasets.load_mni152_gm_mask(resolution=2)

thresholded_mask = math_img("img >= 0.5", img=gm_mask)
target_file = "/home/exp-psy/Desktop/negative_cueing/derivatives/glm_analysis/sub-01/second_lvl/phase_cue.gfeat/cope1.feat/stats/cope1.nii.gz"
resampled_mask = resample_to_img(thresholded_mask, target_file, interpolation="nearest")

output_filename = "resampled_grey_matter_mask_50percent.nii.gz"
resampled_mask.to_filename(output_filename)

print(f"Resampled grey matter mask saved to: {output_filename}")