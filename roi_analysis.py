import os
import glob
import numpy as np
import pandas as pd
from natsort import natsorted

from nilearn.image import load_img

from rsatoolbox.model.model import ModelFixed
from rsatoolbox.data.dataset import Dataset
from rsatoolbox.inference.evaluate import eval_dual_bootstrap
from rsatoolbox.rdm.calc import calc_rdm
from rsatoolbox.rdm.rdms import concat
from rsatoolbox.vis import plot_model_comparison

import matplotlib.pyplot as plt

# I/O
experiment = "eventrelated"
phase_list = ["cue", "delay"]
sub_list = [f"{i:02}" for i in range(1, 25)]
# sub_list.remove("38")
out_dir_roi = f"/home/exp-psy/Desktop/negative_cueing/derivatives/roi-results_{experiment}/"
if not os.path.exists(out_dir_roi):
    os.makedirs(out_dir_roi, exist_ok=True)

out_dir_nRDM = f"/home/exp-psy/Desktop/negative_cueing/derivatives/roi-neural-RDM_{experiment}/"
if not os.path.exists(out_dir_nRDM):
    os.makedirs(out_dir_nRDM, exist_ok=True)

# housekeeping
rdm_list = []
lut_fpath = os.path.join(os.getcwd(), "desc-aparcaseg_dseg.tsv")
lut_df = pd.read_csv(lut_fpath, sep="\t")

# define lists of labels to keep
cortical_rois = [f"ctx-{h}-{region}" for h in ["lh", "rh"] for region in [
    "bankssts", "caudalanteriorcingulate", "caudalmiddlefrontal", "cuneus",
    "entorhinal", "fusiform", "inferiorparietal", "inferiortemporal",
    "insula", "isthmuscingulate", "lateraloccipital", "lateralorbitofrontal",
    "lingual", "medialorbitofrontal", "middletemporal", "parahippocampal",
    "paracentral", "parsopercularis", "parsorbitalis", "parstriangularis",
    "pericalcarine", "postcentral", "posteriorcingulate", "precentral",
    "precuneus", "rostralanteriorcingulate", "rostralmiddlefrontal",
    "superiorfrontal", "superiorparietal", "superiortemporal",
    "supramarginal", "frontalpole", "temporalpole", "transversetemporal"
]]

subcortical_rois = [
    "Left-Thalamus-Proper", "Right-Thalamus-Proper",
    "Left-Caudate", "Right-Caudate",
    "Left-Putamen", "Right-Putamen",
    "Left-Pallidum", "Right-Pallidum",
    "Left-Hippocampus", "Right-Hippocampus",
    "Left-Amygdala", "Right-Amygdala",
    "Left-Accumbens-area", "Right-Accumbens-area"
]

# combine cortical and subcortical ROIs
rois_of_interest = cortical_rois + subcortical_rois
filtered_data = lut_df[lut_df["name"].isin(rois_of_interest)]

for phase in phase_list:
    for roi in filtered_data["name"]:
        print(f"working on roi: {roi}")
        matches = lut_df[lut_df["name"] == roi]
        left_index = matches["index"].values[0]
        # matches = lut_df[lut_df["name"] == roi]
        # right_index = matches["index"].values[0]

        # print(f"left index: {left_index}; right index: {right_index}")

        for sub in sub_list:
            aparc_fpath = os.path.join(
                "/home", 
                "exp-psy", 
                "Desktop", 
                "negative_cueing", 
                "derivatives", 
                f"sub-{sub}", 
                "func", 
                f"sub-{sub}_task-negativesearcheventrelated_run-01_space-MNI152NLin2009cAsym_res-2_desc-aparcaseg_dseg.nii.gz"
            )
            
            aparc_data = load_img(aparc_fpath).get_fdata()

            # create bilateral roi
            target_mask = np.zeros_like(aparc_data, dtype=bool)
            target_mask[aparc_data == float(left_index)] = True
            # target_mask[aparc_data == float(right_index)] = True
            roi_size = target_mask.sum()

            # create obj. to store data
            nifti_fpaths = natsorted(
                glob.glob(
                    os.path.join(
                        "/home", 
                        "exp-psy", 
                        "Desktop", 
                        "negative_cueing", 
                        "derivatives", 
                        f"{experiment}_exp", 
                        f"lss-MNI152NLin2009cAsym", 
                        f"sub-{sub}", 
                        f"sub-{sub}_run-*_contrast-*{phase}*.nii.gz")
                )
            )

            patterns = np.full([len(nifti_fpaths), roi_size], np.nan)
            conditions = [path.split("/")[-1].split("_")[2] for path in nifti_fpaths]
            runs = [path.split("/")[-1].split("_")[1] for path in nifti_fpaths]

            # get beta files
            for c, beta_fpath in enumerate(nifti_fpaths):
                patterns[c, :] = load_img(beta_fpath).get_fdata()[target_mask].squeeze()

            descs = {"sub": sub, "task": "negative_cueing", "experiment": experiment}

            ds = Dataset(
                measurements=patterns,
                descriptors=descs,
                obs_descriptors=dict(
                    run=runs,
                    condition=conditions
                )
            )

            # calculate crossnobis RDMs from the patterns and precision matrices
            rdm_list.append(
                calc_rdm(
                    dataset=ds,
                    method="crossnobis",
                    descriptor="condition",
                    cv_descriptor="run" # wieder anmachen?
                )
            )
            
            del rdm_list[-1].descriptors["noise"] # saves memory

        # combine datasets and save it
        data_rdms = concat(rdm_list)
        #data_rdms.save(os.path.join(out_dir_nRDM, f"phase-{phase}_roi-{roi}"))
        print("creating hypothesis matrices ...")
        
        # create obj. to store data
        nifti_fpaths = natsorted(
            glob.glob(
                os.path.join(
                    "/home", 
                    "exp-psy", 
                    "Desktop", 
                    "negative_cueing", 
                    "derivatives", 
                    "eventrelated_exp", 
                    f"lss-MNI152NLin2009cAsym", 
                    f"sub-01", 
                    f"sub-01_run-01_contrast-*{phase}*.nii.gz")
            )
        )

        patterns = np.full([len(nifti_fpaths), roi_size], np.nan)
        conditions = [path.split("/")[-1].split("_")[2] for path in nifti_fpaths]
        shapes = [item.split(phase)[-1] for item in conditions]

        """
        cue-attent matrix
        """
        unique_shapes = sorted(set(shapes))
        shape_to_num = {shape: i for i, shape in enumerate(unique_shapes)}

        shape_numbers = np.array([shape_to_num[shape] for shape in shapes])

        n_conditions = len(conditions)
        cue_attent_matrix = np.zeros((n_conditions, n_conditions))

        for i, shape1 in enumerate(shapes):
            for j, shape2 in enumerate(shapes):
                cue_attent_matrix[i, j] = 1 if shape1 == shape2 else 0

        """
        cue-inhibition matrix
        """
        cue_inhibition_matrix = 1 - cue_attent_matrix

        """
        negative-positive matrix
        """
        neg_pos_matrix = np.zeros((n_conditions, n_conditions))

        groups = ["positive" if "positive" in cond else "negative" for cond in conditions]

        n_conditions = len(conditions)
        neg_pos = np.zeros((n_conditions, n_conditions))

        for i, group1 in enumerate(groups):
            for j, group2 in enumerate(groups):
                neg_pos[i, j] = 0 if group1 == group2 else 1 

        models_in = [cue_attent_matrix, cue_inhibition_matrix, neg_pos_matrix]
        model_names = ["cue-attent", "cue-inhibition", "negative-positive"]
        models_comp = []

        for model, model_name in zip(models_in, model_names):
            print(f"Processing model: {model_name}")
            models_comp.append(ModelFixed(model_name, model))

        # call function for evaluation
        results = eval_dual_bootstrap(models_comp, data_rdms)
        print(results)

        np.save(os.path.join(out_dir_roi, f"roi-{roi}_phase-{phase}.npy"), results)

        # plot model
        plot_model_comparison(results, sort=True)
        plt.savefig(os.path.join(out_dir_roi, f"roi-{roi}_phase-{phase}.png"), dpi=300)

        # clear list obj
        rdm_list.clear()
