import numpy as np
from tqdm import tqdm

from scene import GaussianModel
from scene.cameras import Camera

from typing import List


def remedy_undersegment(gaussian: GaussianModel, viewcams: List[Camera], mask_assocation, threshold=0.8):
    undersegment_frame_masks = [mask_assocation['global_frame_mask_list'][frame_id] for frame_id in
                                mask_assocation['undersegment_mask_ids']]
    error_undersegment_frame_masks = {}
    remedy_undersegment_frame_masks = []

    instance_seg3D_labels = [set(point_ids) for point_ids in mask_assocation["total_point_ids_list"]]
    frames_gaussian = []
    for frame_id in range(len(viewcams)):
        frames_gaussian.append(set(np.where(mask_assocation['gaussian_in_frame_matrix'][:, frame_id])[0]))

    iterator = tqdm(undersegment_frame_masks, total=len(undersegment_frame_masks),
                    desc="Remedy Error-Classified Undersegment")

    for frame_mask in iterator:
        frame_id, mask_id = frame_mask
        frame_mask_gaussian = mask_assocation['mask_gaussian_pclds'][f"{frame_id}_{mask_id}"]

        frame_gaussian = frames_gaussian[frame_mask[0]]
        ## each instance visible in current frame
        instance_frame_gaussian = [seg3D_labels.intersection(frame_gaussian) for seg3D_labels in instance_seg3D_labels]
        instance_intersect_gaussian = np.array([len(frame_mask_gaussian.intersection(instance_gaussian))
                                                for instance_gaussian in instance_frame_gaussian])
        best_match_instance_idx = np.argsort(instance_intersect_gaussian)[::-1]
        best_match_intersect = instance_intersect_gaussian[best_match_instance_idx[0]]
        if best_match_intersect / len(frame_mask_gaussian) > threshold:
            error_undersegment_frame_masks[frame_mask] = best_match_instance_idx[0]
        else:
            remedy_undersegment_frame_masks.append(mask_assocation['global_frame_mask_list'].index(frame_mask))

    mask_assocation['undersegment_mask_ids'] = remedy_undersegment_frame_masks
    total_mask_list = mask_assocation['total_mask_list']

    for frame_mask in error_undersegment_frame_masks:
        total_mask_list[error_undersegment_frame_masks[frame_mask]].append(frame_mask)

    mask_assocation['total_mask_list'] = total_mask_list

    return mask_assocation
