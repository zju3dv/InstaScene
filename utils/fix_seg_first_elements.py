import numpy as np
import os

# 有时候idx为0的可能是物体！

seg_folder = "/home/bytedance/Projects/Datasets/LERF_Dataset/lerf_ovs/figurines/segment_results/render_1/output/object"

clusteringSeg3D_masks = np.load(os.path.join(seg_folder, "mask3d.npz"),
                                allow_pickle=True)["pred_masks"]
clusteringSeg3D_masks = np.concatenate([np.zeros_like(clusteringSeg3D_masks[:, :1], dtype=bool), clusteringSeg3D_masks],
                                       axis=-1)
pred_dict = {
    "pred_masks": clusteringSeg3D_masks
}
np.savez(os.path.join(seg_folder, f'mask3d.npz'), **pred_dict)
# 加一行

# 2D mask是每一个label都加1
clusteringSeg2D_masks = np.load(os.path.join(seg_folder, "object_dict.npy"),
                                allow_pickle=True).item()  # 没有underseg_mask
clusteringSeg2D_masks_new = {}
for k, v in clusteringSeg2D_masks.items():
    clusteringSeg2D_masks_new[k + 1] = v
np.save(os.path.join(seg_folder, 'object_dict.npy'), clusteringSeg2D_masks_new, allow_pickle=True)
