# InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes

### [Project Page](https://zju3dv.github.io/instascene) | [Paper](https://arxiv.org/abs/2507.08416) | [Arxiv](https://arxiv.org/abs/2507.08416) | [Video](https://www.youtube.com/watch?v=PUb4l_Ttf3I)

> [InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes](https://zju3dv.github.io/instascene),  
> Zesong Yang, Bangbang Yang, Wenqi Dong, Chenxuan Cao, Liyuan Cui, Yuewen Ma, Zhaopeng Cui, Hujun Bao  
> ICCV 2025


https://github.com/user-attachments/assets/3837634c-4ef9-4078-87ab-68a1c3e4faf9

![Pipeline](assets/pipeline.png)

## Installation
- [x] Installation of Scene Decomposition.
```shell
conda create -n instascene python=3.9 -y
conda activate instascene 

pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0+cu118 --extra-index-url https://download.pytorch.org/whl/cu118

pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu11==24.2.*" "cuml-cu11==24.2.*"

pip install -r requirements.txt
```


<!-- Refer to [here](https://github.com/ashawkey/cubvh?tab=readme-ov-file#trouble-shooting) If failed with `raytracing`. -->

Install [`CropFormer`](https://github.com/qqlu/Entity/tree/main/Entityv2/CropFormer) for instance-level segmentation.

```shell
cd semantic_modules/CropFormer
cd mask2former/modeling/pixel_decoder/ops
sh make.sh
cd ../../../../
git clone git@github.com:facebookresearch/detectron2.git
cd detectron2
pip install -e .
pip install git+https://github.com/cocodataset/panopticapi.git
pip install git+https://github.com/mcordts/cityscapesScripts.git
cd ..
pip install -r requirements.txt
pip install -U openmim
mim install mmcv
mkdir ckpts
```

Manually
download [CropFormer checkpoint](https://huggingface.co/datasets/qqlu1992/Adobe_EntitySeg/blob/main/CropFormer_model/Entity_Segmentation/CropFormer_hornet_3x/CropFormer_hornet_3x_03823a.pth)
into `semantic_modules/CropFormer/ckpts`

- [ ] Installation of in-situ generation.

## Data Preprocessing

Please follow the steps below to process your custom dataset, or directly download [our preprocessed datasets](https://drive.google.com/file/d/1u1VSPch9lfnstGpnzEikiso6w-w2wJ6t/view?usp=sharing).

### 1. Run instance-level segmentation.

- It's ok to use other 2D segmentation models, but make sure the input masks don't exhibit overly complex hierarchy relationships; otherwise, our method will default to the finest level.

```bash
cd semantic_modules/CropFormer
bash run_segmentation.sh "$DATA_DIR"
cd ../..
```

### 2. Training 2DGS.

Follow the [original repository](https://github.com/hbb1/2d-gaussian-splatting) to train the 2dgs model.

```bash
python train.py -s data/3dovs/bed -m output/3dovs/bed/train_2dgs
```

Optional mono normal prior ([StableNormal](https://github.com/Stable-X/StableNormal)) is available
to [enhance the reconstruction quality](https://github.com/hugoycj/2d-gaussian-splatting-great-again).

```bash
## Prepare Normal Priors
cd semantic_modules
git clone https://github.com/Stable-X/StableNormal && cd StableNormal
pip install -r requirements.txt
mv ../inference_stablenormal.py ./
python inference_stablenormal.py "$DATA_DIR"
cd ../..

## Training 2DGS with Normal Priors 
python train.py -s data/3dovs/bed --w_normal_prior stablenormal_normals -m output/3dovs/bed/train_2dgs
```

Put the trained `point_cloud.ply` file into the `$DATA_DIR` directory. After successfully executing the above steps, the
data directory should be structured as follows:

   ```
   data
      |——————3D_OVS
      |   |——————bed
      |      |——————point_cloud.ply
      |      |——————images
      |         |——————00.jpg
      |         ...
      |      |——————sam
      |         |——————mask
      |            |——————00.png
      |            ...
      |      |——————sparse
      |         |——————0
      |            |——————cameras.bin
      |            ...
      |      |——————(optional) stablenormal_normals
      |         |——————00.png
      |         ...
      |     ...
   ```

## Training with Spatial Contrastive Learning

> Note that for simple scenes, such as 3D-OVS (simple-object centered without overlap), no need to use spatial relationships to obtain robust semantic priors as shown in our supplementary material. Single-view constrastive learning is sufficient to achieve strong performance.

We train the model on a NVIDIA Tesla A100 GPU (40GB) with 10,000 iterations for about 20 minutes & less than 8GB GPU.
- Reduce the GPU & Speed the time with `--sample_batchsize 8 * 1024` or `-r 2`.
- Use `--gram_feat_3d` for a more robust feature field in complex scenes.
- It's normal to get stuck at the `DBScan Filter Stage`, since the backgrount gaussian points may be divided into multi-regions.
```bash
python train_semantic.py -s data/lerf/waldo_kitchen -m train_semanticgs --use_seg_feature --iterations 10000 --load_filter_segmap
```

After completing the training, we provide a GUI modified from [Omniseg3D](https://github.com/THU-luvision/OmniSeg3D) for real-time ineractive segmentation.
The `point_cloud.ply` in [our preprocessed datasets](https://drive.google.com/file/d/1u1VSPch9lfnstGpnzEikiso6w-w2wJ6t/view?usp=sharing) already has pretrained semantic features.
```bash
python semantic_gui.py \
  --ply_path data/lerf/waldo_kitchen/point_cloud.ply \
  --interactive_note lerf_waldo_kitchen \
  --use_colmap_camera \
  --source_path data/lerf/waldo_kitchen --resolution 1
```
- `Left Mouse` for changing rendering view
- `Click Mode` + 0.9 `Threshold` + `Right Mouse` for segmentation
- `Clear Edit` for clear the segmentation cache
- `Delete 3D` for remove the chosen gaussians
- `Segment 3D` for only keep the chosen gaussians
- `Reload Data` for reload the gaussian model

https://github.com/user-attachments/assets/aca0ae53-744e-4037-99b2-197917953303

## ToDos
🔥 Feel free to raise any requests, including support for additional datasets or broader applications of segmentation~
- [x] Release project page and paper.
- [x] Release scene decomposition code.
- [ ] Release in-situ generation code.

## Acknowledgements
Some codes are modified from 
[Omniseg3D](https://github.com/THU-luvision/OmniSeg3D),
[MaskClustering](https://github.com/PKU-EPIC/MaskClustering),
[2DGS++](https://github.com/hugoycj/2d-gaussian-splatting-great-again),
thanks for the authors for their valuable works.

### Citation

If you find this code useful for your research, please use the following BibTeX entry.

```
@inproceedings{yang2025instascene,
    title={InstaScene: Towards Complete 3D Instance Decomposition and Reconstruction from Cluttered Scenes},
    author={Yang, Zesong and Yang, Bangbang and Dong, Wenqi and Cao, Chenxuan and Cui, Liyuan and Ma, Yuewen and Cui, Zhaopeng and Bao, Hujun},
    booktitle=ICCV,
    year={2025}
}
```