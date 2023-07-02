# multimodal_3D
Exploring Unsupervised Multimodal 3D Understanding, a project for Advanced Topics in 3D CV Praktikum

## For OpenScene Multi-view Feature Fusion
### To create a new virtual environment with conda
- conda create -n openscene python=3.8
- conda activate openscene
### Install necessary packages
- conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
- pip install -r requirements.txt
- conda install -c conda-forge tensorflow
### Download OpenSeg model 
https://drive.google.com/file/d/1DgyH-1124Mo8p6IUJ-ikAiwVZDDfteak/view?usp=sharing

### Run the code to get fused features

- `--openseg_model` : path should be the one that has "saved_model.pb" and "graph_def.pbtxt" files
- `--data_dir :path` should have the preprocessed data such as "scannet_2d" and "scannet_3d"

Take ScanNet as an example, to perform multi-view feature fusion your can run:
```bash
python scannet_openseg.py \
            --data_dir PATH/TO/scannet_processed \
            --output_dir PATH/TO/OUTPUT_DIR \
            --openseg_model PATH/TO/OPENSEG_MODEL \
            --process_id_range 0,100\
            --split train
```
### Important
- Be aware of the data path whether it has `/` or `\` as a separator


## Demo Jupyter Notebook for Fused 2D Features
[notebook](demo_2d.ipynb)
## Demo Jupyter Notebook for Distilled 3D Features
[notebook](demo_3d.ipynb)

## Simplified distillation pipeline
[notebook](distill_simplified.ipynb)
## Simplified pipeline to save distilled 3D features
[notebook](eval_simplified.ipynb)
