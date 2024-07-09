# Swin-HAUnet
This repo holds code for: Swin-HAUnet: A Swin-Hierarchical Attention Unet For Enhanced Medical Image Segmentation

# Usage
1.Download pre-trained swin transformer model (Swin-T)  
[Get pre-trained model in this link] 
(https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/"

2.Prepare data  
*data link:
(https://www.synapse.org/#!Synapse:syn3193805/wiki/217789)  
(https://kaggle.com/competitions/uw-madison-gi-tract-image-segmentation)  
3. Environment  
*Please prepare an environment with python=3.8, and then use the command "pip install -r requirements.txt" for the dependencies.

4.Train/Test  
*train  
'''
sh train.sh or python train.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.01 --batch_size 2  
'''
*test  
'''
sh test.sh or python test.py --dataset Synapse --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
'''
