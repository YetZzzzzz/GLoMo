# GLoMo: Global-Local Modal Fusion for Multimodal Sentiment Analysis
We propose the GloMo to integrate the multiple local representations, and fuse the local and global information for Multimodal Sentiment Analysis.


### The Framework of GLoMo:
![image](https://github.com/YetZzzzzz/GLoMo/blob/main/diagram.png)
Figure 1: The diagram of the GLoMo. The text, video and audio representations are firstly processed by modality-specific encoders to get the global representations, followed by modality-specific MoEs layers to get the local representation of each modality. The global and local representations of the three modalities are then fed into the global-guided fusion module for prediction. The same color for dashed lines and boxes are used to represent the same operation’s inputs.



### Datasets:
**Please move the following datasets into directory ./datasets/.**

The CMU-MOSI and CMU-MOSEI datasets can be downloaded according to [MIB](https://github.com/TmacMai/Multimodal-Information-Bottleneck) and [MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer) through the following link: 
```
pip install gdown
gdown https://drive.google.com/uc?id=12HbavGOtoVCqicvSYWl3zImli5Jz0Nou
gdown https://drive.google.com/uc?id=1VJhSc2TGrPU8zJSVTYwn5kfuG47VaNQ3. 
```

For UR-FUNNY and MUStARD, the dataset can be downloaded according to [HKT](https://github.com/matalvepu/HKT/blob/main/dataset/download.txt) through:
```
Download Link of UR-FUNNY:  https://www.dropbox.com/s/5y8q52vj3jklwmm/ur_funny.pkl?dl=1
Download Link of MUsTARD: https://www.dropbox.com/s/w566pkeo63odcj5/mustard.pkl?dl=1
```
Please rename the files as ur_funny.pkl and mustard.pkl, and move them into the directory ./datasets/.

For CHERMA dataset, you can download from [LFMIM](https://github.com/sunjunaimer/LFMIM) through: 
```
https://pan.baidu.com/s/10PoJcXMDhRg4fzsq96A7rQ
Extraction code: CHER
```
Please put the files into directory ./datasets/CHERMA0723/.

### Prerequisites:
```
- Python 3.8.10
- CUDA 11.5
- pytorch 1.12.1+cu113
- transformers 4.30.2
```
see requirements.txt for more details

### Pretrained model:
Downlaod the [BERT-base](https://huggingface.co/google-bert/bert-base-uncased/tree/main) , and put into directory ./BERT-EN/.

### Run GLoMo

For MOSI and MOSEI, please run the following code in ./GLoMo_MSA/ through:
```
 python3 main_GLoMo.py --max_seq_length=60 --train_batch_size=240 --d_l=48 --layers=4 --dataset='mosi' --VISUAL_DIM=47 --learning_rate=4e-5 --n_epochs=70
 
 python3 main_GLoMo.py --max_seq_length=80 --train_batch_size=64 --d_l=192 --layers=3 --dataset='mosei' --VISUAL_DIM=35 --learning_rate=1e-5 --n_epochs=100
```

For MUStARD and UR-FUNNY, please run the code in ./GLoMo_MHD/ through:
```
python3 main_GLoMo_humor.py --dataset='sarcasm' --train_batch_size=64 --layers=3 --max_seq_length=70 --d_l=160 --n_epochs=5

python3 main_GLoMo_humor.py --dataset='urfunny' --train_batch_size=220 --layers=3 --max_seq_length=80 --d_l=112 --n_epochs=100
```

For CHERMA, please run the code in ./GLoMo_MER/ through:
```
CUDA_VISIBLE_DEVICES='1,2,3' torchrun --nproc_per_node=1 GLoMo_MER.py \
        --learning_rate 2e-5 \
        --dropout_prob 0.5 \
        --d_l 256 \
        --layers 7 \
        --epoch 20 \
        --batch_size 400 \
```



### Citation:
**Addition results and settings of hyper-parameters can be found in GLoMo_appendix_final.pdf.** Please cite our paper if you find our work useful for your research:
```
@inproceedings{zhuang2024glomo,
  title={GLoMo: Global-Local Modal Fusion for Multimodal Sentiment Analysis},
  author={Zhuang, Yan and Zhang, Yanru and Hu, Zheng and Zhang, Xiaoyue and Deng, Jiawen and Ren, Fuji},
  booktitle={Proceedings of the 32nd ACM International Conference on Multimedia},
  pages={1800--1809},
  year={2024}
}
```

### Acknowledgement
Thanks to  [MIB](https://github.com/TmacMai/Multimodal-Information-Bottleneck) , [MAG](https://github.com/WasifurRahman/BERT_multimodal_transformer),  [MCL](https://github.com/TmacMai/Multimodal-Correlation-Learning), [HKT](https://github.com/matalvepu/HKT), [LFMIM](https://github.com/sunjunaimer/LFMIM) for their great help to our codes and research. 
