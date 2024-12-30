For MOSI and MOSEI, please run the following code in ./GLoMo_MSA/:
```
 python3 main_GLoMo.py --max_seq_length=60 --train_batch_size=240 --d_l=48 --layers=4 --dataset='mosi' --VISUAL_DIM=47 --learning_rate=4e-5 --n_epochs=70
 
 python3 main_GLoMo.py --max_seq_length=80 --train_batch_size=64 --d_l=192 --layers=3 --dataset='mosei' --VISUAL_DIM=35 --learning_rate=1e-5 --n_epochs=100
```

We upload the trained wights of MOSI in Baidu Netdisk, you can download from: [link](https://pan.baidu.com/s/1WGbh4kaDe9AwCKRl4IKu-Q?pwd=jcsw) Extraction Code: jcsw
