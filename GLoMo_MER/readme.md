For CHERMA, please run the code in ./GLoMo_MER/:
```
CUDA_VISIBLE_DEVICES='1,2,3' torchrun --nproc_per_node=1 GLoMo_MER.py \
        --learning_rate 2e-5 \
        --dropout_prob 0.5 \
        --d_l 256 \
        --layers 7 \
        --epoch 20 \
        --batch_size 400 \
```
