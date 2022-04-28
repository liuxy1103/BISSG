# BISSG
 code for paper IJCAI2022 "Biological Instance Segmentation with a Superpixel-Guided Graph"


## Instation
cd ./third_party/cython

python setup.py install


cd ./cython_function

python setup.py install


## Implementation
CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --nproc_per_node=1 train.py



## Contact

If you have any problem with the released code, please contact me by email (liuxyu@mail.ustc.edu.cn).

