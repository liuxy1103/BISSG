# BISSG
 code for paper IJCAI2022 "Biological Instance Segmentation with a Superpixel-Guided Graph"


## Instation
This code was implemented with Pytorch 1.0.1 (later versions may work), CUDA 9.0, Python 3.7.4 and Ubuntu 16.04. 

If you have a [Docker](https://www.docker.com/) environment, we strongly recommend you to pull our image as follows:

```shell
docker pull registry.cn-hangzhou.aliyuncs.com/em_seg/v54_higra
```
Otherwise, please execute the following commands to ensure that the dependent software is installed
```shell
cd ./third_party/cython
python setup.py install
cd ../../
cd ./cython_function
python setup.py install
cd ../
```

## Implementation
```shell
CUDA_VISIBLE_DEVICES=0  python -m torch.distributed.launch --nproc_per_node=1 train.py
```


## Contact

If you have any problem with the released code, please contact me by email (liuxyu@mail.ustc.edu.cn).

