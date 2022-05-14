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

## Notice for CVPPP
This was officially confirmed by the authors of the dataset:
In the second half of 2021, the calculation of the SBD indicator has corrected for the bug (The SBD metric on the leaderboard site was higher than the `bestDice' metric before the end of 2021, but it was later corrected by the website):

from
```shell
SBD = np.amax([bestDice,bestDice0])
```
to
```shell
SBD = np.amin([bestDice,bestDice0])
```
Therefore, for a fair comparison, we use the `bestDice' result from the site https://competition.codalab.org as our results in Table 3.

Meanwhile, our best result has been update in the leaderboard, as shown in the follows:
![image](https://user-images.githubusercontent.com/54794058/168408336-22a147db-a7dd-4395-99b1-37c547e82d5a.png)

## Contact

If you have any problem with the released code, please contact me by email (liuxyu@mail.ustc.edu.cn).

