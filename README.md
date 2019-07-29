# Deep Floor Plan Recognition using a Multi-task Network with Room-boundary-Guided Attention
By Zhiliang ZENG, XIANZHI LI, Ying Kin Yu, and Chi-Wing Fu

[2019/07/29: updated demo code & pretrained model]

## Requirements

- Please install tensorflow-gpu
- Please install Python 2.7
- We used Nvidia Titan Xp GPU with CUDA 8.0 installed

Our code have been tested by using tensorflow-gpu==1.10.1

## Python packages

- [numpy]
- [scipy]
- [matplotlib]
- [Pillow]

## demo code usage
Download our pretrained model and put it into [pretrained] folder, then run
```bash
python demo.py --im_path=./demo/45719584.jpg 
```
