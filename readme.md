## Code for ICCV2021 submission8870

## Prerequisites
- [Python 3.6](https://www.python.org/)
- [Pytorch 1.3](http://pytorch.org/)
- [OpenCV 4.0](https://opencv.org/)
- [Numpy 1.15](https://numpy.org/)

## Clone repository
```shell
git clone https://github.com/ICCV2021Submission8870/code.git
cd source/
```

## Download model
Please download the model [LRSCN](https://drive.google.com/file/d/1EuIpupCLmuHTDKJ2ZLVBPgg5b_ill9xL/view?usp=sharing),[HRRN](https://drive.google.com/file/d/1CCBap1EDf7nbw4QWZMVa0H9VqeVBqoLL/view?usp=sharing), then put them into `source/` root path.

## Testing
```shell
    python main.py
```

## Saliency maps 
For all saliency results in our paper, please download [Here](https://drive.google.com/file/d/18Hgvon111wpVLna3niEuOBRyQTElvHSv/view?usp=sharing)

## Evaluation
- To evaluate the performace of our network, you can use MATLAB to run `main.m` in `eval/bde`,`eval/biou`,`eval/f-s-mae` folder. Please change `gt` and `sal` path in `main.m` first.
