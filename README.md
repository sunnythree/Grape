![Grape](/doc/pics/logo.png)  

[中文文档](https://blog.csdn.net/u011913612/article/details/100180166)  

## What is Grape?  
Grape is a machine learning framework implemented by c++ and cuda.  
## Introduction  
When I began to learn deep leanring,I feel that [caffe](https://github.com/BVLC/caffe) has too   
mach dependences,[darknet](https://github.com/pjreddie/darknet) is written by c but c++,  
and [tiny_dnn](https://github.com/tiny-dnn/tiny-dnn) does not support cuda well。  
I hope I can combine the advantages and disadvantages of the three.  
So,Grape uses many of darknet's tool functions, and draws lessons from caffe's  
synced_memory design and tiny_dnn's calculation diagram design.Thanks very  
much for these open source projects.  
Grape has several advantages:  

* dependency free
* support json/xml/binary parameters saving  
* build network by JSON.(JSON is more readable than protobuf)  
* c++ and cuda (specially cuda)is very fast.  

## Compile

* linux:  
change Makefile to open/close GPU,OPENMP,DEBUG,TEST  
and then just excute:  
  `make ` 
* os x:  
os x may do not support cuda,so disable GPU option,and then you just need excute:  
  `make`  
after excute make,Grape and libGrape.so(libGrape.dylib in os x) libGrape.a will generated.
if you want compile unit test,just excute:  
  `make test`  

## Examples  

For now,mnist example is offered,Fully connected neural networks can easily reach 98%  
and convolutional neural networks can easily reach 99%.  
for running full connected neural network to train and test on mnist dataset, excute:  
  `./Grape cfg/mnist/mnist_net.json`  
for running convolution neural network to train and test on mnist dataset,excute:  
  `./Grape cfg/mnist/mnist_net_conv.json`  

## Design Thought  
net contains a lot of graphs:  
![Grape](/doc/pics/net.png)  
the relationship beteen operations and tensor:  
![Grape](/doc/pics/op.png)  
Tensor contains data and connect the operations.The tensor which type is DATA transmit
data from one op to another,The WEIGHT tensor and BIAS tensor only hold data for the op
which create it.

## communicate with  

QQ group: 704153141  
EMAIL:1318288270@qq.com  

## Thranks  
* [darknet](https://github.com/pjreddie/darknet) 
* [caffe](https://github.com/BVLC/caffe)
* [tiny_dnn](https://github.com/tiny-dnn/tiny-dnn)
* [cereal](https://github.com/USCiLab/cereal)

## License  
BSD 2-Clause  