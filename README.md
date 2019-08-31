![Grape](/doc/pics/logo.png)

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

## Examples  

For now,mnist example is offered,Fully connected neural networks can easily reach 98%  
and convolutional neural networks can easily reach 99%.  

## communicate with  

QQ group: 704153141  
EMAIL:1318288270@qq.com  

## Thranks  
* [tiny_dnn](https://github.com/tiny-dnn/tiny-dnn) 
* [caffe](https://github.com/BVLC/caffe)
* [tiny_dnn](https://github.com/tiny-dnn/tiny-dnn)
* [cereal](https://github.com/USCiLab/cereal)

## License  
BSD 2-Clause  