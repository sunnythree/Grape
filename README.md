<bold>Grape is a machine learning framework implemented by c++ and cuda.</bold><br />
![LOGO](/doc/pics/logo.png)<br />

## Introduction
    When I began to learn deep leanring,I feel that <a href="https://github.com/BVLC/caffe">caffe</a> has too 
    mach dependences,<a href="https://github.com/pjreddie/darknet">darknet</a> is written by c but c++,
    and <a href="https://github.com/tiny-dnn/tiny-dnn">tiny_dnn</a> does not support cuda well。
    I hope I can combine the advantages and disadvantages of the three.
    So,Grape uses many of darknet's tool functions, and draws lessons from caffe's 
    synced_memory design and tiny_dnn's calculation diagram design.Thanks very 
    much for these open source projects.
    Grape has several advantages:
    1.dependency free
    2.support json/xml/binary parameters saving
    3.build network by JSON.(JSON is more readable than protobuf)
    4.c++ and cuda (specially cuda)is very fast
    mnist example is offered,Fully connected neural networks can easily reach 98% 
    and convolutional neural networks can easily reach 99%.

## Pull Request
Pull request is welcome.

## communicate with
QQ group: 704153141  

## License
BSD 2-Clause

