![alt tag](https://github.com/lakshbhasin/gpu_parametric_eq/blob/master/img/gpuLogo_blue.png)

A GPU Parametric Equalizer: CS 179 Project by Laksh Bhasin and Sharon Yang.


Overview
====
For more information about the application, please visit: http://gpueq.blogspot.com/


Requirement
====

Hardware requirement and libraries needed:
* NVidia GPU
* CUDA
  (http://www.quantstart.com/articles/Installing-Nvidia-CUDA-on-Ubuntu-14-04-for-Linux-GPU-Computing)
* SFML
  (http://seriousitguy.blogspot.com/2014/05/how-to-setup-sfml-on-ubuntu-1404-lts.html)
* BOOST
  (https://charmie11.wordpress.com/2014/04/23/boost-installation-on-ubuntu-14-04-lts/)
* Qt5
  (https://wiki.qt.io/Install_Qt_5_on_Ubuntu)

** Note: the authors are using:
    * CUDA 7.0.28-2
    * SFML 2.3-1
    * BOOST 1.58.0-1
    * qt5-base 5.4.1-8
On Linuax using NVidia GeForce GTX 770.


Installation
====

To run the application, make sure you have the software and hardware needed.
Clone this git repository, and do:
```
qmake
make
./gui
```

A Qt pop-up window should display the application.


To run just the backend code, do:
```
make -f Makefile.test
./parametric_eq <wav_file> <samp_per_buf> <threads_per_block> <max_num_blocks>
```

** Note: different include paths (especially the CUDA paths) might result in
compile error. Currently, the gui.pro file defaults the CUDA path at /opt/cuda.
Please modify accordingly.

