![alt tag](https://github.com/lakshbhasin/gpu_parametric_eq/blob/master/img/gpuLogo_blue.png)

GPU Parametric Equalizer (GPUEQ): A CS 179 Project by Laksh Bhasin and Sharon Yang.


Overview
====

<a href="https://www.youtube.com/watch?v=NSv1J9dPn40" target="_blank">
<img src="https://github.com/lakshbhasin/gpu_parametric_eq/blob/master/img/demo.png" 
alt="Click for a demostration video on Youtube"/></a>
[Click the screenshot for a demostration video]

For more information about the application, please visit: http://gpueq.blogspot.com/


Requirements
====

The following hardware and libraries are required:
* An NVidia GPU
* GCC (with C++11 support)
* CUDA
 * Ubuntu setup guide: http://www.quantstart.com/articles/Installing-Nvidia-CUDA-on-Ubuntu-14-04-for-Linux-GPU-Computing
* SFML
 * Ubuntu setup guide: http://seriousitguy.blogspot.com/2014/05/how-to-setup-sfml-on-ubuntu-1404-lts.html
* Boost
 * Ubuntu setup guide: https://charmie11.wordpress.com/2014/04/23/boost-installation-on-ubuntu-14-04-lts/
* Qt5
 * Ubuntu setup guide: https://wiki.qt.io/Install_Qt_5_on_Ubuntu

Note that we have only tested our code with the following hardware/libraries/OS:
* NVidia GeForce GTX 770
 * Architecture: Kepler
 * CC: 3.0
* Arch Linux
 * linux kernel 4.0.4-1
* gcc-multilib 4.9.2-4
* cuda 7.0.28-2
* sfml 2.3-1
* boost 1.58.0-1
* qt5-base 5.4.1-8

Also, we have set our CUDA path to /opt/cuda. Different include paths might
result in compilation errors, so please modify the gui.pro file (or the
back-end Makefile.test file) accordingly.

Installation
====

To run the GUI, make sure you have the software and hardware needed. Then,
clone this git repository, and run the following commands from the project's
directory:
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

If there are any errors in compilation, please check your include paths
(especially the CUDA path) and modify gui.pro (or Makefile.test) accordingly.
