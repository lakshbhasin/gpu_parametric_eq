# GPU Parametric Equalizer
A CS 179 project by Laksh Bhasin and Sharon Yang.

To run just the backend code, do:
```
make -f Makefile.test
./parametric_eq <wav_file> <samp_per_buf> <threads_per_block> <max_num_blocks>
```

To run the Qt integration with the backend code, do:
```
qmake
make
./gui
```

** Note: different cuda paths might result in compile error.
See gui.pro file for more information.

