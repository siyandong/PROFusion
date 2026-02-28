# Copyright (c) 2025–present Siyan Dong - siyandong.3 [at] gmail.com
# CC BY-NC-SA 4.0 https://creativecommons.org/licenses/by-nc-sa/4.0/  

mkdir build
cd build
rm -rf *
cmake ..
make -j$(nproc)
cd ..