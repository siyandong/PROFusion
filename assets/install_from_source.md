### Install Eigen 3.4.0 and OpenCV 4.12.0 with CUDA 12.4 in Ubuntu 22.04 

Install required libraries
```bash
conda install -y -c conda-forge libjpeg-turbo libpng libtiff zlib ffmpeg=5.1.2 libdc1394=2.2.6
```

Create a directory to work with the source code of third-party dependencies
```bash
mkdir third_party
```

Install Eigen 3.4.0
```bash
cd third_party

# download source code
wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar -xzf eigen-3.4.0.tar.gz

# build and install
cd eigen-3.4.0
mkdir build && cd build
cmake -D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX ..
make install

cd ../../..
```

Install OpenCV 4.12.0 with CUDA 12.4 
(This is a minimal installation that includes only the necessary components. 
To install full OpenCV, you can refer to [this link](https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7))
```bash
cd third_party

# download source code
wget https://github.com/opencv/opencv/archive/4.12.0.tar.gz -O opencv-4.12.0.tar.gz
wget https://github.com/opencv/opencv_contrib/archive/4.12.0.tar.gz -O opencv_contrib-4.12.0.tar.gz
tar -xzf opencv-4.12.0.tar.gz
tar -xzf opencv_contrib-4.12.0.tar.gz

# build and install
cd opencv-4.12.0
mkdir build && cd build

cmake \
-D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=$CONDA_PREFIX \
-D BUILD_SHARED_LIBS=ON \
-D CMAKE_C_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc \    # replace with your gcc path
-D CMAKE_CXX_COMPILER=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++ \  # replace with your g++ path
-D WITH_GTK=OFF \
-D WITH_CUDA=ON \
-D CUDA_ARCH_BIN=8.0 \                                                # replace with your architecture
-D WITH_CUDNN=OFF \
-D ENABLE_FAST_MATH=ON \
-D CUDA_FAST_MATH=ON \
-D CUDA_NVCC_FLAGS="-allow-unsupported-compiler" \
-D OPENCV_DNN_CUDA=OFF \
-D BUILD_opencv_python3=ON \
-D Python3_EXECUTABLE=$CONDA_PREFIX/bin/python \
-D Python3_INCLUDE_DIR=$CONDA_PREFIX/include/python3.11 \
-D Python3_LIBRARY=$CONDA_PREFIX/lib/libpython3.11.so \
-D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.12.0/modules \    # replace with your opencv_contrib path
-D BUILD_LIST=core,imgproc,imgcodecs,cudev,cudaarithm,cudawarping,cudaimgproc,python3 \
..

make -j$(nproc)
make install

cd ../../..
```

