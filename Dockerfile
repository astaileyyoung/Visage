# ---- libtorch Build Stage ----
FROM ubuntu:22.04 as libtorch-extract
RUN apt-get update && apt-get install -y unzip wget && \
    wget https://download.pytorch.org/libtorch/cu113/libtorch-cxx11-abi-static-with-deps-1.12.1%2Bcu113.zip -O /tmp/libtorch.zip && \
    unzip /tmp/libtorch.zip -d /opt/libtorch && rm /tmp/libtorch.zip

FROM ubuntu:22.04 as video-codec-sdk-extract
RUN apt-get update && apt-get install -y unzip
COPY vendor/Video_Codec_SDK_13.0.19.zip /tmp/sdk.zip
WORKDIR /tmp
RUN unzip sdk.zip

RUN mkdir -p /sdk/include && mkdir -p /sdk/lib
RUN cp Video_Codec_SDK*/Interface/*.h /sdk/include/
RUN cp Video_Codec_SDK*/Lib/linux/stubs/x86_64/*.so /sdk/lib/

# COPY Video_Codec_SDK_13.0.19/Interface/nvcuvid.h \
#      Video_Codec_SDK_13.0.19/Interface/nvEncodeAPI.h \
#      Video_Codec_SDK_13.0.19/Interface/cuviddec.h \
#      /usr/local/cuda/include/

# COPY Video_Codec_SDK_13.0.19/Lib/linux/stubs/x86_64/libnvcuvid.so \
#      Video_Codec_SDK_13.0.19/Lib/linux/stubs/x86_64/libnvidia-encode.so \
#      /usr/local/lib/

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get clean && apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    gcc-9 g++-9 \
    binutils \
    cmake \
    git \
    wget \
    unzip \
    yasm \
    nasm \
    pkg-config \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk2.0-dev \
    libgtk-3-dev \
    libgtk-3-0 \
    cuda-nvcc-11-8 \
    libnpp-dev-11-8 \
    && rm -rf /var/lib/apt/lists/*

ENV CC=gcc-9
ENV CXX=g++-9

ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=/usr/local/lib:/usr/local/cuda/lib64:/usr/local/cuda/lib64/stubs:${LD_LIBRARY_PATH}

# Install nv-codec-headers
RUN git clone https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    git checkout n11.1.5.2 && \
    make && make install && \
    cd .. && rm -rf nv-codec-headers

COPY --from=video-codec-sdk-extract /sdk/include/*.h /usr/local/cuda/include/
COPY --from=video-codec-sdk-extract /sdk/lib/*.so /usr/local/lib/

RUN ln -sf /usr/local/lib/libnvcuvid.so /usr/local/lib/libnvcuvid.so.1 && \
    ln -sf /usr/local/lib/libnvidia-encode.so /usr/local/lib/libnvidia-encode.so.1

# Build FFmpeg 4.4 from source with CUDA/NVENC/NVDEC
RUN git clone --branch n4.4 https://github.com/FFmpeg/FFmpeg.git && \
cd FFmpeg && \
    ./configure --prefix=/usr/local \
      --enable-cuda --enable-cuvid --enable-nvdec \
      --enable-libnpp --extra-cflags="-I/usr/local/include -I/usr/local/cuda/include -I/usr/local/include/ffnvcodec" \
      --extra-ldflags="-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib" \
      --enable-gpl --enable-nonfree \
      --disable-debug --disable-doc \
      --enable-shared \
      --disable-static && \
      make -j"$(nproc)" && make -j"$(nproc)" install && cd .. && rm -rf FFmpeg

COPY scripts/generate-pc-files.sh /tmp/
RUN chmod +x /tmp/generate-pc-files.sh && bash /tmp/generate-pc-files.sh

RUN ln -sf /usr/local/lib/libnvcuvid.so /usr/local/lib/libnvcuvid.so.1 && \
    ln -sf /usr/local/lib/libnvidia-encode.so /usr/local/lib/libnvidia-encode.so.1

RUN rm -rf FFmpeg && rm -rf opencv && rm -rf opencv_contrib && \
    git clone https://github.com/opencv/opencv.git && \
    git clone https://github.com/opencv/opencv_contrib.git

# # # # Build OpenCV with CUDA/cuDNN/cudacodec/FFmpeg
RUN cd opencv && mkdir build && cd build && \
cmake -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -D CMAKE_VERBOSE_MAKEFILE=OFF \
      -D CMAKE_LIBRARY_PATH="/usr/local/cuda/lib64/stubs" \
      -D BUILD_opencv_gapi=OFF \
      \
      -D WITH_CUDA=ON \
      -D WITH_CUDNN=ON \
      -D WITH_NVDEC=ON \
      -D WITH_NVCUVENC=ON \
      -D BUILD_opencv_cudaimgproc=ON \
      -D BUILD_opencv_cudaarithm=ON \
      -D BUILD_opencv_cudawarping=ON \
      -D CUDA_ARCH_BIN="5.0;6.1;7.5;8.6;8.9" \
      -D BUILD_opencv_cudacodec=ON \
      -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda \
      \
      -D WITH_FFMPEG=ON \
      -D FFMPEG_INCLUDE_DIR=/usr/local/include \
      -D FFMPEG_LIB_DIR=/usr/local/lib \
      -D FFMPEG_LIBRARIES="/usr/local/lib/libavcodec.so;/usr/local/lib/libavformat.so;/usr/local/lib/libavutil.so;/usr/local/lib/libswscale.so" \
      \
      -D ENABLE_FAST_MATH=ON \
      -D WITH_CUFFT=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_V4L=ON \
      -D WITH_OPENGL=ON \
      -D WITH_GTK_3=ON \
      -D WITH_TBB=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      .. && \
make -j"$(nproc)" && make -j"$(nproc)" install && ldconfig && rm -rf /opencv /opencv_contrib

COPY --from=libtorch-extract /opt/libtorch /opt/libtorch

# --- TENSORRT SECTION ---
RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/8.6.1/tars/TensorRT-8.6.1.6.Linux.x86_64-gnu.cuda-11.8.tar.gz -O /tmp/TensorRT.tar.gz && \
    tar -xzf /tmp/TensorRT.tar.gz -C /opt && \
    rm /tmp/TensorRT.tar.gz

WORKDIR /app 
COPY . . 
RUN rm -rf build && mkdir build && cd build && cmake .. && make -j"$(nproc)"

RUN rm -f /usr/local/lib/libnvcuvid.so /usr/local/lib/libnvcuvid.so.1 \
      /usr/local/lib/libnvidia-encode.so /usr/local/lib/libnvidia-encode.so.1

#-----Final Stage -----
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    wget \
    libgtk2.0.0 libgtk-3-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh
ENV PATH=$CONDA_DIR/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1

# Create and activate a conda environment
RUN conda update -n base -c defaults conda && \
    conda create -y -n visage python=3.10 numpy pip && \
    conda clean -afy && conda install -n visage -c conda-forge opencv numpy -y

COPY --from=builder /usr/local/lib /usr/local/lib 
COPY --from=builder /opt/libtorch/libtorch/lib /opt/libtorch/libtorch/lib 
COPY --from=builder /opt/TensorRT-8.6.1.6/lib /opt/TensorRT-8.6.1.6/lib
COPY --from=builder /opt/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/bin/trtexec /usr/bin/trtexec
COPY --from=builder /app /app

ENV CONDA_DEFAULT_ENV=visage
RUN conda init bash && \
echo "conda activate visage" >> ~/.bashrc

SHELL ["conda", "run", "-n", "visage", "/bin/bash", "-c"]

ENV OpenCV_DIR=/usr/local/lib/cmake/opencv4
ENV LD_LIBRARY_PATH=/opt/TensorRT-8.6.1.6/targets/x86_64-linux-gnu/lib:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=/opt/TensorRT-8.6.1.6/lib:$LD_LIBRARY_PATH
ENV TENSORRT_DIR=/opt/TensorRT-8.6.1.6
ENV Torch_DIR=/opt/libtorch/libtorch/share/cmake/Torch

WORKDIR /app
