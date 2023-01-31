FROM pointcloudlibrary/env:22.04
LABEL Description="Build environment"

# Install via APT
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    zlib1g-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    git \
    zip \
    unzip \
    vim \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Build from source

# (i) PCL
ARG PCL_VERSION=1.13.0
ARG CMAKE_CXX_STANDARD=17
ARG CMAKE_BUILD_TYPE=Release

RUN wget -qO- https://github.com/PointCloudLibrary/pcl/archive/pcl-${PCL_VERSION}.tar.gz | tar xz \
 && cd pcl-pcl-${PCL_VERSION} \
 && mkdir build && cd build \
 && cmake .. -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
 && cmake --build . -- -j$(nproc) -k \
 && make -j$(nproc) install \
 && cd ../.. && rm -rf pcl-pcl-${PCL_VERSION}

# (ii) LAStools
RUN wget -q https://downloads.rapidlasso.de/LAStools.zip \
 && unzip LAStools.zip && rm LAStools.zip \
 && cd LAStools \
    # Fix the filename if the OS is case-sensitive
 && mv LASlib/src/LASlib-config.cmake LASlib/src/laslib-config.cmake \
    # Fix memset error
 && sed -i "s@memset(&(vlrs\[i\]), 0,@memset((void*)\&(vlrs[i]), 0,@" LASlib/inc/lasdefinitions.hpp \
 && mkdir build && cd build \
 && cmake .. \
 && cmake --build . -- -j$(nproc) -k \
 && make -j$(nproc) install \
 && cd ../.. && rm -rf LAStools

# (iii) OpenCV
ARG OpenCV_VERSION=4.6.0

RUN wget -qO- https://github.com/opencv/opencv/archive/${OpenCV_VERSION}.tar.gz | tar xz \
 && wget -qO- https://github.com/opencv/opencv_contrib/archive/${OpenCV_VERSION}.tar.gz | tar xz \
 && cd opencv-${OpenCV_VERSION} \
 && mkdir build && cd build \
 && cmake .. -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-${OpenCV_VERSION}/modules \
    -DWITH_OPENMP=ON -DBUILD_LIST="core,calib3d,imgproc,imgcodecs,highgui" \
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF -DBUILD_opencv_apps=OFF \
 && cmake --build . -- -j$(nproc) -k \
 && make -j$(nproc) install \
 && cd ../.. && rm -rf opencv-${OpenCV_VERSION} opencv_contrib-${OpenCV_VERSION}

# (iv) AWSSDK
ARG AWSSDK_VERSION=1.11.5

RUN wget -qO- https://github.com/aws/aws-sdk-cpp/archive/refs/tags/1.11.5.tar.gz | tar xz \
 && cd aws-sdk-cpp-${AWSSDK_VERSION} \
 && bash prefetch_crt_dependency.sh \
 && mkdir build && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=MinSizeRel -DBUILD_ONLY="s3" -DENABLE_TESTING=OFF \
 && cmake --build . -- -j$(nproc) -k \
 && make -j$(nproc) install \
 && cd ../.. && rm -rf aws-sdk-cpp-${AWSSDK_VERSION}

# Change user
RUN groupadd --gid 1000 docker \
 && useradd --uid 1000 --create-home -g docker docker \
 && chown -R docker:docker /home/docker

USER docker:docker

WORKDIR /home/docker/build

SHELL ["/bin/bash", "-c"]
