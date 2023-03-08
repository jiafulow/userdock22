FROM pointcloudlibrary/env:22.04
LABEL Description="Build environment"

# Install via APT
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    zlib1g-dev \
    libssl-dev \
    libcurl4-openssl-dev \
    git \
    zip \
    unzip \
    vim \
    curl \
    # This was removed from the base image on Feb 20, 2023
    # but still needed by PCL 1.13.0
    libboost-date-time-dev \
 && rm -rf /var/lib/apt/lists/*

# Build from source
ARG CMAKE_CXX_STANDARD=17
ARG CMAKE_BUILD_TYPE=Release

# (i) PCL
ARG PCL_VERSION=1.13.0

RUN wget -qO- https://github.com/PointCloudLibrary/pcl/archive/pcl-${PCL_VERSION}.tar.gz | tar xz \
 && cd pcl-pcl-${PCL_VERSION} \
 && mkdir build && cd build \
 && cmake .. -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
    -DPCL_ONLY_CORE_POINT_TYPES=ON -DPCL_DISABLE_GPU_TESTS=ON -DPCL_DISABLE_VISUALIZATION_TESTS=ON \
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
 && cmake .. -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
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
    -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_DOCS=OFF \
    -DBUILD_opencv_apps=OFF -DBUILD_opencv_js=OFF -DBUILD_JAVA=OFF -DBUILD_OBJC=OFF \
    -DWITH_OPENEXR=OFF \
 && cmake --build . -- -j$(nproc) -k \
 && make -j$(nproc) install \
 && cd ../.. && rm -rf opencv-${OpenCV_VERSION} opencv_contrib-${OpenCV_VERSION}

# (iv) CCCoreLib
ARG CCCoreLib_SHA=5990bae

RUN git clone --single-branch --branch=master https://github.com/CloudCompare/CCCoreLib \
 && cd CCCoreLib && git checkout -b integration ${CCCoreLib_SHA} \
 && mkdir build && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} -DCCCORELIB_SCALAR_DOUBLE=OFF \
 && cmake --build . -- -j$(nproc) -k \
 && make -j$(nproc) install \
 && cd ../.. && rm -rf CCCoreLib/

# (v) AWSSDK
ARG AWSSDK_VERSION=1.11.35
ARG AWSSDK_BUILD_TYPE=MinSizeRel

RUN wget -qO- https://github.com/aws/aws-sdk-cpp/archive/${AWSSDK_VERSION}.tar.gz | tar xz \
 && cd aws-sdk-cpp-${AWSSDK_VERSION} \
 && sed -i "s/curl /curl --no-progress-meter /" ./prefetch_crt_dependency.sh \
 && ./prefetch_crt_dependency.sh > /dev/null \
 && mkdir build && cd build \
 && cmake .. -DCMAKE_BUILD_TYPE=${AWSSDK_BUILD_TYPE} -DBUILD_ONLY="s3" -DENABLE_TESTING=OFF \
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
