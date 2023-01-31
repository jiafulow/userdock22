FROM pointcloudlibrary/env:22.04
LABEL Description="Build environment"

# Install via APT
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      zlib1g-dev \
      libssl-dev \
      libcurl4-openssl-dev \
      git \
      zip \
      vim \
      curl

# Build from source
ARG PCL_VERSION=1.13.0
ARG CMAKE_CXX_STANDARD=17
ARG CMAKE_BUILD_TYPE=Release

RUN wget -qO- https://github.com/PointCloudLibrary/pcl/archive/pcl-${PCL_VERSION}.tar.gz | tar xz && \
    cd pcl-pcl-${PCL_VERSION} && \
    mkdir build && \
    cd build && \
    cmake .. -DCMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} && \
    cmake --build . -- -j$(nproc) -k && \
    make -j$(nproc) install && \
    cd ../.. && \
    rm -rf pcl-pcl-${PCL_VERSION}

# Change user
RUN groupadd --gid 1000 docker && \
    useradd --uid 1000 --create-home -g docker docker && \
    chown -R docker:docker /home/docker

USER docker:docker

WORKDIR /home/docker/build

SHELL ["/bin/bash", "-c"]
