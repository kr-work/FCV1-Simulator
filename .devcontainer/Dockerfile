FROM mcr.microsoft.com/devcontainers/cpp:1-debian-11

ARG REINSTALL_CMAKE_VERSION_FROM_SOURCE="none"

# Optionally install the cmake for vcpkg
COPY ./reinstall-cmake.sh /tmp/

RUN if [ "${REINSTALL_CMAKE_VERSION_FROM_SOURCE}" != "none" ]; then \
        chmod +x /tmp/reinstall-cmake.sh && /tmp/reinstall-cmake.sh ${REINSTALL_CMAKE_VERSION_FROM_SOURCE}; \
    fi \
    && rm -f /tmp/reinstall-cmake.sh \
    && apt-get update \
    && apt-get install python3-dev libx11-dev xorg-dev libglu1-mesa-dev doxygen -y \
    && apt-get install -y build-essential libssl-dev zlib1g-dev libncurses5-dev libffi-dev libsqlite3-dev libreadline-dev libbz2-dev wget curl \
    && wget https://www.python.org/ftp/python/3.12.0/Python-3.12.0.tgz \
    && tar xzf Python-3.12.0.tgz \
    && cd Python-3.12.0 \
    && ./configure --enable-optimizations \
    && make -j$(nproc) \
    && make altinstall \
    && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* \
    && apt-get clean \
    && cd .. && rm -rf Python-3.12.0 Python-3.12.0.tgz

# update-alternativesを使ってpython3を3.12に変更
RUN update-alternatives --install /usr/bin/python3 python3 /usr/local/bin/python3.12 1 \
    && update-alternatives --set python3 /usr/local/bin/python3.12 \
    && python3.12 -m ensurepip \
    && python3.12 -m pip install --upgrade pip \
    && pip3 install pybind11[global]

# [Optional] Uncomment this section to install additional vcpkg ports.
# RUN su vscode -c "${VCPKG_ROOT}/vcpkg install <your-port-name-here>"

# [Optional] Uncomment this section to install additional packages.
# RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
#     && apt-get -y install --no-install-recommends <your-package-list-here>
