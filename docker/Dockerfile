# CaP-X development image: the system libraries, CUDA toolkit and uv the checkout
# needs, and nothing of the checkout itself.
# Base: NGC PyTorch 25.02 (ships CUDA 12.8 toolkit, nvcc, CUDA_HOME=/usr/local/cuda)
# run.sh bind-mounts the host checkout at /workspace/cap-x, and docker/setup-*.sh
# create its venvs there with uv (per uv.lock), so they outlive the container.
# The NGC torch in the base image is NOT used.
FROM nvcr.io/nvidia/pytorch:25.02-py3

# Target GPU compute capability, overridable at build time:
#   Blackwell DC (RTX PRO 6000 Blackwell)  -> 12.0   (default)
#   Ada       (RTX 6000 Ada, L40, L40S)    -> 8.9
#   Hopper    (H100, H200)                 -> 9.0
#   Ampere    (A100, A40, A6000)           -> 8.0 or 8.6
# Use:  docker build --build-arg TORCH_CUDA_ARCH_LIST=8.9 ...
ARG TORCH_CUDA_ARCH_LIST=12.0

ENV DEBIAN_FRONTEND=noninteractive \
    MUJOCO_GL=egl \
    TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- system deps ---------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs ffmpeg wget curl vim htop tmux screen ca-certificates \
        build-essential cmake pkg-config unzip \
        libsm6 libxext6 libxrender1 libglu1-mesa \
        libgl1 libglx0 libegl1 libopengl0 libosmesa6 \
        libx11-6 libxi6 libxcursor1 libxrandr2 libxinerama1 libxxf86vm1 \
        libjpeg-dev libpng-dev libopenexr-dev libtiff-dev \
        libavcodec-dev libavformat-dev libswscale-dev \
        libavutil-dev libavdevice-dev libswresample-dev libpostproc-dev \
        libv4l-dev libxvidcore-dev libx264-dev \
        libxt6 libvulkan1 vulkan-tools zenity \
    && rm -rf /var/lib/apt/lists/*

# --- uv ------------------------------------------------------------------
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}" \
    UV_LINK_MODE=copy \
    UV_PYTHON_INSTALL_DIR=/opt/uv-python

# cap-x wants Python 3.10 for base/robosuite/b1k and 3.12 for libero + vLLM.
# Bake BOTH into the image so host-mounted venvs (whose `bin/python` is a
# symlink into /opt/uv-python) resolve after a container restart.
RUN uv python install 3.10 3.12

# Allow git operations on bind-mounted host repos (UID mismatch otherwise
# triggers "dubious ownership" errors during setuptools-scm describe).
RUN git config --global --add safe.directory '*'

# NVIDIA Container Toolkit with driver >= 570 does not always inject Vulkan /
# EGL ICD JSONs. Isaac Sim needs both at runtime — hand-write them so the
# NVIDIA GL/EGL libraries that ARE injected become discoverable.
RUN mkdir -p /usr/share/vulkan/icd.d /usr/share/glvnd/egl_vendor.d \
 && printf '%s\n' '{' \
    '    "file_format_version" : "1.0.0",' \
    '    "ICD": {' \
    '        "library_path" : "libGLX_nvidia.so.0",' \
    '        "api_version" : "1.3.289"' \
    '    }' \
    '}' > /usr/share/vulkan/icd.d/nvidia_icd.json \
 && printf '%s\n' '{' \
    '    "file_format_version" : "1.0.0",' \
    '    "ICD": {' \
    '        "library_path" : "libEGL_nvidia.so.0"' \
    '    }' \
    '}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

# --- the mounted checkout ----------------------------------------------
# The base venv docker/setup-robosuite.sh creates in the checkout, active once it
# exists. NGC's CUDA_HOME is what lets the editable CUDA builds (nvidia-curobo,
# sam3, contact_graspnet_pytorch) succeed there.
ENV VIRTUAL_ENV=/workspace/cap-x/.venv \
    PATH="/workspace/cap-x/.venv/bin:${PATH}"

WORKDIR /workspace/cap-x
CMD ["/bin/bash"]
