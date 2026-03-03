#!/bin/bash

export VLLM_ROCM_VERSION="6.3.4"
export VLLM_PYTORCH_VERSION="v2.9.1"
export VLLM_REPO="https://github.com/ai-infos/vllm-gfx906-mobydick.git"
export VLLM_BRANCH="main"
export VLLM_TRITON_REPO="https://github.com/ai-infos/triton-gfx906.git"
export VLLM_TRITON_BRANCH="v3.5.1+gfx906"
export VLLM_PRESET_NAME="latest-rocm-$VLLM_ROCM_VERSION"
