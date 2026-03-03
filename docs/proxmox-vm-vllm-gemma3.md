# Proxmox VM: vLLM + Gemma 3 27B on 1x MI50 32 GB (gfx906)

Step-by-step guide for a fresh Ubuntu 24.04 VM with GPU passthrough, running
vLLM with Gemma 3 27B on a single AMD Instinct MI50 32 GB card.

## Prerequisites

- Proxmox host with IOMMU enabled and `vfio-pci` configured
- One AMD MI50 32 GB passed through to the VM (same setup as your llama.cpp VM)
- Ubuntu 24.04 (Noble) ISO for the guest

## 1. Create the VM in Proxmox

Through the Proxmox web UI or CLI:

| Setting  | Value                                                        |
| -------- | ------------------------------------------------------------ |
| OS       | Ubuntu 24.04 Server                                          |
| CPU      | Host type, 8+ cores                                          |
| RAM      | 64 GB recommended (32 GB minimum)                            |
| Disk     | 120 GB+ (Docker images ~30 GB, model cache ~20 GB)           |
| Network  | virtio, bridged                                              |
| Machine  | q35                                                          |
| BIOS     | OVMF (UEFI)                                                  |

Add GPU passthrough in the VM hardware tab (or edit
`/etc/pve/qemu-server/<VMID>.conf`):

```
hostpci0: <GPU_PCI_ADDRESS>,pcie=1
```

Replace PCI addresses with your actual values (`lspci | grep AMD` on the host).

## 2. Base System Setup

Boot the VM, install Ubuntu 24.04, then:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential dkms linux-headers-$(uname -r) \
  wget curl git apt-transport-https ca-certificates gnupg
sudo reboot
```

## 3. Install ROCm Kernel Driver

Only the kernel-mode driver (amdgpu DKMS) is needed on the VM. The full ROCm
userspace lives inside the Docker container.

```bash
wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb
sudo apt install -y ./amdgpu-install_6.3.60303-1_all.deb
sudo amdgpu-install --usecase=dkms --no-32
sudo reboot
```

After reboot, verify the GPU is visible:

```bash
ls /dev/kfd /dev/dri/render*
```

Expected output (single GPU):

```
/dev/kfd  /dev/dri/renderD128
```

Add your user to the required groups:

```bash
sudo usermod -aG video $USER
sudo usermod -aG render $USER
```

Log out and back in for group changes to take effect.

## 4. Install Docker

```bash
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
```

Log out and back in, then verify:

```bash
docker run --rm hello-world
```

## 5. Pull the vLLM Image

The prebuilt image from the ML-gfx906 project includes ROCm 6.3.3, PyTorch
2.8.0, Triton 3.4.0, and vLLM 0.11.0 -- all compiled for gfx906:

```bash
docker pull docker.io/mixa3607/vllm-gfx906:0.11.0-rocm-6.3.3
```

This is ~20-30 GB. Grab a coffee.

## 6. Create a Model Cache Directory

```bash
mkdir -p ~/models
```

This directory will be mounted into the container so downloaded models persist
across container restarts.

## 7. Run vLLM with Gemma 3 27B

### Recommended: AWQ Quantized Model (single MI50 32 GB profile)

Use `gaunernst/gemma-3-27b-it-qat-autoawq`. It is the same Gemma 3 27B QAT
base model in AWQ format, which vLLM handles natively.

This document targets a single MI50 32 GB with conservative startup values.
After the model loads successfully, increase limits gradually.

Performance reference from the repo's benchmarks (2x MI50, 225W TDP):

| Workload     | Threads | AVG input | AVG output | Token Gen (tok/s) |
| ------------ | ------- | --------- | ---------- | ----------------- |
| Single user  | 1       | 16        | 256        | 19.1              |
| Batch        | 4       | 16        | 256        | 101.2             |
| Single user  | 1       | 4096      | 256        | 16.7              |
| Batch        | 4       | 4096      | 256        | 64.2              |

#### Step A: Apply the Gemma 3 float16 Patch

Gemma 3 on vLLM 0.11.0 needs a runtime patch for float16 stability. Start the
container in interactive mode first:

```bash
docker run -it \
  --name vllm-gemma3 \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  --shm-size 32g \
  --security-opt seccomp=unconfined \
  --cap-add SYS_PTRACE \
  -v ~/models:/root/.cache/huggingface \
  -p 8000:8000 \
  -e VLLM_USE_V1=1 \
  -e VLLM_USE_TRITON_AWQ=1 \
  -e VLLM_USE_TRITON_FLASH_ATTN=True \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e HUGGING_FACE_HUB_TOKEN=hf_YOUR_TOKEN_HERE \
  docker.io/mixa3607/vllm-gfx906:0.11.0-rocm-6.3.3 \
  bash
```

Inside the container, apply both patches:

```bash
# Patch 1: Allow float16 for Gemma 3
echo '
--- /usr/local/lib/python3.12/dist-packages/vllm/config/model.py
+++ /usr/local/lib/python3.12/dist-packages/vllm/config/model.py
@@ -1586,6 +1586,7 @@
     "plamo2": "Numerical instability. Please use bfloat16 or float32 instead.",
     "glm4": "Numerical instability. Please use bfloat16 or float32 instead.",
 }
+_FLOAT16_NOT_SUPPORTED_MODELS = {}
 
 
 def _is_valid_dtype(model_type: str, dtype: torch.dtype):' | patch -d/ -p0

# Patch 2: Clamp float16 values to prevent overflow in Gemma 3 decoder
echo '
--- /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma3.py
+++ /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma3.py
@@ -329,6 +329,9 @@ class Gemma3DecoderLayer(nn.Module):
         residual: Optional[torch.Tensor],
         **kwargs,
     ) -> tuple[torch.Tensor, torch.Tensor]:
+        # https://github.com/huggingface/transformers/pull/36832
+        if hidden_states.dtype == torch.float16:
+            hidden_states = hidden_states.clamp_(-65504, 65504)
         if residual is None:
             residual = hidden_states
             hidden_states = self.input_layernorm(hidden_states)
@@ -341,11 +344,15 @@ class Gemma3DecoderLayer(nn.Module):
             **kwargs,
         )
         hidden_states = self.post_attention_layernorm(hidden_states)
+        if hidden_states.dtype == torch.float16:
+            hidden_states = hidden_states.clamp_(-65504, 65504)
 
         hidden_states, residual = self.pre_feedforward_layernorm(
             hidden_states, residual)
         hidden_states = self.mlp(hidden_states)
         hidden_states = self.post_feedforward_layernorm(hidden_states)
+        if hidden_states.dtype == torch.float16:
+            hidden_states = hidden_states.clamp_(-65504, 65504)
         return hidden_states, residual' | patch -d/ -p0
```

#### Step B: Start vLLM (Still Inside the Container)

```bash
vllm serve gaunernst/gemma-3-27b-it-qat-autoawq \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
```

The first run will download the model (~15 GB) into `~/models` on the host.
Startup takes a few minutes while vLLM compiles kernels and loads weights.

If startup is stable, tune upward in this order:

1. `--gpu-memory-utilization 0.92` then `0.95`
2. `--max-model-len 8192` (then `16384` only if stable)
3. `--max-num-seqs 2`
4. Keep `--enable-prefix-caching` enabled for repeated prompt headers

#### One-Liner for Subsequent Runs

Once you've confirmed the patches work, you can script it into a single command:

```bash
docker run -d \
  --name vllm-gemma3 \
  --restart unless-stopped \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  --group-add render \
  --shm-size 32g \
  --security-opt seccomp=unconfined \
  --cap-add SYS_PTRACE \
  -v ~/models:/root/.cache/huggingface \
  -p 8000:8000 \
  -e VLLM_USE_V1=1 \
  -e VLLM_USE_TRITON_AWQ=1 \
  -e VLLM_USE_TRITON_FLASH_ATTN=True \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e HUGGING_FACE_HUB_TOKEN=hf_YOUR_TOKEN_HERE \
  docker.io/mixa3607/vllm-gfx906:0.11.0-rocm-6.3.3 \
  bash -c '
echo "--- /usr/local/lib/python3.12/dist-packages/vllm/config/model.py
+++ /usr/local/lib/python3.12/dist-packages/vllm/config/model.py
@@ -1586,6 +1586,7 @@
     \"plamo2\": \"Numerical instability. Please use bfloat16 or float32 instead.\",
     \"glm4\": \"Numerical instability. Please use bfloat16 or float32 instead.\",
 }
+_FLOAT16_NOT_SUPPORTED_MODELS = {}


 def _is_valid_dtype(model_type: str, dtype: torch.dtype):" | patch -d/ -p0 &&
echo "--- /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma3.py
+++ /usr/local/lib/python3.12/dist-packages/vllm/model_executor/models/gemma3.py
@@ -329,6 +329,9 @@
         residual: Optional[torch.Tensor],
         **kwargs,
     ) -> tuple[torch.Tensor, torch.Tensor]:
+        if hidden_states.dtype == torch.float16:
+            hidden_states = hidden_states.clamp_(-65504, 65504)
         if residual is None:
             residual = hidden_states
             hidden_states = self.input_layernorm(hidden_states)
@@ -341,11 +344,15 @@
             **kwargs,
         )
         hidden_states = self.post_attention_layernorm(hidden_states)
+        if hidden_states.dtype == torch.float16:
+            hidden_states = hidden_states.clamp_(-65504, 65504)

         hidden_states, residual = self.pre_feedforward_layernorm(
             hidden_states, residual)
         hidden_states = self.mlp(hidden_states)
         hidden_states = self.post_feedforward_layernorm(hidden_states)
+        if hidden_states.dtype == torch.float16:
+            hidden_states = hidden_states.clamp_(-65504, 65504)
         return hidden_states, residual" | patch -d/ -p0 &&
exec vllm serve gaunernst/gemma-3-27b-it-qat-autoawq \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --gpu-memory-utilization 0.90 \
  --enable-prefix-caching
'
```

### News summarization profile (recommended for your workload)

If most requests share the same instruction prefix (system prompt + template),
run vLLM with prefix caching:

```bash
vllm serve gaunernst/gemma-3-27b-it-qat-autoawq \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-seqs 2 \
  --gpu-memory-utilization 0.92 \
  --enable-prefix-caching
```

Notes:

- Prefix caching helps throughput when prompts start with identical token
  prefixes. It does not eliminate KV cache usage for each article body.
- Keep request format deterministic (same whitespace/template/order) so token
  prefixes match exactly.
- If you hit OOM, first lower `--max-num-seqs` from `2` to `1`, then reduce
  `--max-model-len`.

### How many articles at the same time? (1x MI50 32 GB, 8 vCPU)

For Gemma 3 27B AWQ on a single MI50 32 GB, practical concurrent summaries are:

- **Safe default:** 1 concurrent article (`--max-num-seqs 1`)
- **Usually feasible:** 2 concurrent articles (`--max-num-seqs 2`, 4K context)
- **3+ concurrent:** uncommon for 27B unless article length/output limits are
  aggressively reduced

Rule of thumb for news summarization:

- Typical article 800-1500 input tokens, output 200-400 tokens:
  expect **1-2 concurrent** requests.
- Very long articles or high output limits:
  expect **1 concurrent** request.

CPU (8 vCPU) is adequate here; VRAM/KV cache is the primary limiter.

### Why Not GGUF?

The linked model [unsloth/gemma-3-27b-it-qat-GGUF](https://huggingface.co/unsloth/gemma-3-27b-it-qat-GGUF)
(`Q4_K_M`, 16.5 GB) is designed for **llama.cpp**, not vLLM. While vLLM has
experimental GGUF support, it dequantizes weights to fp16 at load time -- the
27B model would expand to ~54 GB in VRAM, far exceeding one MI50 32 GB.

If you want to use GGUF Gemma 3, use the llama.cpp image from this repo instead:

```bash
docker pull docker.io/mixa3607/llama.cpp-gfx906:full-b7091-rocm-6.3.3
```

The AWQ model (`gaunernst/gemma-3-27b-it-qat-autoawq`) is the same underlying
Gemma 3 27B QAT model quantized in a format vLLM handles natively. Same model
quality, correct format for the engine.

## 8. Verify It's Working

```bash
# Watch logs
docker logs -f vllm-gemma3

# Once you see "Uvicorn running on http://0.0.0.0:8000", test it:
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gaunernst/gemma-3-27b-it-qat-autoawq",
    "messages": [{"role": "user", "content": "Explain quantum entanglement in simple terms."}],
    "max_tokens": 256
  }'
```

The vLLM server exposes an OpenAI-compatible API on port 8000. You can point
any OpenAI client library, Open WebUI, or similar tool at
`http://<VM_IP>:8000/v1`.

### OOM Troubleshooting (single MI50 32 GB)

If you see OOM or allocator failures, reduce in this order:

1. Lower `--max-model-len` to `2048`
2. Lower `--gpu-memory-utilization` to `0.85`
3. Keep `--max-num-seqs 1`
4. Disable parallel requests at the client side (single request at a time)

## 9. Performance Tuning (Optional)

Reduce GPU hotspot temperature by ~10°C with negligible performance impact:

```bash
# Install the tool (on the host or inside a privileged container)
pip install upp

# Apply to your GPU
upp -p /sys/class/drm/card0/device/pp_table set --write smcPPTable/TdcLimitGfx=150
```

This resets on reboot. To persist, add it to `/etc/rc.local` or a systemd unit.

## 10. Useful Environment Variables

| Variable                     | Value  | Purpose                                      |
| ---------------------------- | ------ | -------------------------------------------- |
| `VLLM_USE_V1`               | `1`    | Use vLLM v1 engine                           |
| `VLLM_USE_TRITON_AWQ`       | `1`    | Triton kernels for AWQ (required for gfx906) |
| `VLLM_USE_TRITON_FLASH_ATTN`| `True` | Triton flash attention (required for gfx906) |
| `HIP_FORCE_DEV_KERNARG`     | `1`    | ROCm kernel arg optimization                 |
| `VLLM_SLEEP_WHEN_IDLE`      | `1`    | Reduce GPU power when idle                   |

## Version Compatibility (from repo testing)

| ROCm  | PyTorch | vLLM  | Text | Images | Notes                    |
| ----- | ------- | ----- | ---- | ------ | ------------------------ |
| 6.3.3 | 2.8.0   | 0.11.0| OK   | OK     | **Recommended**          |
| 6.3.3 | 2.7.1   | 0.10.2| OK   | OK     | Older, works             |
| 6.4.4 | 2.8.0   | 0.11.0| Fail | Fail   | All requests throw error |
| 6.4.4 | 2.7.1   | 0.10.2| OK   | Fail   | Image requests break     |

Stick with **ROCm 6.3.3**. It is the only fully working base for vLLM on gfx906.

## References

- [ML-gfx906 repo](https://github.com/mixa3607/ML-gfx906)
- [vLLM gfx906 fork (nlzy)](https://github.com/nlzy/vllm-gfx906)
- [Triton gfx906 fork (nlzy)](https://github.com/nlzy/triton-gfx906)
- [gaunernst/gemma-3-27b-it-qat-autoawq](https://huggingface.co/gaunernst/gemma-3-27b-it-qat-autoawq)
- [unsloth/gemma-3-27b-it-qat-GGUF](https://huggingface.co/unsloth/gemma-3-27b-it-qat-GGUF) (for llama.cpp, not vLLM)
- [ROCm vLLM Docker docs](https://github.com/ROCm/vllm/blob/main/docs/deployment/docker.md)
