# Jetson ACT Inference

Minimal standalone inference for a trained ACT policy on Jetson Orin Nano.
No PyTorch or LeRobot dependency — uses TensorRT (preinstalled via JetPack) for GPU inference.

## Hardware

- **Board**: NVIDIA Jetson Orin Nano (8GB, JetPack 6 / L4T R36.4, TensorRT 10.3)
- **Robot**: SO-101 arm (6× Feetech STS3215 servos, `/dev/ttyACM0` @ 1 Mbaud)
- **Wrist camera**: Intel RealSense D405 (640×480 @ 30fps, via pyrealsense2)
- **Front camera**: e-con See3CAM_24CUG USB camera (1280×720 @ 60fps, `/dev/video6`, mounted upside down → flipped in software)

## Setup

### 1. Create venv with system packages (required for TensorRT access)

```bash
python3 -m venv --system-site-packages .edge_act_venv
source .edge_act_venv/bin/activate
pip install -r requirements.txt
```

### 2. Model files

Place these in the `model/` directory:
- `model.onnx` — ONNX export of your trained ACT policy
- `policy_preprocessor_step_3_normalizer_processor.safetensors` — normalization stats
- `policy_postprocessor_step_0_unnormalizer_processor.safetensors` — denormalization stats

#### Exporting ONNX from a trained ACT policy

On your training machine (where PyTorch and LeRobot are installed), use the provided export script:

```bash
python scripts/export_onnx.py --input <path/to/pretrained_model> --output model/model.onnx
```

| Argument    | Required | Description                                      |
|-------------|----------|--------------------------------------------------|
| `--input`   | Yes      | Path to the pretrained ACT model directory        |
| `--output`  | Yes      | Path for the exported ONNX file                   |
| `--opset`   | No       | ONNX opset version (default: 18)                  |

The script automatically discovers model inputs from the policy config and exports with dynamic batch axes. Copy the resulting `.onnx` file and the processor `.safetensors` files from the model directory to `model/` on the Jetson.

### 3. Build TensorRT engine

The Orin Nano only has 8GB shared RAM. You need swap enabled:

```bash
sudo swapon /swapfile
```

Optionally free RAM by stopping the desktop:
```bash
sudo systemctl stop gdm3
```

Then build the engine:
```bash
/usr/src/tensorrt/bin/trtexec \
    --onnx=model/model.onnx \
    --saveEngine=model/act_policy.engine \
    --fp16 \
    --memPoolSize=workspace:512 \
    --builderOptimizationLevel=2 \
    --skipInference
```

This takes 30-60+ minutes due to swap pressure. Monitor with `top` — `trtexec` will use ~5GB RSS and thrash swap. As long as CPU TIME is incrementing, it's working.

### 4. Calibrate your robot

Run calibration via LeRobot on your training machine and copy to:
```
~/.cache/huggingface/lerobot/calibration/robots/so_follower/thing.json
```

### 5. Verify config

Edit `config.yaml`:
- `motors.port` — serial port (`/dev/ttyACM0`)
- `camera.front.device` — V4L2 device index (check with `cat /sys/class/video4linux/video*/name`)
- `camera.front.flip` — set `true` if camera is upside down

### 6. Permissions

```bash
sudo chmod 666 /dev/ttyACM0
# Or permanently: sudo usermod -aG dialout $USER (requires re-login)
```

## Run

```bash
python inference.py                   # Full live inference with motor control
python inference.py --dry-run         # Model only, no cameras/motors (uses dummy data)
python inference.py --config my.yaml  # Custom config file
```

### Test scripts

```bash
python preview_cameras.py             # Live camera preview (both streams, raw + model crop)
python test_motors_inference.py       # Read joints + 20 inference steps, no motor writes
```

## Performance

- **Inference**: ~50-67ms per step (~15-20 Hz) on Orin Nano with FP16 TensorRT engine
- **First step**: ~155ms (CUDA warmup)
- Model inputs: `observation.state` (1×6), `observation.images.front` (1×3×352×640), `observation.images.wrist` (1×3×480×640)
- Model output: `actions` (1×100×6) — 100-step action chunk

## File Structure

```
jetson-act-inference/
├── config.yaml              # All tunable parameters
├── requirements.txt         # Python dependencies (no onnxruntime, no torch)
├── inference.py             # Main control loop (~15 Hz)
├── trt_runtime.py           # TensorRT GPU backend (ctypes + tensorrt)
├── camera.py                # RealSenseCamera + CSICamera (V4L2 fallback)
├── motors.py                # Feetech STS3215 sync read/write + calibration
├── preprocessing.py         # Image resize/normalize, state normalize
├── postprocessing.py        # Action denormalize + temporal ensemble
├── utils.py                 # Config & safetensors loading
├── convert_engine.sh        # trtexec wrapper script
├── preview_cameras.py       # Camera preview utility
├── test_motors_inference.py # Test: read joints + run inference (no writes)
└── model/
    ├── model.onnx
    ├── act_policy.engine
    ├── policy_preprocessor_step_3_normalizer_processor.safetensors
    └── policy_postprocessor_step_0_unnormalizer_processor.safetensors
```

## Data Flow

```
Front camera (1280×720 RGB)           Wrist camera (640×480 RGB)
  → flip 180° (upside down)
  → resize to 640×352                   → (no resize needed)
  → float32 / 255                       → float32 / 255
  → HWC → CHW                           → HWC → CHW
  → mean_std normalize                  → mean_std normalize
          ╲                                    ╱
           ╲                                  ╱
            → TensorRT FP16 engine (GPU) ←──╱
           ╱         ↑                    ╲
          ╱          │                     ╲
Motor positions      │                      → actions (1, 100, 6)
  → sync_read        │                           ↓
  → decode sign-mag  │                    denormalize (mean_std)
  → calibration norm │                           ↓
  → mean_std norm ───╯                 temporal ensemble (w_i = exp(-0.01*i))
                                                 ↓
                                          action (6,)
                                                 ↓
                                       reverse calibration norm
                                                 ↓
                                       encode sign-magnitude
                                                 ↓
                                       sync_write Goal_Position
                                                 ↓
                                           loop @ ~15 Hz
```

## Motor Joint Mapping

| Joint          | ID | Norm Mode    |
|----------------|----|-------------|
| shoulder_pan   | 1  | degrees     |
| shoulder_lift  | 2  | degrees     |
| elbow_flex     | 3  | degrees     |
| wrist_flex     | 4  | degrees     |
| wrist_roll     | 5  | degrees     |
| gripper        | 6  | range_0_100 |

## Troubleshooting

### `No module named 'tensorrt'`
Your venv wasn't created with `--system-site-packages`. Recreate it:
```bash
rm -rf .edge_act_venv
python3 -m venv --system-site-packages .edge_act_venv
source .edge_act_venv/bin/activate
pip install -r requirements.txt
```

### trtexec OOM / killed during engine build
Enable swap and reduce optimization level:
```bash
sudo swapon /swapfile  # ensure swap is active
/usr/src/tensorrt/bin/trtexec --onnx=model/model.onnx --saveEngine=model/act_policy.engine \
    --fp16 --memPoolSize=workspace:256 --builderOptimizationLevel=1 --skipInference
```

### Permission denied on `/dev/ttyACM0`
```bash
sudo chmod 666 /dev/ttyACM0
```

### Camera device index wrong
```bash
cat /sys/class/video4linux/video*/name
```
RealSense uses 6 nodes; the USB camera is typically the next one after those.

### No CSI camera detected
MIPI CSI cameras need matching device tree overlays. The D-Robotics stereo module is not compatible with Jetson — use a USB camera instead.
