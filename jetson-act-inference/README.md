# Jetson ACT Inference

Minimal standalone inference for a trained ACT policy (ONNX) on Jetson Orin Nano.
No PyTorch or LeRobot dependency — only ONNX Runtime, OpenCV, and numpy.

## Hardware

- **Board**: NVIDIA Jetson Orin Nano
- **Robot**: SO-101 arm (6× Feetech STS3215 servos, USB serial @ 1 Mbaud)
- **Camera**: Intel RealSense (wrist-mounted)

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

For Jetson, install `onnxruntime-gpu` from NVIDIA's wheels:
```bash
pip install onnxruntime-gpu --extra-index-url https://elinux.org/Jetson_Zoo
```

### 2. Export your trained model

On your training machine (with PyTorch + LeRobot installed):

```bash
python scripts/export_onnx.py \
    --input path/to/trained_model \
    --output jetson-act-inference/model/act_policy.onnx
```

Copy the preprocessor/postprocessor safetensors from your trained model directory:

```bash
cp path/to/trained_model/policy_preprocessor_step_*_normalizer.safetensors \
   jetson-act-inference/model/policy_preprocessor.safetensors

cp path/to/trained_model/policy_postprocessor_step_*_normalizer.safetensors \
   jetson-act-inference/model/policy_postprocessor.safetensors
```

### 3. Calibrate your robot

Run LeRobot calibration on the Jetson (or copy the file from another machine):

```bash
# Calibration file location:
~/.lerobot/calibration/so101.json
```

### 4. Verify config

Edit `config.yaml` to match your setup:
- `motors.port` — USB serial port (e.g., `/dev/ttyUSB0`)
- `motors.robot_id` — matches calibration filename
- `camera.input_name` — matches the ONNX model's image input name
- `inference.state_input_name` — matches the ONNX model's state input name
- `model.chunk_size` / `model.action_dim` — from your trained model config

You can inspect input/output names with [Netron](https://netron.app) or:
```python
import onnxruntime as ort
s = ort.InferenceSession("model/act_policy.onnx")
print([i.name for i in s.get_inputs()])
print([o.name for o in s.get_outputs()])
```

## Run

```bash
python inference.py                   # Normal operation
python inference.py --dry-run         # Camera + model only, no motor writes
python inference.py --config my.yaml  # Custom config file
```

## File Structure

```
jetson-act-inference/
├── config.yaml          # All tunable parameters
├── requirements.txt     # Python dependencies
├── inference.py         # Main 30 Hz control loop
├── camera.py            # RealSense RGB capture
├── motors.py            # Feetech STS3215 read/write + calibration
├── preprocessing.py     # Image resize/normalize, state normalize
├── postprocessing.py    # Action denormalize + temporal ensemble
├── utils.py             # Config & safetensors loading
├── README.md
└── model/               # (you provide these)
    ├── act_policy.onnx
    ├── policy_preprocessor.safetensors
    └── policy_postprocessor.safetensors
```

## Data Flow

```
Camera frame (640×480 RGB uint8)
  → resize to 224×224
  → float32 / 255
  → HWC → CHW
  → normalize with preprocessor stats
                                        ╲
                                         → ONNX model → (1, chunk_size, 6)
                                        ╱                      ↓
Motor positions (6× raw servo values)                  denormalize actions
  → decode sign-magnitude                                      ↓
  → calibration normalize                           temporal ensemble (optional)
    (degrees / 0-100)                                          ↓
  → normalize with preprocessor stats               take action (action_dim,)
                                                               ↓
                                                    reverse calibration normalize
                                                               ↓
                                                    encode sign-magnitude
                                                               ↓
                                                    sync_write Goal_Position
                                                               ↓
                                                         loop @ 30 Hz
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
