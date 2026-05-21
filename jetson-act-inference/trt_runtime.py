"""TensorRT inference backend for Jetson.

Uses only preinstalled JetPack components:
  - tensorrt (Python bindings, preinstalled)
  - libcudart.so (CUDA runtime, preinstalled)

No onnxruntime, no pycuda, no extra pip packages needed.
"""

import ctypes
import numpy as np

# Load CUDA runtime via ctypes — always present on Jetson
_cudart = ctypes.CDLL("libcudart.so")

# Declare function signatures so ctypes handles 64-bit pointers correctly on aarch64
_cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
_cudart.cudaMalloc.restype = ctypes.c_int

_cudart.cudaFree.argtypes = [ctypes.c_void_p]
_cudart.cudaFree.restype = ctypes.c_int

_cudart.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
_cudart.cudaMemcpy.restype = ctypes.c_int

_cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
_cudart.cudaStreamCreate.restype = ctypes.c_int

_cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
_cudart.cudaStreamSynchronize.restype = ctypes.c_int

_cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
_cudart.cudaStreamDestroy.restype = ctypes.c_int

_cudart.cudaSetDevice.argtypes = [ctypes.c_int]
_cudart.cudaSetDevice.restype = ctypes.c_int


class _CudaBuffer:
    """GPU memory buffer managed via ctypes calls to libcudart."""

    def __init__(self, nbytes: int):
        self.nbytes = nbytes
        self.ptr = ctypes.c_void_p()
        ret = _cudart.cudaMalloc(ctypes.byref(self.ptr), ctypes.c_size_t(nbytes))
        if ret != 0:
            raise RuntimeError(f"cudaMalloc({nbytes}) failed with error {ret}")

    def copy_from_host(self, arr: np.ndarray) -> None:
        ret = _cudart.cudaMemcpy(
            self.ptr,
            ctypes.c_void_p(arr.ctypes.data),
            ctypes.c_size_t(arr.nbytes),
            ctypes.c_int(1),  # cudaMemcpyHostToDevice
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy H→D failed with error {ret}")

    def copy_to_host(self, arr: np.ndarray) -> None:
        ret = _cudart.cudaMemcpy(
            ctypes.c_void_p(arr.ctypes.data),
            self.ptr,
            ctypes.c_size_t(arr.nbytes),
            ctypes.c_int(2),  # cudaMemcpyDeviceToHost
        )
        if ret != 0:
            raise RuntimeError(f"cudaMemcpy D→H failed with error {ret}")

    def free(self) -> None:
        if self.ptr:
            _cudart.cudaFree(self.ptr)
            self.ptr = None

    def __del__(self):
        self.free()


class TRTSession:
    """TensorRT inference session.

    Drop-in replacement for onnxruntime.InferenceSession with the same
    .run(output_names, feeds) interface.

    Usage:
        session = TRTSession("model/act_policy.engine")
        outputs = session.run(None, {"observation.images.camera": img, "observation.state": state})
        actions = outputs[0]
    """

    def __init__(self, engine_path: str):
        self._buffers = {}

        # Ensure CUDA device is initialized
        ret = _cudart.cudaSetDevice(ctypes.c_int(0))
        if ret != 0:
            raise RuntimeError(f"cudaSetDevice(0) failed with error {ret}")

        import tensorrt as trt

        self._trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(f"Failed to load TensorRT engine from {engine_path}")

        self.context = self.engine.create_execution_context()

        # Create a CUDA stream
        self._stream = ctypes.c_void_p()
        ret = _cudart.cudaStreamCreate(ctypes.byref(self._stream))
        if ret != 0:
            raise RuntimeError(f"cudaStreamCreate failed with error {ret}")
        self._stream_int = self._stream.value or 0  # int for TensorRT API

        # Discover I/O tensors
        self.input_names: list[str] = []
        self.output_names: list[str] = []
        self._buffers: dict[str, _CudaBuffer] = {}
        self._shapes: dict[str, tuple] = {}
        self._dtypes: dict[str, np.dtype] = {}

        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))

            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
                shape = self.engine.get_tensor_shape(name)
                # Handle dynamic shapes: use profile 0 optimal shape
                if -1 in shape:
                    _, opt, _ = self.engine.get_tensor_profile_shape(name, 0)
                    self.context.set_input_shape(name, opt)
                    shape = opt
                self._shapes[name] = tuple(shape)
            else:
                self.output_names.append(name)

            self._dtypes[name] = dtype

        # Infer output shapes (after input shapes are set)
        for name in self.output_names:
            self._shapes[name] = tuple(self.context.get_tensor_shape(name))

        # Allocate GPU buffers for all tensors
        for name in self.input_names + self.output_names:
            shape = self._shapes[name]
            nbytes = int(np.prod(shape)) * self._dtypes[name].itemsize
            buf = _CudaBuffer(nbytes)
            self._buffers[name] = buf
            self.context.set_tensor_address(name, buf.ptr.value)

        print(f"TensorRT engine loaded: {engine_path}")
        print(f"  Inputs:  { {n: self._shapes[n] for n in self.input_names} }")
        print(f"  Outputs: { {n: self._shapes[n] for n in self.output_names} }")

    def run(self, output_names: list[str] | None, feeds: dict[str, np.ndarray]) -> list[np.ndarray]:
        """Run inference. Same signature as onnxruntime.InferenceSession.run().

        Args:
            output_names: list of output tensor names, or None for all.
            feeds: dict mapping input names to numpy arrays.

        Returns:
            List of output numpy arrays.
        """
        # Copy inputs to GPU
        for name, arr in feeds.items():
            arr = np.ascontiguousarray(arr, dtype=self._dtypes[name])
            expected = int(np.prod(self._shapes[name])) * self._dtypes[name].itemsize
            if arr.nbytes != expected:
                raise ValueError(
                    f"Input '{name}': got {arr.nbytes} bytes (shape {arr.shape}), "
                    f"expected {expected} bytes (shape {self._shapes[name]})"
                )
            self._buffers[name].copy_from_host(arr)

        # Execute
        stream_ptr = self._stream_int
        ok = self.context.execute_async_v3(stream_ptr)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        ret = _cudart.cudaStreamSynchronize(self._stream)
        if ret != 0:
            raise RuntimeError(f"cudaStreamSynchronize failed with error {ret}")

        # Copy outputs to host
        names = output_names or self.output_names
        results = []
        for name in names:
            out = np.empty(self._shapes[name], dtype=self._dtypes[name])
            self._buffers[name].copy_to_host(out)
            results.append(out)
        return results

    def get_providers(self) -> list[str]:
        return ["TensorrtExecutionProvider"]

    def close(self) -> None:
        for buf in self._buffers.values():
            buf.free()
        self._buffers.clear()
        if self._stream:
            _cudart.cudaStreamDestroy(self._stream)
            self._stream = None

    def __del__(self):
        self.close()
