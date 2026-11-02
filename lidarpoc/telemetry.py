"""Read host GPU usage through NVIDIA's maintained NVML bindings, without Torch."""
import time


def gpu_usage():
    snapshot = {'timestamp': time.time(), 'gpus': [], 'error': None}
    try:
        import pynvml as nvml  # Provided by nvidia-ml-py, not the deprecated pynvml package.
    except ImportError:
        snapshot['error'] = 'Install nvidia-ml-py in the server environment to enable GPU monitoring'
        return snapshot

    def query(function, *args):
        try:
            return function(*args)
        except nvml.NVMLError:
            return None

    def text(value):
        return value.decode() if isinstance(value, bytes) else value

    try:
        nvml.nvmlInit()
    except nvml.NVMLError as error:
        snapshot['error'] = str(error)
        return snapshot
    try:
        for index in range(nvml.nvmlDeviceGetCount()):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            utilization = query(nvml.nvmlDeviceGetUtilizationRates, handle)
            memory = query(nvml.nvmlDeviceGetMemoryInfo, handle)
            power = query(nvml.nvmlDeviceGetPowerUsage, handle)
            snapshot['gpus'].append({
                'id': index, 'uuid': text(nvml.nvmlDeviceGetUUID(handle)),
                'name': text(nvml.nvmlDeviceGetName(handle)),
                'utilization_percent': utilization.gpu if utilization is not None else None,
                'memory_used_gb': memory.used / 1024**3 if memory is not None else None,
                'memory_total_gb': memory.total / 1024**3 if memory is not None else None,
                'temperature_c': query(nvml.nvmlDeviceGetTemperature, handle, nvml.NVML_TEMPERATURE_GPU),
                'power_w': power / 1000 if power is not None else None,
            })
    except nvml.NVMLError as error:
        snapshot['error'] = str(error)
    finally:
        nvml.nvmlShutdown()
    return snapshot
