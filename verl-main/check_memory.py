import time
import pynvml

try:
    pynvml.nvmlInit()
    
    while True:
        device_count = pynvml.nvmlDeviceGetCount()
        is_begin = True
        for i in range(device_count):
            if i == 3:
                continue
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            used_gb = mem_info.used / 1024**3
            total_gb = mem_info.total / 1024**3
            if used_gb > 1:
                if is_begin:
                    print('-'*40)
                    current_time = time.strftime("%Y-%m-%d %H:%M:%S")
                    print(f"[{current_time}]")
                    is_begin = False
                print(f"GPU {i}: {used_gb:.1f}GB/{total_gb:.1f}GB ({used_gb/total_gb:.1%})")
                
                # Get process information
                processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                if processes:
                    print(f"Processes using GPU {i}:")
                    for process in processes:
                        pid = process.pid
                        used_mem = process.usedGpuMemory / 1024**3
                        try:
                            process_name = pynvml.nvmlSystemGetProcessName(pid)
                            print(f"  - {process_name} (PID {pid}): {used_mem:.1f}GB")
                        except pynvml.NVMLError:
                            print(f"  - Unknown process (PID {pid}): {used_mem:.1f}GB")
        time.sleep(1)

except KeyboardInterrupt:
    print("\nMonitoring stopped by user")
except pynvml.NVMLError as e:
    print(f"NVML Error: {e}")
except Exception as e:
    print(f"Unexpected error: {e}")
finally:
    pynvml.nvmlShutdown()
