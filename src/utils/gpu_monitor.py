"""GPU temperature monitoring and throttling utilities."""
import subprocess
import time
from typing import Optional


def get_gpu_temperature() -> Optional[float]:
    """
    Get current GPU temperature in Celsius using nvidia-smi.
    
    Returns:
        Temperature in Celsius, or None if unavailable.
    """
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            temp_str = result.stdout.strip()
            return float(temp_str)
    except (subprocess.TimeoutExpired, ValueError, FileNotFoundError):
        pass
    return None


def wait_for_cooldown(max_temp: float, check_interval: float = 2.0, cooldown_target: Optional[float] = None) -> None:
    """
    Pause execution until GPU temperature drops below threshold.
    
    Args:
        max_temp: Maximum allowed temperature in Celsius
        check_interval: Seconds between temperature checks
        cooldown_target: Target temperature to resume (default: max_temp - 5)
    """
    if cooldown_target is None:
        cooldown_target = max_temp - 5
    
    temp = get_gpu_temperature()
    if temp is None:
        return  # Can't monitor, continue
    
    if temp >= max_temp:
        print(f"   ⚠️  GPU temp: {temp}°C (límite: {max_temp}°C). Pausando entrenamiento...")
        while temp is not None and temp >= cooldown_target:
            time.sleep(check_interval)
            temp = get_gpu_temperature()
            if temp is not None:
                print(f"   🌡️  GPU temp: {temp}°C (esperando {cooldown_target}°C)", end='\r')
        if temp is not None:
            print(f"\n   ✅ GPU enfriada a {temp}°C. Reanudando entrenamiento...")


class GPUTempMonitor:
    """Context manager for GPU temperature monitoring during training."""
    
    def __init__(self, max_temp: float = 85.0, check_every: int = 10):
        """
        Args:
            max_temp: Maximum allowed temperature in Celsius
            check_every: Check temperature every N batches
        """
        self.max_temp = max_temp
        self.check_every = check_every
        self.batch_count = 0
        self.last_temp = None
        
    def check(self) -> None:
        """Check temperature and throttle if needed."""
        self.batch_count += 1
        if self.batch_count % self.check_every == 0:
            self.last_temp = get_gpu_temperature()
            if self.last_temp is not None:
                wait_for_cooldown(self.max_temp)
            
    def reset(self) -> None:
        """Reset batch counter (e.g., at epoch end)."""
        self.batch_count = 0
