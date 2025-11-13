"""Test GPU temperature monitoring."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.gpu_monitor import get_gpu_temperature, wait_for_cooldown


def main():
    print("=== Test de monitoreo de temperatura GPU ===\n")
    
    temp = get_gpu_temperature()
    
    if temp is None:
        print("❌ No se pudo detectar la temperatura de la GPU")
        print("   Verifica que nvidia-smi esté disponible")
        return
    
    print(f"✅ Temperatura actual: {temp}°C")
    print(f"\nSimulando límite de 82°C...")
    
    if temp >= 82:
        print(f"⚠️  Temperatura excede 82°C. Pausando...")
        wait_for_cooldown(max_temp=82.0, check_interval=2.0)
    else:
        print(f"✅ Temperatura OK ({temp}°C < 82°C)")
    
    print("\n✅ Test completado")


if __name__ == "__main__":
    main()
