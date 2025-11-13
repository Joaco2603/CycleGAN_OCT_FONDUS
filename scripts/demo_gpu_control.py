"""Demostración del sistema de control de temperatura GPU."""
import subprocess
import time


def get_gpu_temp():
    """Obtener temperatura actual de GPU."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=temperature.gpu', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=2
        )
        if result.returncode == 0:
            return float(result.stdout.strip())
    except:
        return None


def main():
    print("\n" + "="*60)
    print("  DEMOSTRACIÓN: Control de Temperatura GPU")
    print("="*60 + "\n")
    
    temp = get_gpu_temp()
    
    if temp is None:
        print("❌ No se pudo detectar la GPU")
        print("   Asegúrate de tener nvidia-smi instalado\n")
        return
    
    print(f"🌡️  Temperatura actual: {temp}°C\n")
    print("📋 Configuración en config.yaml:\n")
    print("   training:")
    print("     max_gpu_temp: 82.0      # ← Límite de temperatura")
    print("     temp_check_interval: 10  # ← Revisar cada 10 batches\n")
    
    print("✅ Comportamiento durante entrenamiento:\n")
    print(f"   • Si temp < 82°C: Entrenamiento normal")
    print(f"   • Si temp ≥ 82°C: ⚠️  PAUSA automática")
    print(f"   • Espera hasta: < 77°C (82 - 5)")
    print(f"   • Luego: ✅ REANUDA entrenamiento\n")
    
    if temp < 70:
        status = "🟢 Excelente - GPU fría"
    elif temp < 80:
        status = "🟡 Normal - En rango seguro"
    elif temp < 85:
        status = "🟠 Cálida - Cercana al límite"
    else:
        status = "🔴 Caliente - Por encima del límite"
    
    print(f"Estado actual: {status} ({temp}°C)\n")
    
    print("💡 Comandos útiles:\n")
    print("   Ver temperatura en tiempo real:")
    print("   → nvidia-smi -l 1\n")
    print("   Entrenar con control de temperatura:")
    print("   → python train.py --config config.yaml\n")
    print("   Ajustar límite (editar config.yaml):")
    print("   → max_gpu_temp: 75.0  # Más conservador")
    print("   → max_gpu_temp: 85.0  # Más permisivo\n")
    
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
