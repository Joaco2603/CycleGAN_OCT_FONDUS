# Control de Temperatura GPU

## Resumen

Sistema automático para limitar la temperatura de la GPU durante el entrenamiento. Pausa el entrenamiento cuando la temperatura excede el límite configurado y lo reanuda cuando se enfría.

## Configuración

Edita `config.yaml`:

```yaml
training:
  max_gpu_temp: 65.0  # Temperatura máxima en °C (pausa si excede)
  temp_check_interval: 2  # Revisar temperatura cada N batches (más bajo = mejor control)
```

**⚠️ IMPORTANTE**: La temperatura puede subir **ligeramente por encima** del límite entre chequeos.

Ejemplo: Si revisas cada 10 batches y la GPU sube 2°C por batch:
- Batch 10: 63°C ✅ (continúa)
- Batch 11-19: GPU sigue calentándose...
- Batch 20: 73°C ⚠️ (detecta y pausa)

**Solución**: Usa `temp_check_interval: 2` para control más preciso.

### Valores recomendados

| GPU | Temp. Segura | Temp. Límite | Check Interval |
|-----|--------------|--------------|----------------|
| RTX 3070 | 75-80°C | 82°C | 2-5 |
| RTX 3080 | 75-80°C | 82°C | 2-5 |
| RTX 4090 | 80-85°C | 87°C | 2-5 |

**Nota**: 
- Valores más bajos de `temp_check_interval` = control más preciso pero overhead mayor
- Para límites estrictos (<70°C), usa `temp_check_interval: 1` o `2`
- Para límites normales (>75°C), usa `temp_check_interval: 5` o `10`

## Funcionamiento

1. **Monitoreo**: Cada `temp_check_interval` batches, el sistema verifica la temperatura usando `nvidia-smi`
2. **Pausa**: Si la temperatura ≥ `max_gpu_temp`, pausa el entrenamiento
3. **Enfriamiento**: Espera hasta que la temperatura baje 5°C por debajo del límite
4. **Reanudación**: Continúa el entrenamiento automáticamente

### Ejemplo de salida

```
   GPU: NVIDIA GeForce RTX 3070
   Memoria: 8.0 GB
   🌡️  Temperatura inicial: 41°C (límite: 82°C)
   
Epoch 1/200 — G: 2.3451, D: 0.8234
   ⚠️  GPU temp: 83°C (límite: 82°C). Pausando entrenamiento...
   🌡️  GPU temp: 81°C (esperando 77°C)
   🌡️  GPU temp: 78°C (esperando 77°C)
   🌡️  GPU temp: 76°C (esperando 77°C)
   ✅ GPU enfriada a 76°C. Reanudando entrenamiento...
```

## Verificación manual

Comprobar temperatura actual:

```powershell
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits
```

## Estrategias adicionales para reducir temperatura

Si la GPU sigue sobrecalentándose:

1. **Reducir batch size** en `config.yaml`:
   ```yaml
   optim:
     batch_size: 2  # Reduce de 4 a 2
   ```

2. **Mejorar ventilación física**:
   - Limpiar polvo de ventiladores
   - Asegurar flujo de aire en el case
   - Considerar ventiladores adicionales

3. **Límite de potencia** (opcional):
   ```powershell
   nvidia-smi -pl 200  # Limitar a 200W (RTX 3070 default: 220W)
   ```

4. **Reducir resolución temporalmente**:
   ```yaml
   data:
     image_size: 128  # Reduce de 256 a 128 para pruebas
   ```

## Desactivar control de temperatura

Para entrenar sin límites (no recomendado):

```yaml
training:
  max_gpu_temp: 95.0  # Temperatura muy alta = efectivamente desactivado
  temp_check_interval: 1000  # Revisar muy raramente
```

## Troubleshooting

### ⚠️ GPU sube por encima del límite configurado

**Síntoma**: Configuraste `max_gpu_temp: 65.0` pero la GPU llega a 68-70°C

**Causa**: El sistema solo revisa cada N batches. Entre chequeos, la GPU continúa calentándose.

**Soluciones**:
1. **Aumentar frecuencia de chequeo** (RECOMENDADO):
   ```yaml
   temp_check_interval: 2  # Revisar cada 2 batches
   ```
   O incluso más agresivo:
   ```yaml
   temp_check_interval: 1  # Revisar CADA batch (máximo control)
   ```

2. **Reducir límite preventivamente**:
   ```yaml
   max_gpu_temp: 60.0  # 5°C por debajo del objetivo real
   ```

3. **Reducir batch size** para menos calor por batch:
   ```yaml
   optim:
     batch_size: 2  # Reduce de 3 a 2
   ```

4. **Combinar estrategias**:
   ```yaml
   training:
     max_gpu_temp: 62.0  # Límite más bajo
   temp_check_interval: 1  # Chequeo continuo
   optim:
     batch_size: 2  # Menos carga
   ```

### nvidia-smi no encontrado

**Síntoma**: El sistema no detecta temperatura
**Solución**: 
1. Verifica que los drivers NVIDIA estén instalados
2. Añade `C:\Program Files\NVIDIA Corporation\NVSMI` al PATH

### Pausas muy frecuentes

**Síntoma**: El entrenamiento pausa cada pocos batches
**Solución**:
1. Aumenta `max_gpu_temp` en 2-3°C
2. Reduce `batch_size` o `image_size`
3. Mejora ventilación del sistema

### No pausa aunque la GPU esté caliente

**Síntoma**: GPU >85°C pero sigue entrenando
**Solución**:
1. Verifica que `max_gpu_temp` esté configurado correctamente
2. Reduce `temp_check_interval` para revisar más frecuentemente
3. Comprueba que nvidia-smi funcione: `nvidia-smi`

---

**Archivos relacionados**:
- `src/utils/gpu_monitor.py` - Implementación del monitor
- `src/training/train.py` - Integración en el loop de entrenamiento
- `config.yaml` - Configuración de límites

**Última actualización**: 2025-11-13
