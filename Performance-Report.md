# Análisis de Rendimiento: Paralelización de Transformaciones de Imagen con OpenMP

> **Repositorio:** `image-transforms-omp`  
> **Wiki:** Reporte de rendimiento — Paralelismo a nivel de tareas  
> **Lenguaje:** C + OpenMP  
> **Fecha:** 2025

---

## Tabla de Contenidos

1. [Descripción del Proyecto](#1-descripción-del-proyecto)
2. [Especificaciones del Equipo](#2-especificaciones-del-equipo)
3. [Transformaciones Implementadas](#3-transformaciones-implementadas)
4. [Estrategia de Paralelización](#4-estrategia-de-paralelización)
5. [Imágenes de Prueba](#5-imágenes-de-prueba)
6. [Compilación y Ejecución](#6-compilación-y-ejecución)
7. [Resultados de Rendimiento](#7-resultados-de-rendimiento)
8. [Análisis y Conclusiones](#8-análisis-y-conclusiones)
9. [Capturas del Monitor de Sistema](#9-capturas-del-monitor-de-sistema)
10. [Referencias](#10-referencias)

---

## 1. Descripción del Proyecto

Este proyecto implementa seis transformaciones de imágenes BMP de gran formato utilizando **OpenMP** para paralelización a nivel de tareas (`#pragma omp task`). El objetivo es evaluar el impacto del número de *threads* (6, 12 y 18) en el tiempo de ejecución sobre tres imágenes de más de 2000×2000 píxeles.

Las transformaciones aplicadas son:

| # | Transformación | Canal |
|---|---|---|
| 1 | Inversión horizontal (flip H) | Escala de grises |
| 2 | Inversión vertical (flip V) | Escala de grises |
| 3 | Desenfoque Box Blur (kernel 5×5) | Escala de grises |
| 4 | Inversión horizontal (flip H) | Color (BGR) |
| 5 | Inversión vertical (flip V) | Color (BGR) |
| 6 | Desenfoque Box Blur (kernel 5×5) | Color (BGR) |

---

## 2. Especificaciones del Equipo

> ⚠️ **Instrucción:** Sustituye los valores marcados con `[COMPLETAR]` con los datos reales de tu equipo. Puedes obtenerlos con los comandos indicados.

### 2.1 Procesador

| Campo | Valor |
|---|---|
| **Modelo** | [COMPLETAR — ej. Intel Core i7-12700K] |
| **Arquitectura** | [COMPLETAR — ej. Alder Lake / x86_64] |
| **Número de cores físicos** | [COMPLETAR — ej. 12 (8P + 4E)] |
| **Threads físicos (lógicos)** | [COMPLETAR — ej. 20] |
| **Frecuencia base** | [COMPLETAR — ej. 3.6 GHz] |
| **Frecuencia turbo máxima** | [COMPLETAR — ej. 5.0 GHz] |
| **Caché L3** | [COMPLETAR — ej. 25 MB] |

**Comando para obtener información del CPU (Linux):**
```bash
lscpu
# o bien:
cat /proc/cpuinfo | grep -E "model name|cpu MHz|siblings|cpu cores" | sort -u
```

**Comando para Windows (PowerShell):**
```powershell
Get-WmiObject Win32_Processor | Select-Object Name, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed
```

### 2.2 Memoria RAM

| Campo | Valor |
|---|---|
| **Capacidad total** | [COMPLETAR — ej. 32 GB] |
| **Tipo** | [COMPLETAR — ej. DDR4-3200] |
| **Canales** | [COMPLETAR — ej. Dual Channel] |

```bash
# Linux
free -h
sudo dmidecode -t memory | grep -E "Size|Speed|Type"
```

### 2.3 Sistema Operativo

| Campo | Valor |
|---|---|
| **SO** | [COMPLETAR — ej. Ubuntu 22.04 LTS / Windows 11 22H2] |
| **Kernel (Linux)** | [COMPLETAR — ej. 6.2.0-39-generic] |
| **Compilador GCC** | [COMPLETAR — ej. GCC 12.3.0] |
| **Versión OpenMP** | [COMPLETAR — ej. OpenMP 4.5] |

```bash
# Linux
uname -a
gcc --version
echo |cpp -fopenmp -dM |grep -i openmp
```

---

## 3. Transformaciones Implementadas

### 3.1 Conversión a Escala de Grises

Antes de aplicar las transformaciones grises, cada píxel BGR se convierte usando la fórmula de luminancia BT.601:

```
Gray = 0.07·B + 0.72·G + 0.21·R
```

Esta operación es paralelizada con `#pragma omp parallel for` dado que cada píxel es independiente.

### 3.2 Inversión Horizontal

Dado un píxel en la columna `col` de la fila `row`, su nueva posición es `(row, W-1-col)`.

```c
#pragma omp parallel for schedule(static)
for (long row = 0; row < H; row++) {
    for (long col = 0; col < W; col++) {
        flipped[row * W + col] = gray[row * W + (W - 1 - col)];
    }
}
```

### 3.3 Inversión Vertical

El píxel en la fila `row` se copia a la fila `H-1-row`.

```c
#pragma omp parallel for schedule(static)
for (long row = 0; row < H; row++) {
    memcpy(&flipped[row * W], &gray[(H - 1 - row) * W], W);
}
```

### 3.4 Desenfoque Box Blur (kernel 5×5)

Para cada píxel se promedia el valor de todos los vecinos dentro de la ventana `[-half, half]²`, manejando bordes con recorte (*clamp*).

```c
int half = BLUR_KERNEL / 2;  // 2 para kernel 5x5

#pragma omp parallel for schedule(static)
for (long row = 0; row < H; row++) {
    for (long col = 0; col < W; col++) {
        long sum = 0, count = 0;
        for (int dy = -half; dy <= half; dy++) {
            for (int dx = -half; dx <= half; dx++) {
                long r = row + dy, c = col + dx;
                if (r >= 0 && r < H && c >= 0 && c < W) {
                    sum += gray[r * W + c];
                    count++;
                }
            }
        }
        blurred[row * W + col] = (unsigned char)(sum / count);
    }
}
```

---

## 4. Estrategia de Paralelización

### 4.1 Paralelismo a Nivel de Tareas (`omp task`)

Las 6 transformaciones por imagen son **independientes entre sí**: ninguna depende del resultado de otra. Esto las hace ideales para distribuirlas como **tareas OpenMP**.

```
omp parallel → omp single
    │
    ├── omp task → flip_h_gray    (internamente: omp parallel for)
    ├── omp task → flip_v_gray    (internamente: omp parallel for)
    ├── omp task → blur_gray      (internamente: omp parallel for)
    ├── omp task → flip_h_color   (internamente: omp parallel for)
    ├── omp task → flip_v_color   (internamente: omp parallel for)
    └── omp task → blur_color     (internamente: omp parallel for)
              │
         omp taskwait
```

### 4.2 Paralelismo Anidado

Cada tarea contiene internamente bucles `#pragma omp parallel for`, aprovechando todos los threads disponibles para el procesamiento de píxeles.

### 4.3 Configuración de Threads

Se evalúan tres configuraciones:

| Configuración | Threads | Uso esperado |
|---|---|---|
| A | **6** | 1 thread por tarea (sin solapamiento interno) |
| B | **12** | 2 threads disponibles por tarea |
| C | **18** | 3 threads disponibles por tarea |

---

## 5. Imágenes de Prueba

Se utilizaron tres imágenes BMP de gran formato:

| ID | Archivo | Resolución | Tamaño aprox. |
|---|---|---|---|
| IMG-1 | `landscape_4k.bmp` | [COMPLETAR] | [COMPLETAR] |
| IMG-2 | `portrait_high.bmp` | [COMPLETAR] | [COMPLETAR] |
| IMG-3 | `texture_map.bmp` | [COMPLETAR] | [COMPLETAR] |

> 📌 Todas las imágenes deben tener **más de 2000 píxeles en ancho Y alto** y estar en formato BMP de **24 bits sin compresión**.

**Cómo convertir una imagen a BMP 24-bit sin compresión:**
```bash
# Con ImageMagick
convert input.jpg -type TrueColor -compress None output.bmp

# Con FFMPEG
ffmpeg -i input.jpg -vcodec bmp output.bmp
```

---

## 6. Compilación y Ejecución

### 6.1 Compilación

```bash
gcc -O2 -fopenmp image_transforms_omp.c -o image_transforms_omp
```

| Flag | Descripción |
|---|---|
| `-O2` | Optimización nivel 2 |
| `-fopenmp` | Habilita directivas OpenMP |

### 6.2 Ejecución

```bash
# Con 6 threads
./image_transforms_omp landscape_4k.bmp portrait_high.bmp texture_map.bmp 6

# Con 12 threads
./image_transforms_omp landscape_4k.bmp portrait_high.bmp texture_map.bmp 12

# Con 18 threads
./image_transforms_omp landscape_4k.bmp portrait_high.bmp texture_map.bmp 18
```

### 6.3 Salida Esperada

```
=============================================
 Transformaciones BMP con OpenMP (tasks)
 Kernel de blur: 5x5
 Threads solicitados: 12
=============================================

[IMG] landscape_4k.bmp  (3840 x 2160 px)
[OMP] Threads: 12
  [T3] flip_h_gray    -> 0.412 s
  [T1] flip_v_gray    -> 0.389 s
  [T7] flip_h_color   -> 0.401 s
  [T5] flip_v_color   -> 0.384 s
  [T9] blur_gray      -> 1.823 s
  [T11] blur_color    -> 2.105 s
[TOTAL] landscape_4k.bmp  con 12 threads -> 2.207 s
```

---

## 7. Resultados de Rendimiento

> 📝 **Instrucción:** Completa la tabla con los tiempos medidos en tu equipo. Ejecuta cada configuración al menos 3 veces y anota el promedio.

### 7.1 Tiempos por Imagen y Número de Threads (segundos)

#### IMG-1: `landscape_4k.bmp` — [Resolución: COMPLETAR]

| Transformación | 6 threads | 12 threads | 18 threads |
|---|---|---|---|
| flip_h_gray | | | |
| flip_v_gray | | | |
| blur_gray | | | |
| flip_h_color | | | |
| flip_v_color | | | |
| blur_color | | | |
| **TOTAL** | | | |

#### IMG-2: `portrait_high.bmp` — [Resolución: COMPLETAR]

| Transformación | 6 threads | 12 threads | 18 threads |
|---|---|---|---|
| flip_h_gray | | | |
| flip_v_gray | | | |
| blur_gray | | | |
| flip_h_color | | | |
| flip_v_color | | | |
| blur_color | | | |
| **TOTAL** | | | |

#### IMG-3: `texture_map.bmp` — [Resolución: COMPLETAR]

| Transformación | 6 threads | 12 threads | 18 threads |
|---|---|---|---|
| flip_h_gray | | | |
| flip_v_gray | | | |
| blur_gray | | | |
| flip_h_color | | | |
| flip_v_color | | | |
| blur_color | | | |
| **TOTAL** | | | |

---

### 7.2 Speedup y Eficiencia

El **speedup** se calcula respecto a la ejecución con 6 threads (caso base):

```
Speedup(N) = T(6) / T(N)
Eficiencia(N) = Speedup(N) / (N / 6)
```

| Imagen | Speedup 12T | Speedup 18T | Eficiencia 12T | Eficiencia 18T |
|---|---|---|---|---|
| IMG-1 | | | | |
| IMG-2 | | | | |
| IMG-3 | | | | |

---

### 7.3 Gráficas de Rendimiento

> 📌 Se recomienda incluir aquí las siguientes gráficas (generadas con Python/matplotlib, Excel o cualquier herramienta):
>
> 1. **Tiempo total vs. Número de threads** (una línea por imagen)
> 2. **Speedup vs. Número de threads** (con línea ideal de speedup lineal para referencia)
> 3. **Eficiencia vs. Número de threads**
> 4. **Tiempo por transformación** (barras agrupadas para 6/12/18 threads)

**Script Python para generar gráficas (plantilla):**

```python
import matplotlib.pyplot as plt
import numpy as np

threads = [6, 12, 18]

# Sustituir con tus valores reales
img1_total = [X.XXX, X.XXX, X.XXX]
img2_total = [X.XXX, X.XXX, X.XXX]
img3_total = [X.XXX, X.XXX, X.XXX]

plt.figure(figsize=(10, 6))
plt.plot(threads, img1_total, 'o-', label='IMG-1 (landscape_4k)')
plt.plot(threads, img2_total, 's-', label='IMG-2 (portrait_high)')
plt.plot(threads, img3_total, '^-', label='IMG-3 (texture_map)')
plt.xlabel('Número de Threads')
plt.ylabel('Tiempo Total (s)')
plt.title('Tiempo de Ejecución vs. Threads — OpenMP Tasks')
plt.legend()
plt.grid(True)
plt.xticks(threads)
plt.savefig('tiempo_vs_threads.png', dpi=150)
plt.show()
```

---

## 8. Análisis y Conclusiones

### 8.1 Observaciones Esperadas

**Transformaciones de flip (H y V):**
- Son operaciones de **copia de memoria** con acceso secuencial.
- Limitadas por el ancho de banda de memoria, no por CPU.
- El speedup puede **platearse** rápidamente con más threads por contención de memoria.

**Desenfoque Box Blur:**
- Es la transformación más costosa computacionalmente (O(W·H·k²)).
- Se beneficia **más** del paralelismo al ser intensiva en cómputo.
- Se espera mejor speedup al pasar de 6 → 12 threads que de 12 → 18.

**Ley de Amdahl:**
- Aunque todas las transformaciones son paralelizables, la lectura/escritura del archivo BMP es secuencial y representa una fracción fija del tiempo total.
- Esto limita el speedup máximo teórico.

### 8.2 Comportamiento del Scheduler de OpenMP Tasks

Con `#pragma omp task`, el runtime de OpenMP asigna las tareas a threads libres del pool. Con:

- **6 threads:** Cada tarea ocupa un thread; ejecución simultánea perfecta de las 6 tareas, pero sin paralelismo interno extra.
- **12 threads:** Las 6 tareas se distribuyen; cada tarea tiene ~2 threads para sus bucles internos.
- **18 threads:** ~3 threads por tarea; mayor paralelismo interno, pero más overhead de sincronización.

### 8.3 Conclusiones

> 📝 **Instrucción:** Completa esta sección con base en tus resultados experimentales reales.

```
[COMPLETAR con tus conclusiones basadas en los datos medidos]

Ejemplo de estructura:
- "Se observó un speedup de X al pasar de 6 a 12 threads, representando
  una mejora del Y% en el tiempo total."
- "El blur_color fue consistentemente la operación más costosa, representando
  el Z% del tiempo total con 6 threads."
- "Aumentar de 12 a 18 threads mostró rendimientos decrecientes, con una
  eficiencia del A% debido a..."
```

---

## 9. Capturas del Monitor de Sistema

> 📌 **Instrucción:** Captura pantallas del monitor de sistema **durante la ejecución** del programa con cada configuración de threads. Las capturas deben mostrar claramente el uso de todos los núcleos.

### 9.1 Herramientas Recomendadas

| SO | Herramienta | Cómo abrir |
|---|---|---|
| Linux | `htop` | `htop` en terminal |
| Linux | GNOME System Monitor | `gnome-system-monitor` |
| Linux | `glances` | `sudo apt install glances && glances` |
| Windows | Task Manager | Ctrl+Shift+Esc → Rendimiento → CPU |
| Windows | Process Explorer | Descargar de Sysinternals |
| macOS | Activity Monitor | Spotlight → "Activity Monitor" |

### 9.2 Capturas Requeridas

Para cada número de threads (6, 12, 18) incluir:

- [ ] **Captura con 6 threads** — Monitor mostrando uso de CPU por núcleo
- [ ] **Captura con 12 threads** — Monitor mostrando uso de CPU por núcleo  
- [ ] **Captura con 18 threads** — Monitor mostrando uso de CPU por núcleo

> Las capturas deben evidenciar que múltiples núcleos están siendo utilizados simultáneamente, validando la ejecución paralela real.

**Ejemplo de comando para capturar con htop en Linux:**
```bash
# En una terminal, lanzar htop
htop

# En otra terminal, ejecutar el programa
./image_transforms_omp img1.bmp img2.bmp img3.bmp 12

# Tomar captura de pantalla durante la ejecución
```

### 9.3 Imágenes de Capturas

> Insertar aquí las capturas con el siguiente formato Markdown:

```markdown
#### Ejecución con 6 Threads
![Monitor 6 threads](screenshots/monitor_6_threads.png)

#### Ejecución con 12 Threads
![Monitor 12 threads](screenshots/monitor_12_threads.png)

#### Ejecución con 18 Threads
![Monitor 18 threads](screenshots/monitor_18_threads.png)
```

---

## 10. Referencias

1. **OpenMP Architecture Review Board.** (2021). *OpenMP Application Programming Interface, Version 5.2*. https://www.openmp.org/spec-html/5.2/openmp.html
2. **Chapman, B., Jost, G., & van der Pas, R.** (2008). *Using OpenMP: Portable Shared Memory Parallel Programming*. MIT Press.
3. **Amdahl, G.** (1967). Validity of the single processor approach to achieving large scale computing capabilities. *AFIPS Spring Joint Computer Conference*.
4. **Microsoft.** BMP File Format Specification. https://learn.microsoft.com/en-us/windows/win32/gdi/bmp-reference
5. **GNU GCC.** OpenMP flags and directives. https://gcc.gnu.org/onlinedocs/gcc/OpenMP.html
6. Archivos base del proyecto: `ImageGrises_original.c`, `bmp_headers_v3.c`, `memory_report.c`, `malloc_ejem.c`

---

## Estructura del Repositorio

```
image-transforms-omp/
├── src/
│   └── image_transforms_omp.c    ← Código principal
├── images/
│   ├── landscape_4k.bmp
│   ├── portrait_high.bmp
│   └── texture_map.bmp
├── results/
│   ├── landscape_4k_flip_h_gray.bmp
│   ├── landscape_4k_flip_v_gray.bmp
│   ├── ... (18 imágenes de salida)
├── screenshots/
│   ├── monitor_6_threads.png
│   ├── monitor_12_threads.png
│   └── monitor_18_threads.png
├── plots/
│   └── tiempo_vs_threads.png
├── README.md
└── wiki/
    └── Performance-Report.md     ← Este archivo
```

---

*Reporte generado para la Wiki del repositorio GitHub — Proyecto OpenMP Image Transforms*
