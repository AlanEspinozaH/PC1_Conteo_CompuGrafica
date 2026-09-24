# Conteo de personas — Computación Gráfica

Prototipo de visión clásica para detectar regiones móviles, conservar un ID
entre fotogramas y contar cruces de una línea. El sistema no realiza
reconocimiento biométrico: el resultado representa **eventos de cruce de
regiones móviles** y debe validarse contra un conteo manual.

## Requisitos

- Python 3.10 o superior.
- Windows 10/11 o Ubuntu 22.04/24.04.
- El video oficial `conteo_pc1_grafica.mp4`, descargado en `video/`.

### Ubuntu

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si PowerShell bloquea la activación, puede ejecutarse Python directamente:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe main.py --video video/conteo_pc1_grafica.mp4
```

## Secuencia recomendada de ejecución

### 1. Comprobar primero 30 segundos de estabilización

```bash
python evaluate_stabilization.py \
  --video video/conteo_pc1_grafica.mp4 \
  --max-frames 900
```

El número 900 equivale aproximadamente a 30 segundos cuando el video tiene 30
FPS. Si la prueba termina correctamente, evaluar los cinco minutos completos:

```bash
python evaluate_stabilization.py --video video/conteo_pc1_grafica.mp4
```

Este comando no crea el video final. Mide el movimiento geométrico residual con
y sin estabilización. En videos verticales de alta resolución puede tardar
varios minutos; muestra progreso cada 300 frames.

### 2. Seleccionar y guardar la ROI

```bash
python main.py \
  --video video/conteo_pc1_grafica.mp4 \
  --roi-file config/roi_oficial.json \
  --select-roi-only
```

Se abre una vista reducida cuando el video vertical no cabe en la pantalla.
Marcar solamente la zona de paso, presionar `Enter` o `Space` y verificar las
coordenadas impresas. El programa las convierte a la resolución original y las
guarda en `config/roi_oficial.json`.

No hace falta copiar manualmente `x`, `y`, `ancho` y `alto`. Como alternativa,
todavía puede usarse `--roi X Y W H`.

### 3. Procesar con estabilización

```bash
python main.py \
  --video video/conteo_pc1_grafica.mp4 \
  --roi-file config/roi_oficial.json \
  --no-display \
  --output-dir output/con_estabilizacion
```

Esperar hasta que reaparezca el prompt de la terminal. El programa informa el
avance cada 300 frames y al finalizar crea el CSV y el video anotado.

### 4. Repetir sin estabilización

Debe utilizarse exactamente el mismo video y el mismo archivo de ROI:

```bash
python main.py \
  --video video/conteo_pc1_grafica.mp4 \
  --roi-file config/roi_oficial.json \
  --no-stabilization \
  --no-display \
  --output-dir output/sin_estabilizacion
```

En PowerShell puede escribirse cada comando en una sola línea. Las rutas con
`/` también funcionan en Windows porque Python utiliza `pathlib`.

### Calibración rápida del detector

No edites `MIN_AREA` para cada prueba. Usa `--min-area` y limita la ejecución
con `--max-frames`. Por ejemplo, para revisar los primeros 2000 frames:

```bash
python main.py \
  --video video/conteo_pc1_grafica.mp4 \
  --roi-file config/roi_oficial.json \
  --min-area 500 \
  --max-frames 2000
```

En PowerShell, escribe el mismo comando en una línea o sustituye `\` por el
acento grave (`` ` ``). Prueba al menos 400, 500, 600 y 800 con la misma ROI.
Un valor demasiado bajo fragmenta una persona en varios IDs; uno demasiado
alto deja de detectar personas alejadas. La elección final debe basarse en una
comparación contra conteo manual, no solo en que la máscara se vea bien.

### Orientación vertical y línea de conteo

La orientación vertical del video no determina la orientación de la línea. El
programa actual dibuja una **línea horizontal** en la mitad de la ROI:

- Es correcta si las personas se desplazan principalmente de arriba hacia
  abajo o de abajo hacia arriba en la imagen.
- Es incorrecta si se desplazan principalmente de izquierda a derecha. En ese
  caso debe implementarse una línea vertical antes de validar el conteo.

## Archivos principales

| Archivo | Responsabilidad |
|---|---|
| `main.py` | Integra video, ROI, detector, tracker, contador y salidas |
| `stabilizer.py` | Módulo importado por `main.py`; no se ejecuta directamente |
| `evaluate_stabilization.py` | Compara movimiento residual con/sin estabilización |
| `preprocessing.py` | Genera la máscara binaria de movimiento |
| `detector.py` | Convierte contornos en cajas y centroides |
| `tracker.py` | Mantiene identificadores entre frames |
| `counter.py` | Registra cruces de la línea |
| `tests/` | Pruebas automáticas |

`stabilizer.py` debe permanecer en la raíz del proyecto y agregarse al mismo
commit que `main.py`. El comando `git apply` ya lo modifica automáticamente.

Parámetros opcionales del estabilizador:

```text
--stabilization-window 30   mayor = trayectoria más suave
--border-scale 1.02        zoom constante para ocultar bordes
```

Los resultados se guardan en:

- `eventos.csv`: frame, tiempo, ID y dirección.
- `video_resultado.mp4`: ROI, detecciones, IDs y conteos.

## Estabilización implementada

El archivo `stabilizer.py` aplica el siguiente flujo:

1. Detecta esquinas del fondo con Shi–Tomasi.
2. Sigue cada punto con Lucas–Kanade.
3. Comprueba el seguimiento hacia adelante y hacia atrás.
4. Estima traslación y rotación global con RANSAC.
5. Acumula la trayectoria observada de la cámara.
6. Suaviza la trayectoria mediante una media exponencial causal.
7. Lleva cada frame desde la trayectoria observada hacia la suavizada.

La ROI se excluye de la selección de características cuando queda suficiente
fondo disponible. Así, las personas que pasan por la zona de conteo tienen
menos influencia en la estimación del movimiento de la cámara.

Este algoritmo corrige vibración leve. No convierte un paneo amplio en una
cámara fija ni recupera regiones que quedaron fuera del encuadre.

## Pruebas

```bash
python -m pytest -q
```

Las pruebas generan imágenes sintéticas y verifican que:

- El primer frame no se modifica.
- Una traslación pequeña se reduce de forma medible.
- Los frames sin características usan una transformación identidad.
- La ROI de conteo se excluye del cálculo del movimiento global.
- `reset()` elimina el estado de un video anterior.

## Validación experimental

Procesar exactamente el mismo segmento y la misma ROI con estabilización ON y
OFF. Comparar cada evento automático con una anotación manual, usando la misma
dirección y una tolerancia temporal acordada (por ejemplo, ±1 segundo). Reportar
TP, FP, FN, precisión, recall y F1; no basta con mostrar que el video se ve más
estable.

## Cómo contribuir

No trabajar directamente sobre `main`. Antes de comenzar una tarea nueva:

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/nombre-descriptivo
```

Si la rama ya existe, solamente cambiar a ella:

```bash
git switch feature/tracking-conteo
```

Antes del commit:

```bash
git status --short
git diff
python -m pytest -q
```

Agregar solo los archivos correspondientes al cambio. Para la estabilización:

```bash
git add \
  .gitignore \
  README.md \
  main.py \
  stabilizer.py \
  evaluate_stabilization.py \
  requirements.txt \
  tests/test_stabilizer.py \
  tests/test_main_roi.py \
  config/roi_oficial.json

git commit -m "feat: implementar estabilización de trayectoria de cámara"
git push -u origin feature/tracking-conteo
```

Después, abrir un **Pull Request** desde `feature/tracking-conteo` hacia `main`.
El archivo `config/roi_oficial.json` sí debe compartirse: permite que todo el
grupo use exactamente la misma región. No subir el video oficial, entornos
`.venv`, CSV ni videos generados. Esas rutas están excluidas por `.gitignore`.

Cada Pull Request debe indicar:

- Qué problema corrige.
- Qué archivos modifica.
- Cómo se ejecutó la prueba.
- Resultado de `pytest`.
- Limitaciones observadas.
