# Conteo de Esquejes con Vision Artificial

El proyecto incluye datasets, videos de validacion, modelos entrenados, exportaciones a OpenVINO y scripts de inferencia para diferentes cultivos.

## Contenido del repositorio

```text
Crisantemo/
  Validacion modelo CR/        Resultados, videos y documentos de validacion.
  dataset entrenamiento.../    Imagenes y videos usados para entrenamiento/validacion.
  modelos/                     Modelos YOLO, exportaciones OpenVINO y scripts de conteo.
  tools/                       Herramientas auxiliares, incluida exportacion a OpenVINO.

Snapdragon/
  dataset/                     Imagenes y videos de prueba/validacion.
  modelos/                     Modelo, exportacion OpenVINO y script de conteo.

Solidago/
  dataset/                     Carpeta preparada para futuros datos.
```

## Versionado de archivos grandes

Este repositorio usa Git LFS para videos, modelos y otros binarios grandes:

- `*.mp4`
- `*.mkv`
- `*.pt`
- `*.bin`
- `*.pptx`

Antes de clonar o descargar el repositorio completo, instala Git LFS:

```powershell
git lfs install
```

Despues de clonar:

```powershell
git clone https://github.com/jumarinj-lab/Conteo-de-Esquejes.git
cd Conteo-de-Esquejes
git lfs pull
```

## Requisitos

Los scripts de inferencia estan hechos en Python y usan principalmente:

- Python 3.11
- OpenCV
- Ultralytics YOLO
- OpenVINO
- Supervision
- CVZone
- NumPy

Instalacion sugerida:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install ultralytics openvino opencv-python supervision cvzone numpy
```

## Uso basico

Los scripts tienen rutas y parametros configurados al inicio de cada archivo. Si se cambia el video, modelo, camara o carpeta OpenVINO, editar esas constantes antes de ejecutar.

### Crisantemo - conteo sobre video

```powershell
cd Crisantemo\modelos
python model_ConteoDeEsquejesVideo.py
```

### Crisantemo - conteo en tiempo real

```powershell
cd Crisantemo\modelos
python model_ConteoDeEsquejesRealTime.py
```

Si la camara no abre, revisar en el script los valores:

- `CAM_INDEX`
- `USE_DSHOW`
- `CAM_W`, `CAM_H`, `CAM_FPS`

### Crisantemo - deteccion/seguimiento de cama

```powershell
cd Crisantemo\modelos
python model_BordesCama.py
```

```powershell
cd Crisantemo\modelos
python model_BordesCamaRealTime.py
```

### Snapdragon - prueba de conteo

```powershell
cd Snapdragon\modelos
python PruebaSNAP.py
```

## Exportar modelos a OpenVINO

El proyecto incluye una herramienta para exportar modelos `.pt` a OpenVINO:

```powershell
cd Crisantemo
python tools\export_openvino.py best26V2.pt
```

Mas detalles en:

```text
Crisantemo/tools/README_vscode_openvino.md
```

## Notas de trabajo

- Los modelos OpenVINO existentes se cargan desde carpetas como `best26V2_openvino_model` o `PruebaSNAP_openvino_model`.
- Si la carpeta OpenVINO no existe, algunos scripts intentan exportarla automaticamente desde el `.pt`.
- Los videos y modelos grandes dependen de Git LFS; si aparecen archivos pequenos de texto en lugar del contenido real, ejecutar `git lfs pull`.
- Los scripts muestran una ventana `VIEW`; presionar `ESC` para salir.

