# Flujo VS Code para OpenVINO

Este proyecto ya queda listo para exportar nuevas versiones del modelo a OpenVINO desde VS Code.

## Archivos clave

- `tools/export_openvino.py`: exporta un `.pt` a OpenVINO.
- `.vscode/tasks.json`: tareas listas para correr desde VS Code.
- `.vscode/settings.json`: fija Python 3.11 para este proyecto.

## Como usarlo

1. Abre la carpeta `proyecto conteo topspin - caribe` en VS Code.
2. Presiona `Ctrl+Shift+P`.
3. Ejecuta `Tasks: Run Task`.
4. Usa una de estas tareas:
   - `Export OpenVINO Latest`
   - `Export OpenVINO Prompt`

## Que hace el script

1. Toma un modelo `.pt` de la carpeta `modelos`.
2. Exporta una carpeta OpenVINO con el nombre base del modelo.
   Ejemplo: `best26V2.pt` -> `best26V2_openvino_model`
3. Copia ese export a la carpeta estandar `modelos/best_openvino_model`.
4. Si ya existia `best_openvino_model`, la mueve antes a `best_openvino_model_backup`.

## Ejemplos manuales

```powershell
C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe tools\export_openvino.py best26V2.pt
```

```powershell
C:\Users\HP\AppData\Local\Programs\Python\Python311\python.exe tools\export_openvino.py best30.pt
```

## Notas

- El script usa `Python 3.11`.
- El paquete `openvino` ya quedo instalado en modo usuario para este Python.
- Si cambias de PC o de usuario, probablemente tengas que reinstalar `openvino` y `ultralytics`.
