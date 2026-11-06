# Mejora del entrenamiento: sin tope de épocas y búsqueda de hiperparámetros con Optuna

Este repositorio es la continuación de [clasificacion_residuos_ia](https://github.com/gnavidad1978/clasificacion_residuos_ia), cuya versión entregada en la materia es la [v1.0-entrega](https://github.com/gnavidad1978/clasificacion_residuos_ia/releases/tag/v1.0-entrega). Aplica las recomendaciones de la Dra. Lina María Aguilar Lobo (4 de noviembre de 2026) a los tres modelos prioritarios, MobileNetV3-Large, ResNet-50 y YOLO11n-cls, y compara tres formas de entrenarlos:

| Experimento | Configuración | Qué cambia |
|---|---|---|
| original | `configs/*.yaml` (resultados de la versión entregada) | 15 épocas como máximo, misma receta para todos |
| sin_tope | `configs/sin_tope/` | Hasta 60 épocas; paro temprano por pérdida de validación (paciencia 6); tasa de aprendizaje que baja por meseta |
| ajustado | `configs/ajustado/` (los genera `tune.py`) | Sin tope y con los hiperparámetros que encontró Optuna |

Archivos nuevos o modificados respecto al repositorio original:

| Archivo | Cambio |
|---|---|
| `model.py` | Optimizadores Adam y RAdam además de AdamW y SGD; programador "meseta" (ReduceLROnPlateau) además del coseno |
| `utils.py` | El nombre del resultado incluye el experimento; guarda la brecha entre pérdida de validación y de entrenamiento en la mejor época |
| `configs/sin_tope/` | Configuraciones sin tope de épocas |
| `tune.py` | Búsqueda con Optuna (optimizador, tasa de aprendizaje y decaimiento de pesos; objetivo F1 macro de validación), reanudable, con poda por mediana |
| `report_tuning.py` | Tabla de los tres experimentos, curvas comparadas y resumen de Optuna |
| `notebooks/Proyecto_tuning.ipynb` | Cuaderno de Colab de estos experimentos: https://colab.research.google.com/github/gnavidad1978/clasificacion_residuos_ia_tuning/blob/main/notebooks/Proyecto_tuning.ipynb |

Uso desde la línea de comandos:

```
python main.py fit --config configs/sin_tope/mobilenet_v3.yaml --seed_everything 42
python tune.py --config configs/sin_tope/mobilenet_v3.yaml --pruebas 15 --epocas 10
python main.py fit --config configs/ajustado/mobilenet_v3.yaml --seed_everything 42
python report_tuning.py --original <carpeta con los resultados de la versión entregada>
```

El resto de este documento corresponde al repositorio original.

---

# Clasificación de residuos por material: redes convolucionales contra Vision Transformer

Proyecto final de la materia Inteligencia Artificial (DCI019), Doctorado en Ciencias e Ingeniería, Universidad Autónoma de Guadalajara. Autora: Gabriela Medina Navidad.

El proyecto compara tres redes convolucionales (MobileNetV3, ResNet-50 y YOLO11n-cls) contra dos Vision Transformers de distinto tamaño (DeiT-Small, comparable a ResNet-50, y ViT-B/16) en la clasificación de residuos. Se usan cinco categorías de la NAE-SEMADES-007/2008 de Jalisco (vidrio, plástico, metal, papel y cartón, y orgánico) más una clase residual, "otros", que proviene de Miscellaneous Trash de RealWaste (ver Datos). Se reportan desempeño y costo (parámetros, tamaño, tiempo de entrenamiento y tiempo por foto), porque el modelo elegido debe funcionar en un sistema embebido. Es un avance de la tesis doctoral sobre visión artificial embebida para la validación de residuos antes de su depósito.

La organización sigue la del repositorio de la clase [clase_arquitectura_cugdl](https://github.com/UlisesMoyaSanchez/clase_arquitectura_cugdl): PyTorch Lightning con LightningCLI y archivos YAML, de modo que los parámetros se cambian sin tocar el código.

## Estructura

```
clasificacion_residuos_ia/
├── main.py              punto de entrada (LightningCLI)
├── model.py             fábrica de redes y LightningModule
├── dataset.py           LightningDataModule que lee la partición fija
├── prepare_data.py      descarga RealWaste, acomodo a seis categorías, partición, exploración y línea base
├── categorias.py        categorías y constantes compartidas
├── utils.py             callback que guarda desempeño y costo de cada corrida
├── benchmark.py         velocidad de cada red con datos sintéticos, en GPU o en CPU (costo de cómputo)
├── evaluate_test.py     evaluación final con el conjunto de prueba (una sola vez)
├── report.py            tabla comparativa, F1 por categoría, curvas, matrices de confusión y errores
├── configs/             un YAML por modelo
│   ├── mobilenet_v3.yaml
│   ├── resnet50.yaml
│   ├── yolo11n_cls.yaml
│   ├── deit_small.yaml
│   └── vit_b_16.yaml
├── notebooks/
│   └── Proyecto_final.ipynb   cuaderno de Colab que ejecuta y presenta el proyecto
└── requirements.txt
```

Los datos, los registros de TensorBoard, los modelos y los resultados no se suben al repositorio (ver `.gitignore`); en Colab se guardan en Google Drive.

## Cómo ejecutar el cuaderno en Colab

1. Abrir el cuaderno directamente desde GitHub con este enlace: https://colab.research.google.com/github/gnavidad1978/clasificacion_residuos_ia/blob/main/notebooks/Proyecto_final.ipynb
2. Activar la GPU: Entorno de ejecución, Cambiar tipo de entorno de ejecución, GPU T4, Guardar.
3. Ejecutar todo: Entorno de ejecución, Ejecutar todo (Ctrl+F9).
4. Aceptar los dos avisos: que el cuaderno viene de GitHub (Ejecutar de todos modos) y el permiso para conectar Google Drive.
5. El archivo de RealWaste debe estar en `MyDrive/proyecto_IA/realwaste.zip`; si no está, `prepare_data.py` lo descarga (unos 657 MB).

Tiempos aproximados con GPU T4: alrededor de 8 minutos por entrenamiento, es decir cerca de dos horas para los 5 modelos con 3 semillas. La medición de costo (`benchmark.py`) tarda un par de minutos en GPU y unos 45 minutos en CPU; si ya existe `costo_computo.csv`, el cuaderno no la repite. Para una prueba rápida se puede dejar `SEMILLAS = [42]` en la primera celda.

Si Colab se desconecta, no se pierde lo terminado: cada entrenamiento guarda su resultado en `MyDrive/proyecto_IA/repo_resultados` y, al volver a ejecutar todo, la celda de entrenamiento salta lo que ya existe. Conviene evitar que la computadora se suspenda mientras entrena (cargador conectado y suspensión en Nunca).

Mientras entrena, si TensorBoard muestra *Data could not be loaded*, no afecta el entrenamiento; basta con volver a ejecutar esa celda al terminar.

El conjunto de prueba se evalúa una sola vez, al final, quitando el comentario de la última celda del cuaderno. `evaluate_test.py` evalúa las tres semillas de cada modelo y, si ya existen resultados de prueba, se niega a repetirlos.

En resumen, el cuaderno ejecuta:

```bash
git clone https://github.com/gnavidad1978/clasificacion_residuos_ia.git
cd clasificacion_residuos_ia
pip install -r requirements.txt
python prepare_data.py --zip /content/drive/MyDrive/proyecto_IA/realwaste.zip
python main.py fit --config configs/mobilenet_v3.yaml --seed_everything 42
python benchmark.py
python report.py
python evaluate_test.py   # solo al final, una vez
```

## Cambiar parámetros sin tocar el código

Editando el YAML, o agregando el parámetro al comando:

```bash
# Otra semilla
python main.py fit --config configs/vit_b_16.yaml --seed_everything 7

# Más épocas y otra tasa de aprendizaje
python main.py fit --config configs/resnet50.yaml --trainer.max_epochs 20 --model.lr 0.0001

# SGD con momentum en lugar de AdamW
python main.py fit --config configs/mobilenet_v3.yaml --model.optimizer sgd --model.lr 0.01

# Prueba rápida en CPU
python main.py fit --config configs/mobilenet_v3.yaml --trainer.precision 32-true --trainer.max_epochs 1

# Ver todos los parámetros disponibles
python main.py fit --help
```

Si se cambia `--model.backbone` desde la línea de comandos, conviene cambiar también `--trainer.logger.init_args.name` para que TensorBoard la guarde con su nombre.

## Protocolo experimental

1. **Semilla fija.** `seed_everything` en cada YAML; `prepare_data.py` usa `np.random.default_rng(semilla)`. La partición es la misma en todas las corridas; las semillas de entrenamiento (42, 7 y 2026) cambian el orden de las fotos, los aumentos y la inicialización de la última capa. El entrenamiento en GPU no se configuró en modo determinista, así que repetir una corrida puede dar valores ligeramente distintos; por eso se reportan media y desviación de tres semillas.
2. **Partición antes de mirar los datos.** 70 % entrenamiento, 15 % validación y 15 % prueba, estratificada por categoría. Se reparten grupos de fotos casi idénticas (huella dHash de 16 por 16, umbral de 10 bits), no fotos sueltas, para que una foto y su casi gemela no queden en conjuntos distintos. La exploración y las figuras usan solo entrenamiento.
3. **Línea base.** Clase mayoritaria de entrenamiento, evaluada en validación (`resultados/linea_base.json`).
4. **Métrica elegida de antemano.** F1 macro, porque las categorías no están balanceadas y todas pesan igual en el bote. Con ella se elige la mejor época (ModelCheckpoint) y se detiene el entrenamiento (EarlyStopping, paciencia de 4). Se reportan además exactitud y matriz de confusión.

La normalización usa los valores publicados de ImageNet, que no se calculan con nuestros datos y por tanto no filtran información de validación ni de prueba. Los aumentos (volteo horizontal, giro de hasta 15 grados, brillo, contraste y saturación) solo se aplican a entrenamiento; no se usa volteo vertical porque un residuo sostenido en la mano no aparece de cabeza. El conjunto de prueba se usa una sola vez, al final, con `python evaluate_test.py`.

Todas las redes se entrenan con el mismo bucle, las mismas fotos, los mismos aumentos y la misma regla de paro; la única diferencia de receta es la tasa de aprendizaje, 0.0003 en las CNN y 0.00003 en los transformers. YOLO11n-cls se entrena con este mismo bucle, no con el entrenador de Ultralytics. No se ajustaron los hiperparámetros de cada red por separado.

DeiT-Small y ViT-B/16 usan pesos preentrenados solo con ImageNet-1K (ViT-B/16: pesos IMAGENET1K_V1 de torchvision, entrenados desde cero con una receta modificada de DeiT; DeiT-Small: pesos fb_in1k de timm). Los Vision Transformers dependen más que las CNN de un preentrenamiento grande, así que las conclusiones se limitan a transformers preentrenados en ImageNet-1K.

El tiempo por foto se mide en el procesador de Colab con 4 hilos y una foto a la vez. Es solo una referencia relativa entre modelos: ese procesador es x86 de 2 núcleos y una Raspberry Pi 5 es ARM de 4 núcleos, así que los milisegundos no se trasladan al dispositivo, que debe medirse directamente. El tiempo total de entrenamiento no sirve para comparar arquitecturas en Colab gratuito, porque leer y transformar las fotos con 2 procesadores tarda lo mismo para cualquier red y cada modelo se detiene en una época distinta; por eso `benchmark.py` mide las fotos por segundo de cada red con datos sintéticos ya cargados en memoria, en GPU si está disponible y si no en CPU (los resultados entregados se midieron en CPU porque se agotó la cuota de GPU), y el reporte muestra segundos por época.

## Datos

RealWaste: S. Single, S. Iranmanesh y R. Raad, "RealWaste: A Novel Real-Life Data Set for Landfill Waste Classification Using Deep Learning," *Information*, vol. 14, no. 12, p. 633, 2023, doi: 10.3390/info14120633. Conjunto en UCI Machine Learning Repository, doi: 10.24432/C5SS4G, licencia CC BY 4.0.

Acomodo de las clases: Cardboard y Paper a papel y cartón; Glass a vidrio; Metal a metal; Plastic a plástico; Food Organics y Vegetation a orgánico; Textile Trash se excluye porque el campus no tiene bote de textiles. Miscellaneous Trash se asignó inicialmente a residuos inorgánicos de difícil reciclaje, pero una revisión visual de 50 fotos al azar del conjunto de entrenamiento (semilla 2026) mostró que no corresponde a esa categoría: solo una foto era claramente de difícil reciclaje según la norma y a lo sumo siete lo serían, mientras que el resto eran plásticos, metales, papel, electrónicos, textiles, residuos sanitarios y otros materiales. Por eso se conserva como clase residual y se reporta como "otros" (internamente conserva el identificador `dificil_reciclaje` para no alterar la partición). Se puede excluir con `prepare_data.py --excluir-misc`.

Limitaciones declaradas: RealWaste no contiene residuos sostenidos en la mano, se tomó en Australia y no publica identificadores de objeto, por lo que fotos del mismo objeto desde ángulos distintos pueden quedar en conjuntos diferentes. Algunas etiquetas originales no coinciden con la NAE-SEMADES-007/2008: por ejemplo, el papel aluminio, las envolturas de golosinas y los envases multicapa, que la norma coloca en difícil reciclaje, aparecen en RealWaste como metal, plástico o cartón.
