"""Busqueda automatica de hiperparametros con Optuna (el "tuning" que recomendo la Dra. Lina).

En lugar de probar valores a mano, se le da a Optuna un rango para cada hiperparametro y un
objetivo. Optuna prueba combinaciones, aprende de las anteriores cuales zonas del rango
funcionan mejor y devuelve la mejor combinacion encontrada.

Que se busca:
    optimizador         adamw, adam, radam o sgd
    tasa de aprendizaje de 0.00001 a 0.01 para los Adam (rango sugerido por la Dra. Lina);
                        de 0.001 a 0.1 para SGD, que necesita tasas mas altas para avanzar
    decaimiento de pesos de 0.000001 a 0.1
    programador         coseno (baja la tasa hasta casi cero en --epocas epocas, como la receta
                        original) o meseta (la reduce a la mitad cuando la perdida de validacion
                        deja de bajar). Se agrego porque en el experimento sin tope la meseta dejo
                        la tasa alta casi todo el entrenamiento y MobileNetV3 empeoro.
Objetivo: el mayor F1 macro en VALIDACION. El grupo de prueba nunca se usa aqui.

Cada prueba entrena como maximo --epocas epocas (15, el horizonte de la receta original) y
Optuna corta antes las combinaciones que van claramente peor que la mediana de las anteriores
(poda), excepto en la ultima epoca, donde cortar ya no ahorra tiempo. Las pruebas con coseno
no tienen paro temprano: el coseno da sus mejores epocas al final, cuando la tasa ya es casi
cero, y un paro temprano las cortaba a la mitad (asi paso en la busqueda v2: las combinaciones
cercanas a la receta original pararon en 6 a 9 epocas). Las pruebas con meseta conservan el paro
temprano de configs/sin_tope (paciencia de 6 epocas sobre la perdida de validacion).

La primera prueba es siempre la receta original (AdamW, tasa 0.0003, decaimiento 0.05, coseno),
para que la busqueda parta de ella y solo pueda igualarla o mejorarla. Las primeras 5 pruebas son al azar para explorar; las demas las propone el metodo TPE
a partir de las anteriores. Todas las pruebas usan la misma semilla, asi que las diferencias
se deben a los hiperparametros.

Si gana el coseno, la configuracion ajustada conserva su horizonte: maximo --epocas epocas,
porque el coseno necesita saber de antemano cuantas habra. Si gana la meseta, conserva el
limite de seguridad de la configuracion sin tope.

Cada busqueda vive en su propia base de datos (--almacen). La primera (9 nov 2026, sin
programador y con 10 epocas por prueba) quedo en optuna_<modelo>.db; la actual usa
optuna_<modelo>_v2.db y la tercera (la actual, con la receta original como primera prueba y
sin paro temprano en las pruebas con coseno) optuna_<modelo>_v3.db, para no mezclar pruebas
hechas con reglas distintas.

Si Colab se desconecta, la busqueda se reanuda donde se quedo: el avance se guarda en una
base de datos (--almacen) que se respalda despues de cada prueba (--respaldo).

Al terminar escribe:
    configs/ajustado/<config>.yaml      la configuracion con los mejores valores
    resultados/optuna_<modelo>.csv      todas las pruebas, para el reporte

Uso:
    python tune.py --config configs/sin_tope/mobilenet_v3.yaml --pruebas 15 --epocas 15 \
        --almacen optuna_mobilenet_v3_large_v3.db
"""

import argparse
import copy
import os
import shutil

import lightning as L
import optuna
import torch
import yaml
from lightning.pytorch.callbacks import Callback, EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import ResiduosDataModule
from model import ResiduosModule


class PodaOptuna(Callback):
    """Informa a Optuna el F1 de validacion de cada epoca y corta la prueba si va mal."""

    def __init__(self, prueba: optuna.Trial):
        self.prueba = prueba
        self.mejor_f1 = 0.0
        self.historia = []

    def on_train_epoch_end(self, trainer, pl_module):
        # En Lightning este punto llega despues de la validacion de la epoca, cuando ya
        # estan disponibles la perdida de entrenamiento y la de validacion de esa epoca
        m = trainer.callback_metrics
        if "val/f1" not in m:
            return
        f1 = float(m["val/f1"])
        self.mejor_f1 = max(self.mejor_f1, f1)
        self.historia.append({"epoca": trainer.current_epoch + 1, "f1_val": f1,
                              "perdida_val": float(m["val/loss"]),
                              "perdida_ent": float(m["train/loss"]) if "train/loss" in m else None})
        self.prueba.report(f1, step=trainer.current_epoch)
        # En la ultima epoca ya no se ahorra nada: la prueba se deja terminar
        ultima = trainer.current_epoch + 1 >= trainer.max_epochs
        if not ultima and self.prueba.should_prune():
            raise optuna.TrialPruned(f"podada en la epoca {trainer.current_epoch + 1}")


def sugerir(prueba: optuna.Trial) -> dict:
    optimizador = prueba.suggest_categorical("optimizer", ["adamw", "adam", "radam", "sgd"])
    if optimizador == "sgd":
        lr = prueba.suggest_float("lr_sgd", 1e-3, 1e-1, log=True)
    else:
        lr = prueba.suggest_float("lr_adam", 1e-5, 1e-2, log=True)
    weight_decay = prueba.suggest_float("weight_decay", 1e-6, 1e-1, log=True)
    programador = prueba.suggest_categorical("programador", ["coseno", "meseta"])
    return {"optimizer": optimizador, "lr": lr, "weight_decay": weight_decay, "programador": programador}


def valores(params: dict) -> dict:
    """Convierte los parametros guardados por Optuna a los del modelo."""
    lr = params["lr_sgd"] if params["optimizer"] == "sgd" else params["lr_adam"]
    # Las pruebas de la primera busqueda no elegian programador: usaban la meseta de configs/sin_tope
    return {"optimizer": params["optimizer"], "lr": float(lr), "weight_decay": float(params["weight_decay"]),
            "programador": params.get("programador", "meseta")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML de configs/sin_tope/")
    ap.add_argument("--pruebas", type=int, default=15, help="numero total de combinaciones a probar")
    ap.add_argument("--epocas", type=int, default=15, help="epocas maximas por prueba (y horizonte del coseno)")
    ap.add_argument("--semilla", type=int, default=42)
    ap.add_argument("--almacen", default=None, help="base de datos local de Optuna")
    ap.add_argument("--respaldo", default=None, help="copia de la base de datos en Drive")
    ap.add_argument("--resultados", default="resultados")
    a = ap.parse_args()

    torch.set_float32_matmul_precision("high")
    cfg = yaml.safe_load(open(a.config))
    modelo = cfg["model"]["backbone"]
    nombre_config = os.path.basename(a.config)
    almacen = a.almacen or f"optuna_{modelo}.db"
    nombre_busqueda = os.path.splitext(os.path.basename(almacen))[0]

    # Reanudar: si hay respaldo en Drive y no hay copia local, se trae
    if a.respaldo and os.path.exists(a.respaldo) and not os.path.exists(almacen):
        shutil.copy(a.respaldo, almacen)

    previas = []
    if os.path.exists(almacen):
        try:
            previas = optuna.load_study(study_name=modelo, storage=f"sqlite:///{almacen}").trials
        except KeyError:
            previas = []
    estudio = optuna.create_study(
        study_name=modelo, storage=f"sqlite:///{almacen}", load_if_exists=True, direction="maximize",
        # La semilla del muestreador cambia con las pruebas ya hechas: al reanudar no repite
        # las mismas combinaciones del principio
        sampler=optuna.samplers.TPESampler(seed=a.semilla + len(previas), n_startup_trials=5),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    if not estudio.trials:
        # Primera prueba: la receta original (la de la version entregada)
        estudio.enqueue_trial({"optimizer": cfg["model"]["optimizer"], "lr_adam": float(cfg["model"]["lr"]),
                               "weight_decay": float(cfg["model"]["weight_decay"]), "programador": "coseno"})
    terminadas = [t for t in estudio.trials
                  if t.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)]
    faltan = max(0, a.pruebas - len(terminadas))
    print(f"{modelo}: {len(terminadas)} pruebas ya hechas, faltan {faltan}")

    datos = ResiduosDataModule(**cfg["data"])
    trainer_cfg = cfg["trainer"]

    def objetivo(prueba: optuna.Trial) -> float:
        params = sugerir(prueba)
        print(f"\nPrueba {prueba.number}: {params}")
        L.seed_everything(a.semilla, workers=True, verbose=False)
        red = ResiduosModule(**{**cfg["model"], **params})
        poda = PodaOptuna(prueba)
        callbacks = [poda]
        if params["programador"] == "meseta":
            callbacks.append(EarlyStopping(monitor="val/loss", mode="min", patience=6, min_delta=0.001))
        trainer = L.Trainer(
            max_epochs=a.epocas,
            accelerator=trainer_cfg.get("accelerator", "auto"),
            devices=trainer_cfg.get("devices", "auto"),
            precision=trainer_cfg.get("precision", "16-mixed") if torch.cuda.is_available() else "32-true",
            logger=TensorBoardLogger("runs_optuna", name=nombre_busqueda, version=f"prueba_{prueba.number:02d}",
                                     default_hp_metric=False),
            callbacks=callbacks,
            enable_checkpointing=False, enable_progress_bar=False, enable_model_summary=False,
        )
        try:
            trainer.fit(red, datamodule=datos)
        finally:
            prueba.set_user_attr("historia", poda.historia)
            prueba.set_user_attr("epocas", len(poda.historia))
        print(f"Prueba {prueba.number}: mejor F1 val {poda.mejor_f1:.3f} en {len(poda.historia)} epocas")
        return poda.mejor_f1

    def respaldar(estudio, prueba):
        if a.respaldo:
            os.makedirs(os.path.dirname(a.respaldo) or ".", exist_ok=True)
            shutil.copy(almacen, a.respaldo)

    if faltan:
        estudio.optimize(objetivo, n_trials=faltan, callbacks=[respaldar], catch=(RuntimeError,))

    completas = [t for t in estudio.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completas:
        print("No hay pruebas completas todavia.")
        return

    # Tabla de todas las pruebas para el reporte
    filas = []
    for t in estudio.trials:
        if t.state not in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED):
            continue
        v = valores(t.params)
        historia = t.user_attrs.get("historia", [])
        filas.append({"prueba": t.number, "estado": "completa" if t.state.name == "COMPLETE" else "podada",
                      "f1_val": t.value if t.value is not None else (max(h["f1_val"] for h in historia) if historia else None),
                      "epocas": t.user_attrs.get("epocas"), **v})
    os.makedirs(a.resultados, exist_ok=True)
    import pandas as pd
    tabla = pd.DataFrame(filas).sort_values("prueba")
    tabla.insert(0, "modelo", modelo)
    tabla.to_csv(os.path.join(a.resultados, f"optuna_{modelo}.csv"), index=False)

    mejor = valores(estudio.best_trial.params)
    print(f"\nMejor combinacion para {modelo} (prueba {estudio.best_trial.number}, "
          f"F1 val {estudio.best_value:.3f}): {mejor}")
    print(f"Pruebas: {len(completas)} completas, {len(filas) - len(completas)} podadas")

    # Configuracion ajustada: la de sin tope con los mejores valores
    nuevo = copy.deepcopy(cfg)
    nuevo["model"].update(mejor)
    if mejor["programador"] == "coseno":
        # El coseno baja la tasa hasta casi cero en max_epochs epocas: se conserva el mismo
        # horizonte con el que gano en la busqueda y, como en la busqueda, sin paro temprano
        nuevo["trainer"]["max_epochs"] = a.epocas
        nuevo["trainer"]["callbacks"] = [cb for cb in nuevo["trainer"]["callbacks"]
                                         if not cb["class_path"].endswith("EarlyStopping")]
    nuevo["trainer"]["logger"]["init_args"]["name"] = f"{modelo}_ajustado"
    for cb in nuevo["trainer"]["callbacks"]:
        if cb["class_path"].endswith("RegistroResultados"):
            cb["init_args"]["experimento"] = "ajustado"
    os.makedirs("configs/ajustado", exist_ok=True)
    ruta = os.path.join("configs/ajustado", nombre_config)
    with open(ruta, "w") as f:
        f.write(f"# Configuracion generada por tune.py a partir de {a.config}\n"
                f"# Mejores valores de Optuna: prueba {estudio.best_trial.number} de {len(filas)}, "
                f"F1 macro de validacion {estudio.best_value:.3f} con {a.epocas} epocas maximas por prueba.\n"
                f"# Uso: python main.py fit --config {ruta} --seed_everything 42\n\n")
        yaml.safe_dump(nuevo, f, sort_keys=False, allow_unicode=True)
    if a.respaldo:
        shutil.copy(ruta, os.path.join(os.path.dirname(a.respaldo), nombre_config))
    print(f"Guardada {ruta}")


if __name__ == "__main__":
    main()
