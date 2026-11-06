"""Utilidades: callback que registra resultados y costo de cada corrida.

El callback ``RegistroResultados`` se declara en los YAML de configs/. Al terminar
el entrenamiento carga el mejor modelo, lo evalua en validacion y guarda en
``resultados/<backbone>_semilla<semilla>.json``: F1 macro, exactitud, mejor epoca,
historia por epoca, parametros, tamano, minutos de entrenamiento y milisegundos por
foto en CPU. report.py junta esos archivos en la tabla comparativa y las figuras.
"""

import copy
import json
import os
import time

import numpy as np
import torch
import lightning as L
from sklearn.metrics import f1_score, accuracy_score


def contar_parametros(modelo: torch.nn.Module) -> int:
    return sum(p.numel() for p in modelo.parameters())


def medir_inferencia_cpu(modelo: torch.nn.Module, img_size: int = 224, hilos: int = 4,
                         repeticiones: int = 30) -> float:
    """Mediana de milisegundos por foto, una foto a la vez, en CPU con ``hilos`` hilos.

    Es una aproximacion a la Raspberry Pi 5 (4 nucleos); no sustituye la medicion
    en el dispositivo.
    """
    m = copy.deepcopy(modelo).to("cpu").eval()
    antes = torch.get_num_threads()
    torch.set_num_threads(hilos)
    x = torch.randn(1, 3, img_size, img_size)
    tiempos = []
    with torch.no_grad():
        for i in range(repeticiones + 5):
            t0 = time.perf_counter()
            m(x)
            if i >= 5:  # las primeras 5 son de calentamiento
                tiempos.append((time.perf_counter() - t0) * 1000)
    torch.set_num_threads(antes)
    return float(np.median(tiempos))


def medir_rendimiento(modelo: torch.nn.Module, img_size: int = 224, lote: int = 32,
                      pasos: int = 20, dispositivo: str = None) -> dict:
    """Fotos por segundo que el modelo procesa al entrenar y al inferir, con datos sinteticos.

    Al usar tensores ya en memoria se elimina el tiempo de leer y transformar fotos, de modo
    que la cifra refleja solo el costo de computo de la arquitectura.
    """
    dispositivo = dispositivo or ("cuda" if torch.cuda.is_available() else "cpu")
    m = copy.deepcopy(modelo).to(dispositivo)
    usar_amp = dispositivo == "cuda"
    x = torch.randn(lote, 3, img_size, img_size, device=dispositivo)
    y = torch.randint(0, 6, (lote,), device=dispositivo)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    escalador = torch.amp.GradScaler(enabled=usar_amp)
    criterio = torch.nn.CrossEntropyLoss()
    sincronizar = torch.cuda.synchronize if dispositivo == "cuda" else (lambda: None)
    if dispositivo == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def paso_entrenamiento():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dispositivo, enabled=usar_amp):
            perdida = criterio(m(x), y)
        escalador.scale(perdida).backward()
        escalador.step(opt)
        escalador.update()

    m.train()
    for _ in range(3):
        paso_entrenamiento()
    sincronizar()
    t0 = time.perf_counter()
    for _ in range(pasos):
        paso_entrenamiento()
    sincronizar()
    fotos_s_ent = pasos * lote / (time.perf_counter() - t0)

    m.eval()
    with torch.no_grad(), torch.autocast(device_type=dispositivo, enabled=usar_amp):
        for _ in range(3):
            m(x)
        sincronizar()
        t0 = time.perf_counter()
        for _ in range(pasos):
            m(x)
        sincronizar()
    fotos_s_inf = pasos * lote / (time.perf_counter() - t0)
    memoria = torch.cuda.max_memory_allocated() / 1e6 if dispositivo == "cuda" else None
    return {"fotos_por_s_entrenamiento": fotos_s_ent, "fotos_por_s_inferencia": fotos_s_inf,
            "memoria_GPU_MB": memoria, "dispositivo": dispositivo}


@torch.no_grad()
def predecir(modelo, cargador, dispositivo):
    modelo.eval().to(dispositivo)
    reales, predichas = [], []
    for x, y in cargador:
        predichas += modelo(x.to(dispositivo)).argmax(1).cpu().tolist()
        reales += y.tolist()
    return reales, predichas


class RegistroResultados(L.Callback):
    """Registra historia, desempeno en validacion y costo de cada corrida."""

    def __init__(self, dir_resultados: str = "resultados", hilos_cpu: int = 4, experimento: str = ""):
        """experimento: etiqueta que se agrega al nombre del archivo (por ejemplo "sin_tope" o
        "ajustado"), para que corridas de distintos experimentos no se sobrescriban."""
        super().__init__()
        self.dir_resultados = dir_resultados
        self.hilos_cpu = hilos_cpu
        self.experimento = experimento

    def on_fit_start(self, trainer, pl_module):
        self.inicio = time.time()
        self.historia = []
        print(f"\nEntrenando {pl_module.hparams.backbone} con semilla {os.environ.get('PL_GLOBAL_SEED')}")

    def on_train_epoch_start(self, trainer, pl_module):
        self.inicio_epoca = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        # En Lightning este punto llega despues de la validacion de la epoca
        m = trainer.callback_metrics
        valor = lambda k: float(m[k]) if k in m else None
        fila = {
            "epoca": trainer.current_epoch + 1,
            "perdida_ent": valor("train/loss"),
            "perdida_val": valor("val/loss"),
            "f1_val": valor("val/f1"),
            "exactitud_val": valor("val/acc"),
            "segundos": time.time() - self.inicio_epoca,
        }
        self.historia.append(fila)
        fmt = lambda v: f"{v:.3f}" if v is not None else "  -  "
        print(f"  epoca {fila['epoca']:2d} | perdida ent {fmt(fila['perdida_ent'])} | "
              f"perdida val {fmt(fila['perdida_val'])} | F1 val {fmt(fila['f1_val'])} | "
              f"{fila['segundos']:.0f} s")

    def on_fit_end(self, trainer, pl_module):
        minutos = (time.time() - self.inicio) / 60
        ruta_ckpt = trainer.checkpoint_callback.best_model_path
        mejor = type(pl_module).load_from_checkpoint(ruta_ckpt, map_location="cpu") if ruta_ckpt else pl_module

        reales, predichas = predecir(mejor, trainer.datamodule.val_dataloader(), pl_module.device)
        backbone = pl_module.hparams.backbone
        semilla = int(os.environ.get("PL_GLOBAL_SEED", -1))
        nombre = f"{backbone}_{self.experimento}_semilla{semilla}" if self.experimento else f"{backbone}_semilla{semilla}"

        os.makedirs(os.path.join(self.dir_resultados, "pesos"), exist_ok=True)
        ruta_pesos = os.path.join(self.dir_resultados, "pesos", f"{nombre}.pt")
        torch.save(mejor.net.state_dict(), ruta_pesos)

        f1s = [h["f1_val"] if h["f1_val"] is not None else -1 for h in self.historia]
        mejor_epoca = int(np.argmax(f1s)) + 1 if f1s else None
        # Brecha: cuanto mas alta es la perdida de validacion que la de entrenamiento en la
        # epoca elegida. Cuanto mas chica, mas "pegadas" las curvas (menos memorizacion).
        h_mejor = self.historia[mejor_epoca - 1] if mejor_epoca else {}
        brecha = (h_mejor["perdida_val"] - h_mejor["perdida_ent"]
                  if h_mejor.get("perdida_val") is not None and h_mejor.get("perdida_ent") is not None else None)
        resultado = {
            "modelo": backbone,
            "experimento": self.experimento or "original",
            "semilla": semilla,
            "version_tensorboard": getattr(trainer.logger, "version", None),
            "checkpoint": ruta_ckpt,
            "mejor_epoca": mejor_epoca,
            "brecha_perdida_mejor_epoca": brecha,
            "f1_macro_val": f1_score(reales, predichas, average="macro", zero_division=0),
            "exactitud_val": accuracy_score(reales, predichas),
            "parametros_millones": contar_parametros(mejor.net) / 1e6,
            "tamano_MB": os.path.getsize(ruta_pesos) / 1e6,
            "minutos_entrenamiento": minutos,
            "epocas_entrenadas": len(self.historia),
            "segundos_por_epoca": float(np.mean([h["segundos"] for h in self.historia])) if self.historia else None,
            "ms_por_foto_cpu": medir_inferencia_cpu(mejor.net, trainer.datamodule.hparams.img_size,
                                                    self.hilos_cpu),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
            "hiperparametros": {k: v for k, v in pl_module.hparams.items()},
            "historia": self.historia,
            "reales_val": reales,
            "predichas_val": predichas,
        }
        os.makedirs(self.dir_resultados, exist_ok=True)
        with open(os.path.join(self.dir_resultados, f"{nombre}.json"), "w") as f:
            json.dump(resultado, f, indent=1)
        texto_brecha = f", brecha {brecha:.3f}" if brecha is not None else ""
        print(f"\n{nombre}: F1 macro validacion {resultado['f1_macro_val']:.3f} "
              f"(mejor epoca {resultado['mejor_epoca']}{texto_brecha}), {minutos:.1f} min, "
              f"{resultado['parametros_millones']:.1f} M parametros, "
              f"{resultado['ms_por_foto_cpu']:.0f} ms por foto en CPU")
