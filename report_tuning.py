"""Reporte de los experimentos de entrenamiento: original, sin tope y ajustado.

Compara, para cada modelo, tres formas de entrenarlo:
    original   15 epocas como maximo y la misma receta para todos (version entregada, v1.0-entrega)
    control    la receta original repetida en el Colab actual (configs/control), semilla 42
    sin_tope   misma receta, pero cada modelo para solo cuando su perdida de validacion deja de bajar
    ajustado   sin tope y con los hiperparametros que encontro Optuna (tune.py)

Ademas del F1, reporta la BRECHA: perdida de validacion menos perdida de entrenamiento en la
epoca elegida. Mide que tan separadas quedan las dos curvas; cuanto mas chica, mas "pegadas".

Genera:
    resultados/tabla_experimentos.csv
    resultados/tabla_optuna.csv                     mejor combinacion de cada modelo
    resultados/figuras/curvas_experimentos.png      perdidas de entrenamiento y validacion
    resultados/figuras/optuna.png                   F1 de cada prueba de Optuna

Uso:
    python report_tuning.py --original /content/drive/MyDrive/proyecto_IA/repo_resultados
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

MODELOS = ["mobilenet_v3_large", "resnet50", "yolo11n_cls"]
EXPERIMENTOS = ["original", "control", "sin_tope", "ajustado"]
COLORES = {"original": "tab:gray", "control": "tab:green", "sin_tope": "tab:blue", "ajustado": "tab:orange"}


def media_desv(s, dec=3):
    s = s.dropna()
    if len(s) == 0:
        return "-"
    return f"{s.mean():.{dec}f}" if len(s) == 1 else f"{s.mean():.{dec}f} ± {s.std(ddof=1):.{dec}f}"


def brecha(r):
    """Perdida de validacion menos perdida de entrenamiento en la mejor epoca."""
    h = r["historia"]
    if not h or not r.get("mejor_epoca"):
        return None
    e = h[int(r["mejor_epoca"]) - 1]
    if e.get("perdida_val") is None or e.get("perdida_ent") is None:
        return None
    return e["perdida_val"] - e["perdida_ent"]


def leer(carpeta, experimento_por_omision=None):
    filas = []
    for f in sorted(glob.glob(os.path.join(carpeta, "*.json"))):
        if f.endswith("linea_base.json"):
            continue
        r = json.load(open(f))
        if "modelo" not in r or "historia" not in r:
            continue
        r["experimento"] = r.get("experimento") or experimento_por_omision or "original"
        filas.append(r)
    return filas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--original", default=None,
                    help="carpeta con los resultados de la version entregada (repo_resultados)")
    a = ap.parse_args()

    filas = leer(a.resultados)
    if a.original and os.path.isdir(a.original):
        filas += leer(a.original, "original")
    filas = [r for r in filas if r["modelo"] in MODELOS]
    assert filas, "No hay resultados todavia"
    res = pd.DataFrame(filas)
    res["brecha"] = [brecha(r) for r in filas]
    res["epocas"] = [len(r["historia"]) for r in filas]
    dir_fig = os.path.join(a.resultados, "figuras")
    os.makedirs(dir_fig, exist_ok=True)

    # Tabla de los tres experimentos
    tabla = res.groupby(["modelo", "experimento"]).agg(
        semillas=("semilla", "count"),
        f1_macro_val=("f1_macro_val", media_desv),
        exactitud_val=("exactitud_val", media_desv),
        epocas=("epocas", lambda s: media_desv(s, 1)),
        mejor_epoca=("mejor_epoca", lambda s: media_desv(s, 1)),
        brecha=("brecha", media_desv),
    )
    orden = [(m, e) for m in MODELOS for e in EXPERIMENTOS if (m, e) in tabla.index]
    tabla = tabla.loc[orden]
    print("Experimentos (validacion, media ± desviacion entre semillas). "
          "brecha = perdida val menos perdida ent en la mejor epoca\n")
    print(tabla.to_string())
    tabla.to_csv(os.path.join(a.resultados, "tabla_experimentos.csv"))

    # Hiperparametros usados en cada experimento
    hp = (res.drop_duplicates(["modelo", "experimento"])
          .set_index(["modelo", "experimento"])["hiperparametros"].apply(pd.Series))
    columnas = [c for c in ["optimizer", "lr", "weight_decay", "programador"] if c in hp]
    hp = hp.reindex(orden)[columnas]
    if "programador" in hp:  # la version entregada no guardaba este dato; usaba coseno
        hp["programador"] = hp["programador"].fillna("coseno")
    print("\nHiperparametros de cada experimento:\n")
    print(hp.to_string())

    # Optuna
    tablas_optuna = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(a.resultados, "optuna_*.csv")))]
    if tablas_optuna:
        optuna_todo = pd.concat(tablas_optuna)
        # La mejor combinacion es la misma que eligio tune.py: la de mayor F1 entre las pruebas
        # completas. Las podadas pueden haber alcanzado un F1 parecido antes de cortarse, pero
        # no terminaron su entrenamiento.
        completas_todo = optuna_todo[optuna_todo["estado"] == "completa"]
        mejores = completas_todo.sort_values("f1_val", ascending=False).groupby("modelo").head(1).set_index("modelo")
        resumen = optuna_todo.groupby("modelo").agg(pruebas=("prueba", "count"),
                                                    podadas=("estado", lambda s: int((s == "podada").sum())))
        cols = [c for c in ["prueba", "f1_val", "optimizer", "lr", "weight_decay", "programador"] if c in mejores]
        resumen = resumen.join(mejores[cols]
                               .rename(columns={"prueba": "mejor_prueba", "f1_val": "mejor_f1_val"}))
        print("\nBusqueda con Optuna (F1 de validacion con epocas limitadas por prueba):\n")
        print(resumen.to_string())
        resumen.to_csv(os.path.join(a.resultados, "tabla_optuna.csv"))

        modelos_opt = list(resumen.index)
        fig, ejes = plt.subplots(1, len(modelos_opt), figsize=(5 * len(modelos_opt), 3.8), squeeze=False)
        for ax, m in zip(ejes[0], modelos_opt):
            t = optuna_todo[optuna_todo["modelo"] == m]
            for opt, g in t.groupby("optimizer"):
                completas = g[g["estado"] == "completa"]
                podadas = g[g["estado"] == "podada"]
                p = ax.scatter(completas["prueba"], completas["f1_val"], label=opt, s=30)
                ax.scatter(podadas["prueba"], podadas["f1_val"], marker="x", s=30, color=p.get_facecolor()[0])
            ax.set_title(f"{m}: pruebas de Optuna")
            ax.set_xlabel("numero de prueba (x = podada)")
            ax.set_ylabel("mejor F1 de validacion")
            ax.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(dir_fig, "optuna.png"), dpi=150)
        plt.close()

    # Curvas: una corrida por modelo y experimento, de preferencia la semilla 42
    elegidas = (res.assign(orden=(res["semilla"] != 42).astype(int))
                .sort_values(["orden", "semilla"]).groupby(["modelo", "experimento"]).head(1))
    elegidas = elegidas.assign(n_exp=elegidas["experimento"].map(
        {e: i for i, e in enumerate(EXPERIMENTOS)})).sort_values("n_exp")
    modelos = [m for m in MODELOS if m in set(elegidas["modelo"])]
    fig, ejes = plt.subplots(1, len(modelos), figsize=(5.5 * len(modelos), 4.3), squeeze=False)
    for ax, m in zip(ejes[0], modelos):
        for _, r in elegidas[elegidas["modelo"] == m].iterrows():
            h = pd.DataFrame(r["historia"])
            c = COLORES.get(r["experimento"], "black")
            ax.plot(h["epoca"], h["perdida_ent"], color=c, ls="--", lw=1.2)
            ax.plot(h["epoca"], h["perdida_val"], color=c, lw=1.8, label=r["experimento"])
            ax.axvline(r["mejor_epoca"], color=c, ls=":", lw=1)
        ax.set_title(f"{m}, semilla 42")
        ax.set_xlabel("epoca")
        ax.set_ylabel("perdida (linea continua: validacion; punteada: entrenamiento)", fontsize=8)
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_fig, "curvas_experimentos.png"), dpi=150)
    plt.close()
    print(f"\nFiguras guardadas en {dir_fig}")


if __name__ == "__main__":
    main()
