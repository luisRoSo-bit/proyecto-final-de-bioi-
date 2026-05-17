"""
============================================================
ANALISIS DE DATOS BIOMEDICOS EN EJERCICIO
============================================================
Procesa archivos CSV con variantes de "ejercicio" en el nombre.
Mantiene consistencia de etiquetas con analisis_reposo.py:
  Persona 1 = Harold, Persona 2 = Juanes, Persona 3 = Lau,
  Persona 4 = Luis,   Persona 5 = Mariana
Excluye archivos de usuarios no presentes en el analisis de reposo.
============================================================
"""

import os
import re
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Configuracion ──────────────────────────────────────────
CARPETA_DATOS = os.path.dirname(os.path.abspath(__file__))
CARPETA_SALIDA = os.path.join(CARPETA_DATOS, "resultados_ejercicio")

# Rangos fisiologicos para EJERCICIO
RANGOS = {
    'bpm':     (60, 200),
    'temp':    (30, 42),
    'ppg':     (0, 500000),
    'ecg':     (-5, 5),
    'posture': (0, 20),
}

# Usuarios a EXCLUIR del analisis (no tienen datos de reposo)
EXCLUIR = ['alejo']

# Mapeo fijo para consistencia con reposo
MAPEO_PERSONAS = {
    'harold':  'Persona 1',
    'juanes':  'Persona 2',
    'lau':     'Persona 3',
    'luis':    'Persona 4',
    'mari':    'Persona 5',
    'mariana': 'Persona 5',
}

COLORES = ['#6C5CE7', '#00B894', '#E17055', '#0984E3', '#FDCB6E',
           '#E84393', '#00CEC9', '#2D3436', '#55EFC4', '#FAB1A0']

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor':   '#16213e',
    'axes.edgecolor':   '#e0e0e0',
    'axes.labelcolor':  '#e0e0e0',
    'text.color':       '#e0e0e0',
    'xtick.color':      '#e0e0e0',
    'ytick.color':      '#e0e0e0',
    'grid.color':       '#2a2a4a',
    'grid.alpha':       0.5,
    'font.family':      'sans-serif',
    'font.size':        11,
})


# ── 1. Descubrir archivos de ejercicio ─────────────────────
def descubrir_archivos(carpeta):
    """Busca CSVs con variantes de 'ejercicio' (ejercicio, ejericio, jercicio).
    Excluye archivos de usuarios en la lista EXCLUIR."""
    todos = glob.glob(os.path.join(carpeta, "*.csv"))
    patron = re.compile(r'(ejercicio|ejericio|jercicio)', re.IGNORECASE)
    ejercicio = [f for f in todos if patron.search(os.path.basename(f))]
    # Filtrar usuarios excluidos
    ejercicio = [f for f in ejercicio
                 if not any(ex in os.path.basename(f).lower() for ex in EXCLUIR)]
    return sorted(ejercicio)


def asignar_persona(nombre_archivo):
    """Asigna etiqueta de persona consistente con reposo."""
    base = os.path.basename(nombre_archivo).lower()
    for clave, persona in MAPEO_PERSONAS.items():
        if clave in base:
            return persona
    return None  # se asigna despues


# ── 2. Cargar y limpiar datos ──────────────────────────────
def cargar_y_limpiar(archivo):
    """Carga CSV, aplica filtros de rango y deteccion IQR."""
    df = pd.read_csv(archivo)
    df.columns = df.columns.str.strip().str.lower()

    total_original = len(df)
    df.dropna(how='all', subset=[c for c in df.columns if c != 'time'], inplace=True)

    # --- Filtrado por rangos fisiologicos ---
    mascara = pd.Series(True, index=df.index)
    for col, (lo, hi) in RANGOS.items():
        if col in df.columns:
            fuera = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            mascara &= ~fuera

    df_rango = df[mascara].copy()

    # --- Deteccion IQR adicional (solo bpm y ecg) ---
    outliers_iqr = 0
    for col in ['bpm', 'ecg']:
        if col in df_rango.columns:
            datos = df_rango[col].dropna()
            if len(datos) > 10:
                q1 = datos.quantile(0.25)
                q3 = datos.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 2.0 * iqr   # factor 2x para ejercicio (mas variabilidad)
                upper = q3 + 2.0 * iqr
                iqr_mask = df_rango[col].notna() & ((df_rango[col] < lower) | (df_rango[col] > upper))
                outliers_iqr += iqr_mask.sum()
                df_rango = df_rango[~iqr_mask]

    descartadas = total_original - len(df_rango)
    return df_rango, total_original, descartadas, outliers_iqr


# ── 3. Calcular estadisticas ──────────────────────────────
def calcular_stats(df, persona):
    metricas = ['bpm', 'temp', 'ppg', 'ecg', 'posture']
    fila = {'Persona': persona}
    for m in metricas:
        if m in df.columns:
            datos = df[m].dropna()
            if len(datos) > 0:
                fila[f'{m}_promedio'] = round(datos.mean(), 4)
                fila[f'{m}_std']     = round(datos.std(), 4)
                fila[f'{m}_min']     = round(datos.min(), 4)
                fila[f'{m}_max']     = round(datos.max(), 4)
                fila[f'{m}_mediana'] = round(datos.median(), 4)
    fila['mediciones_validas'] = len(df)
    return fila


# ── 4. Graficas ───────────────────────────────────────────
def grafica_barras_comparativa(resumen, carpeta_salida):
    metricas = [('bpm_promedio', 'Frecuencia Cardiaca en Ejercicio (BPM)', 'BPM'),
                ('temp_promedio', 'Temperatura en Ejercicio (C)', 'C'),
                ('ppg_promedio', 'PPG Promedio en Ejercicio', 'u.a.'),
                ('ecg_promedio', 'ECG Promedio en Ejercicio', 'mV')]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparacion de Signos Vitales en EJERCICIO por Persona',
                 fontsize=18, fontweight='bold', y=0.98, color='#FDFEFE')

    for ax, (col, titulo, unidad) in zip(axes.flat, metricas):
        if col not in resumen.columns or resumen[col].dropna().empty:
            ax.text(0.5, 0.5, 'Sin datos validos', transform=ax.transAxes,
                    ha='center', va='center', fontsize=14, color='#888')
            ax.set_title(titulo, fontsize=13, fontweight='bold', pad=10)
            continue
        vals = resumen[col].values
        personas = resumen['Persona'].values
        bars = ax.bar(personas, vals, color=COLORES[:len(vals)],
                      edgecolor='white', linewidth=0.5, width=0.6)
        ax.set_title(titulo, fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel(unidad, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{v:.1f}', ha='center', va='bottom', fontsize=9,
                        fontweight='bold', color='#FDFEFE')
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    ruta = os.path.join(carpeta_salida, "comparacion_ejercicio.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {os.path.basename(ruta)}")


def grafica_radar(resumen, carpeta_salida):
    metricas_r = ['bpm_promedio', 'ppg_promedio', 'posture_promedio']
    etiquetas  = ['BPM', 'PPG', 'Postura']

    disponibles = [m for m in metricas_r if m in resumen.columns and resumen[m].notna().any()]
    etiquetas_d = [etiquetas[metricas_r.index(m)] for m in disponibles]
    if len(disponibles) < 3:
        return

    datos_norm = resumen[disponibles].copy()
    for c in disponibles:
        rng = datos_norm[c].max() - datos_norm[c].min()
        datos_norm[c] = (datos_norm[c] - datos_norm[c].min()) / rng if rng > 0 else 0.5

    N = len(disponibles)
    angulos = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    angulos += angulos[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#1a1a2e')
    ax.set_facecolor('#16213e')

    for i, (_, row) in enumerate(resumen.iterrows()):
        vals = datos_norm.iloc[i].values.tolist() + [datos_norm.iloc[i].values[0]]
        ax.plot(angulos, vals, 'o-', linewidth=2, color=COLORES[i % len(COLORES)],
                label=row['Persona'], markersize=6)
        ax.fill(angulos, vals, alpha=0.1, color=COLORES[i % len(COLORES)])

    ax.set_xticks(angulos[:-1])
    ax.set_xticklabels(etiquetas_d, fontsize=12, fontweight='bold')
    ax.set_title('Perfil Normalizado en Ejercicio', fontsize=16,
                 fontweight='bold', pad=20, color='#FDFEFE')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(color='#4a4a6a', alpha=0.4)

    ruta = os.path.join(carpeta_salida, "radar_ejercicio.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {os.path.basename(ruta)}")


def grafica_boxplot(datos_limpios, carpeta_salida):
    metricas = ['bpm', 'ecg']
    titulos = {'bpm': 'Frecuencia Cardiaca (BPM) en Ejercicio',
               'ecg': 'ECG en Ejercicio (mV)'}

    for metrica in metricas:
        todas, labels = [], []
        for persona, df in datos_limpios.items():
            if metrica in df.columns:
                vals = df[metrica].dropna()
                if len(vals) > 0:
                    todas.append(vals.values)
                    labels.append(persona)
        if not todas:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        bp = ax.boxplot(todas, patch_artist=True, labels=labels,
                        medianprops=dict(color='#FDFEFE', linewidth=2),
                        whiskerprops=dict(color='#e0e0e0'),
                        capprops=dict(color='#e0e0e0'),
                        flierprops=dict(markerfacecolor='#E17055', marker='o',
                                        markersize=4, alpha=0.6))
        for patch, color in zip(bp['boxes'], COLORES[:len(todas)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(titulos[metrica], fontsize=15, fontweight='bold', color='#FDFEFE')
        ax.set_ylabel(titulos[metrica].split('(')[-1].replace(')', ''), fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.tick_params(axis='x', rotation=30)

        ruta = os.path.join(carpeta_salida, f"boxplot_{metrica}_ejercicio.png")
        fig.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  [OK] {os.path.basename(ruta)}")


def grafica_heatmap(resumen, carpeta_salida):
    cols = [c for c in resumen.columns if '_promedio' in c]
    if not cols:
        return

    datos = resumen[cols].copy()
    datos.columns = [c.replace('_promedio', '').upper() for c in cols]
    datos.index = resumen['Persona']

    datos_norm = datos.copy()
    for c in datos_norm.columns:
        rng = datos_norm[c].max() - datos_norm[c].min()
        if rng > 0:
            datos_norm[c] = (datos_norm[c] - datos_norm[c].min()) / rng

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(datos_norm.values, cmap='magma', aspect='auto')

    ax.set_xticks(range(len(datos_norm.columns)))
    ax.set_xticklabels(datos_norm.columns, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(datos_norm.index)))
    ax.set_yticklabels(datos_norm.index, fontsize=11)

    for i in range(len(datos_norm.index)):
        for j in range(len(datos_norm.columns)):
            val_real = datos.iloc[i, j]
            txt = f'{val_real:.1f}' if not np.isnan(val_real) else 'N/A'
            c = 'white' if (np.isnan(datos_norm.iloc[i, j]) or datos_norm.iloc[i, j] < 0.5) else 'black'
            ax.text(j, i, txt, ha='center', va='center', fontsize=9, fontweight='bold', color=c)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Valor normalizado', color='#e0e0e0')
    cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e0e0e0')

    ax.set_title('Mapa de Calor - Promedios en EJERCICIO',
                 fontsize=15, fontweight='bold', pad=15, color='#FDFEFE')
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, "heatmap_ejercicio.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {os.path.basename(ruta)}")


def grafica_outliers(resumen, outliers_por_persona, carpeta_salida):
    fig, ax = plt.subplots(figsize=(10, 6))
    personas = resumen['Persona'].values
    validas = resumen['mediciones_validas'].values
    descartadas = [resumen.loc[resumen['Persona'] == p, 'descartadas'].values[0] for p in personas]
    outliers_iqr = [outliers_por_persona.get(p, 0) for p in personas]
    totales = [resumen.loc[resumen['Persona'] == p, 'total_original'].values[0] for p in personas]
    pct_outliers = [100 * d / t if t > 0 else 0 for d, t in zip(descartadas, totales)]

    x = np.arange(len(personas))
    w = 0.35

    b1 = ax.bar(x - w/2, validas, w, label='Validas', color='#00B894',
                edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w/2, descartadas, w, label='Descartadas', color='#E17055',
                edgecolor='white', linewidth=0.5)

    # Porcentaje de outliers como texto
    for i, (bar, pct) in enumerate(zip(b2, pct_outliers)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{pct:.1f}%', ha='center', va='bottom', fontsize=9,
                fontweight='bold', color='#FDCB6E')

    for bar, v in zip(b1, validas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                str(v), ha='center', va='bottom', fontsize=8, color='#FDFEFE')
    for bar, v in zip(b2, descartadas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 50,
                str(v), ha='center', va='top', fontsize=8, color='#FDFEFE')

    ax.set_xticks(x)
    ax.set_xticklabels(personas, rotation=30)
    ax.set_title('Mediciones Validas vs Outliers Descartados (Ejercicio)',
                 fontsize=15, fontweight='bold', color='#FDFEFE')
    ax.set_ylabel('Cantidad de registros')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, "outliers_ejercicio.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {os.path.basename(ruta)}")


def grafica_desviaciones(resumen, carpeta_salida):
    cols_std = [c for c in resumen.columns if '_std' in c]
    if not cols_std:
        return

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(resumen))
    ancho = 0.15
    n = len(cols_std)

    for i, col in enumerate(cols_std):
        offset = (i - n/2 + 0.5) * ancho
        nombre = col.replace('_std', '').upper()
        vals = resumen[col].fillna(0).values
        ax.bar(x + offset, vals, ancho, label=nombre,
               color=COLORES[i % len(COLORES)], edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(resumen['Persona'], rotation=30)
    ax.set_title('Desviacion Estandar por Metrica en Ejercicio',
                 fontsize=15, fontweight='bold', color='#FDFEFE')
    ax.set_ylabel('Desviacion Estandar')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, "desviacion_ejercicio.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {os.path.basename(ruta)}")


def tabla_resumen_imagen(resumen, carpeta_salida):
    fig, ax = plt.subplots(figsize=(16, 3 + 0.5 * len(resumen)))
    ax.axis('off')

    cols_show = ['Persona']
    rename = {'Persona': 'Persona'}
    for c in resumen.columns:
        if c == 'Persona':
            continue
        if '_promedio' in c or c == 'mediciones_validas':
            cols_show.append(c)
            nombre = c.replace('_promedio', ' Prom').replace('mediciones_validas', 'N Validas')
            rename[c] = nombre.upper()

    tabla_data = resumen[cols_show].copy()
    tabla_data.columns = [rename.get(c, c) for c in cols_show]

    for c in tabla_data.columns:
        if tabla_data[c].dtype in [np.float64, np.float32]:
            tabla_data[c] = tabla_data[c].round(2)

    table = ax.table(cellText=tabla_data.values,
                     colLabels=tabla_data.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    for j in range(len(tabla_data.columns)):
        cell = table[0, j]
        cell.set_facecolor('#E17055')
        cell.set_text_props(color='white', fontweight='bold')

    for i in range(1, len(tabla_data) + 1):
        for j in range(len(tabla_data.columns)):
            cell = table[i, j]
            cell.set_facecolor('#3d1e1e' if i % 2 == 0 else '#2a1515')
            cell.set_text_props(color='#e0e0e0')

    ax.set_title('Resumen Consolidado - Signos Vitales en EJERCICIO',
                 fontsize=16, fontweight='bold', pad=20, color='#FDFEFE')
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, "tabla_ejercicio.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [OK] {os.path.basename(ruta)}")


# ── MAIN ──────────────────────────────────────────────────
def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # 1. Descubrir archivos
    archivos = descubrir_archivos(CARPETA_DATOS)
    if not archivos:
        print("  No se encontraron archivos de ejercicio.")
        return

    print(f"\n{'='*55}")
    print(f"  ARCHIVOS DE EJERCICIO ENCONTRADOS: {len(archivos)}")
    print(f"{'='*55}")

    # 2. Asignar personas consistentes con reposo
    archivo_persona = []
    extra_counter = 6
    for f in archivos:
        p = asignar_persona(f)
        if p is None:
            p = f'Persona {extra_counter}'
            extra_counter += 1
        archivo_persona.append((f, p))

    # Ordenar por numero de persona
    archivo_persona.sort(key=lambda x: int(x[1].split()[-1]))

    for f, p in archivo_persona:
        print(f"  {p}: {os.path.basename(f)}")

    # 3. Procesar cada archivo
    print(f"\n{'='*55}")
    print("  LIMPIEZA DE DATOS (EJERCICIO)")
    print(f"{'='*55}")

    resultados = []
    datos_limpios = {}
    outliers_por_persona = {}

    for archivo, persona in archivo_persona:
        df_limpio, total, descartadas, outliers_iqr = cargar_y_limpiar(archivo)
        stats = calcular_stats(df_limpio, persona)
        stats['total_original'] = total
        stats['descartadas'] = descartadas
        stats['outliers_iqr'] = outliers_iqr
        resultados.append(stats)
        datos_limpios[persona] = df_limpio
        outliers_por_persona[persona] = outliers_iqr

        pct = 100 * descartadas / max(total, 1)
        print(f"\n  {persona}: {os.path.basename(archivo)}")
        print(f"    Total registros : {total}")
        print(f"    Validos         : {len(df_limpio)}")
        print(f"    Descartados     : {descartadas} ({pct:.1f}%)")
        print(f"    Outliers IQR    : {outliers_iqr}")

    # 4. DataFrame consolidado
    resumen = pd.DataFrame(resultados)

    ruta_csv = os.path.join(CARPETA_SALIDA, "resumen_ejercicio_usuarios.csv")
    resumen.to_csv(ruta_csv, index=False, encoding='utf-8-sig')

    print(f"\n{'='*55}")
    print("  RESUMEN CONSOLIDADO (EJERCICIO)")
    print(f"{'='*55}")
    print(resumen.to_string(index=False))
    print(f"\n  [OK] CSV exportado: {ruta_csv}")

    # 5. Generar graficas
    print(f"\n{'='*55}")
    print("  GENERANDO GRAFICAS")
    print(f"{'='*55}")

    grafica_barras_comparativa(resumen, CARPETA_SALIDA)
    grafica_radar(resumen, CARPETA_SALIDA)
    grafica_boxplot(datos_limpios, CARPETA_SALIDA)
    grafica_heatmap(resumen, CARPETA_SALIDA)
    grafica_outliers(resumen, outliers_por_persona, CARPETA_SALIDA)
    grafica_desviaciones(resumen, CARPETA_SALIDA)
    tabla_resumen_imagen(resumen, CARPETA_SALIDA)

    print(f"\n{'='*55}")
    print(f"  PROCESO COMPLETADO")
    print(f"  Resultados en: {CARPETA_SALIDA}")
    print(f"{'='*55}\n")

    print("CRITERIOS DE LIMPIEZA APLICADOS (EJERCICIO):")
    print("-" * 45)
    print(f"  BPM      : {RANGOS['bpm'][0]} - {RANGOS['bpm'][1]}")
    print(f"  Temp     : {RANGOS['temp'][0]} - {RANGOS['temp'][1]} C")
    print(f"  PPG      : {RANGOS['ppg'][0]} - {RANGOS['ppg'][1]}")
    print(f"  ECG      : {RANGOS['ecg'][0]} - {RANGOS['ecg'][1]} mV")
    print(f"  Posture  : {RANGOS['posture'][0]} - {RANGOS['posture'][1]}")
    print(f"  IQR      : Factor 2.0x (tolerante a variabilidad)")


if __name__ == "__main__":
    main()
