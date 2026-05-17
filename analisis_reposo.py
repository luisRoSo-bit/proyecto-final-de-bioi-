"""
============================================================
ANÁLISIS DE DATOS BIOMÉDICOS EN REPOSO
============================================================
Procesa archivos CSV con "reposo" en el nombre.
Limpia outliers, calcula estadísticas y genera gráficas.
Usa etiquetas anónimas (Persona 1, Persona 2, etc.)
============================================================
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import warnings
warnings.filterwarnings('ignore')

# ── Configuración ──────────────────────────────────────────
CARPETA_DATOS = os.path.dirname(os.path.abspath(__file__))
CARPETA_SALIDA = os.path.join(CARPETA_DATOS, "resultados_reposo")

# Rangos fisiológicos normales en reposo
RANGOS = {
    'bpm':  (40, 120),
    'temp': (30, 42),
    'ppg':  (0, 500000),
    'ecg':  (-5, 5),
    'posture': (0, 20),
}

COLORES = ['#6C5CE7', '#00B894', '#E17055', '#0984E3', '#FDCB6E',
           '#E84393', '#00CEC9', '#2D3436', '#55EFC4', '#FAB1A0']

plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'grid.color': '#2a2a4a',
    'grid.alpha': 0.5,
    'font.family': 'sans-serif',
    'font.size': 11,
})


# ── 1. Descubrir archivos reposo ───────────────────────────
def descubrir_archivos(carpeta):
    todos = glob.glob(os.path.join(carpeta, "*.csv"))
    reposo = [f for f in todos if "reposo" in os.path.basename(f).lower()]
    print(f"\n{'='*55}")
    print(f"  ARCHIVOS DE REPOSO ENCONTRADOS: {len(reposo)}")
    print(f"{'='*55}")
    for i, f in enumerate(reposo, 1):
        print(f"  Persona {i}: {os.path.basename(f)}")
    return reposo


# ── 2. Cargar y limpiar datos ──────────────────────────────
def cargar_y_limpiar(archivo, persona_id):
    df = pd.read_csv(archivo)
    df.columns = df.columns.str.strip().str.lower()

    total_original = len(df)
    # Eliminar filas completamente vacías
    df.dropna(how='all', subset=[c for c in df.columns if c != 'time'], inplace=True)

    # Aplicar rangos fisiológicos
    mascara_valida = pd.Series(True, index=df.index)
    for col, (lo, hi) in RANGOS.items():
        if col in df.columns:
            fuera = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            mascara_valida &= ~fuera

    df_limpio = df[mascara_valida].copy()
    descartadas = total_original - len(df_limpio)

    print(f"\n  Persona {persona_id}: {os.path.basename(archivo)}")
    print(f"    Total registros : {total_original}")
    print(f"    Válidos         : {len(df_limpio)}")
    print(f"    Descartados     : {descartadas} ({100*descartadas/max(total_original,1):.1f}%)")

    return df_limpio, total_original, descartadas


# ── 3. Calcular estadísticas ──────────────────────────────
def calcular_stats(df, persona_id):
    metricas = ['bpm', 'temp', 'ppg', 'ecg', 'posture']
    fila = {'Persona': f'Persona {persona_id}'}
    for m in metricas:
        if m in df.columns:
            datos = df[m].dropna()
            fila[f'{m}_promedio'] = round(datos.mean(), 4)
            fila[f'{m}_std'] = round(datos.std(), 4)
            fila[f'{m}_min'] = round(datos.min(), 4)
            fila[f'{m}_max'] = round(datos.max(), 4)
            fila[f'{m}_mediana'] = round(datos.median(), 4)
    fila['mediciones_validas'] = len(df)
    return fila


# ── 4. Gráficas ───────────────────────────────────────────
def grafica_barras_comparativa(resumen, carpeta_salida):
    metricas = [('bpm_promedio', 'Frecuencia Cardíaca (BPM)', 'BPM'),
                ('temp_promedio', 'Temperatura (°C)', '°C'),
                ('ppg_promedio', 'PPG Promedio', 'u.a.'),
                ('ecg_promedio', 'ECG Promedio', 'mV')]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparación de Signos Vitales en Reposo por Persona',
                 fontsize=18, fontweight='bold', y=0.98, color='#FDFEFE')

    for ax, (col, titulo, unidad) in zip(axes.flat, metricas):
        if col not in resumen.columns:
            ax.set_visible(False)
            continue
        vals = resumen[col].values
        personas = resumen['Persona'].values
        bars = ax.bar(personas, vals, color=COLORES[:len(vals)],
                      edgecolor='white', linewidth=0.5, width=0.6)
        ax.set_title(titulo, fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel(unidad, fontsize=10)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                    f'{v:.1f}', ha='center', va='bottom', fontsize=9,
                    fontweight='bold', color='#FDFEFE')
        ax.tick_params(axis='x', rotation=30)

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    ruta = os.path.join(carpeta_salida, "comparacion_signos_vitales.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ {ruta}")


def grafica_radar(resumen, carpeta_salida):
    metricas_radar = ['bpm_promedio', 'temp_promedio', 'ppg_promedio', 'posture_promedio']
    etiquetas = ['BPM', 'Temp', 'PPG', 'Postura']

    disponibles = [m for m in metricas_radar if m in resumen.columns]
    etiquetas_disp = [etiquetas[metricas_radar.index(m)] for m in disponibles]
    if len(disponibles) < 3:
        return

    datos_norm = resumen[disponibles].copy()
    for c in disponibles:
        rng = datos_norm[c].max() - datos_norm[c].min()
        if rng > 0:
            datos_norm[c] = (datos_norm[c] - datos_norm[c].min()) / rng
        else:
            datos_norm[c] = 0.5

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
    ax.set_xticklabels(etiquetas_disp, fontsize=12, fontweight='bold')
    ax.set_title('Perfil Normalizado por Persona', fontsize=16,
                 fontweight='bold', pad=20, color='#FDFEFE')
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=9)
    ax.grid(color='#4a4a6a', alpha=0.4)
    ax.tick_params(colors='#e0e0e0')

    ruta = os.path.join(carpeta_salida, "radar_personas.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ {ruta}")


def grafica_boxplot(datos_limpios, carpeta_salida):
    metricas = ['bpm', 'temp']
    titulos = {'bpm': 'Frecuencia Cardíaca (BPM)', 'temp': 'Temperatura (°C)'}

    for metrica in metricas:
        todas = []
        labels = []
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

        ax.set_title(f'Distribución de {titulos[metrica]} en Reposo',
                     fontsize=15, fontweight='bold', color='#FDFEFE')
        ax.set_ylabel(titulos[metrica], fontsize=11)
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        ax.tick_params(axis='x', rotation=30)

        ruta = os.path.join(carpeta_salida, f"boxplot_{metrica}.png")
        fig.savefig(ruta, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"  ✔ {ruta}")


def grafica_heatmap(resumen, carpeta_salida):
    cols = [c for c in resumen.columns if '_promedio' in c]
    if not cols:
        return

    datos = resumen[cols].copy()
    datos.columns = [c.replace('_promedio', '').upper() for c in cols]
    datos.index = resumen['Persona']

    # Normalizar por columna
    datos_norm = datos.copy()
    for c in datos_norm.columns:
        rng = datos_norm[c].max() - datos_norm[c].min()
        if rng > 0:
            datos_norm[c] = (datos_norm[c] - datos_norm[c].min()) / rng

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(datos_norm.values, cmap='viridis', aspect='auto')

    ax.set_xticks(range(len(datos_norm.columns)))
    ax.set_xticklabels(datos_norm.columns, fontsize=11, fontweight='bold')
    ax.set_yticks(range(len(datos_norm.index)))
    ax.set_yticklabels(datos_norm.index, fontsize=11)

    for i in range(len(datos_norm.index)):
        for j in range(len(datos_norm.columns)):
            val_real = datos.iloc[i, j]
            ax.text(j, i, f'{val_real:.1f}', ha='center', va='center',
                    fontsize=9, fontweight='bold',
                    color='white' if datos_norm.iloc[i, j] < 0.5 else 'black')

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Valor normalizado', color='#e0e0e0')
    cbar.ax.yaxis.set_tick_params(color='#e0e0e0')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#e0e0e0')

    ax.set_title('Mapa de Calor — Promedios por Persona',
                 fontsize=15, fontweight='bold', pad=15, color='#FDFEFE')
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, "heatmap_promedios.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ {ruta}")


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
        ax.bar(x + offset, resumen[col], ancho, label=nombre,
               color=COLORES[i % len(COLORES)], edgecolor='white', linewidth=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(resumen['Persona'], rotation=30)
    ax.set_title('Desviación Estándar por Métrica y Persona',
                 fontsize=15, fontweight='bold', color='#FDFEFE')
    ax.set_ylabel('Desviación Estándar')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, "desviacion_estandar.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ {ruta}")


def grafica_mediciones_validas(resumen, total_por_persona, carpeta_salida):
    fig, ax = plt.subplots(figsize=(10, 6))
    personas = resumen['Persona'].values
    validas = resumen['mediciones_validas'].values
    descartadas = [total_por_persona[p] - v for p, v in
                   zip(personas, validas)]

    x = np.arange(len(personas))
    w = 0.4
    b1 = ax.bar(x - w/2, validas, w, label='Válidas', color='#00B894',
                edgecolor='white', linewidth=0.5)
    b2 = ax.bar(x + w/2, descartadas, w, label='Descartadas', color='#E17055',
                edgecolor='white', linewidth=0.5)

    for bar, v in zip(b1, validas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                str(v), ha='center', va='bottom', fontsize=9, color='#FDFEFE')
    for bar, v in zip(b2, descartadas):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                str(v), ha='center', va='bottom', fontsize=9, color='#FDFEFE')

    ax.set_xticks(x)
    ax.set_xticklabels(personas, rotation=30)
    ax.set_title('Mediciones Válidas vs Descartadas',
                 fontsize=15, fontweight='bold', color='#FDFEFE')
    ax.set_ylabel('Cantidad de registros')
    ax.legend(fontsize=11)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, "validas_vs_descartadas.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ {ruta}")


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
            nombre = c.replace('_promedio', ' Prom').replace('mediciones_validas', 'N Válidas')
            rename[c] = nombre.upper()

    tabla_data = resumen[cols_show].copy()
    tabla_data.columns = [rename.get(c, c) for c in cols_show]

    # Redondear
    for c in tabla_data.columns:
        if tabla_data[c].dtype in [np.float64, np.float32]:
            tabla_data[c] = tabla_data[c].round(2)

    table = ax.table(cellText=tabla_data.values,
                     colLabels=tabla_data.columns,
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)

    # Estilo cabecera
    for j in range(len(tabla_data.columns)):
        cell = table[0, j]
        cell.set_facecolor('#6C5CE7')
        cell.set_text_props(color='white', fontweight='bold')

    # Estilo filas
    for i in range(1, len(tabla_data) + 1):
        for j in range(len(tabla_data.columns)):
            cell = table[i, j]
            cell.set_facecolor('#2d2d5e' if i % 2 == 0 else '#1e1e4a')
            cell.set_text_props(color='#e0e0e0')

    ax.set_title('Resumen Consolidado de Signos Vitales en Reposo',
                 fontsize=16, fontweight='bold', pad=20, color='#FDFEFE')
    plt.tight_layout()
    ruta = os.path.join(carpeta_salida, "tabla_resumen.png")
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✔ {ruta}")


# ── MAIN ──────────────────────────────────────────────────
def main():
    os.makedirs(CARPETA_SALIDA, exist_ok=True)

    # 1. Descubrir archivos
    archivos = descubrir_archivos(CARPETA_DATOS)
    if not archivos:
        print("  ⚠ No se encontraron archivos con 'reposo' en el nombre.")
        return

    # 2. Procesar cada archivo
    print(f"\n{'='*55}")
    print("  LIMPIEZA DE DATOS")
    print(f"{'='*55}")

    resultados = []
    datos_limpios = {}
    total_por_persona = {}

    for i, archivo in enumerate(sorted(archivos), 1):
        df_limpio, total, descartadas = cargar_y_limpiar(archivo, i)
        stats = calcular_stats(df_limpio, i)
        stats['total_original'] = total
        stats['descartadas'] = descartadas
        resultados.append(stats)
        datos_limpios[f'Persona {i}'] = df_limpio
        total_por_persona[f'Persona {i}'] = total

    # 3. DataFrame consolidado
    resumen = pd.DataFrame(resultados)

    # Exportar CSV
    ruta_csv = os.path.join(CARPETA_SALIDA, "resumen_reposo_usuarios.csv")
    resumen.to_csv(ruta_csv, index=False, encoding='utf-8-sig')

    print(f"\n{'='*55}")
    print("  RESUMEN CONSOLIDADO")
    print(f"{'='*55}")
    print(resumen.to_string(index=False))
    print(f"\n  ✔ CSV exportado: {ruta_csv}")

    # 4. Generar todas las gráficas
    print(f"\n{'='*55}")
    print("  GENERANDO GRÁFICAS")
    print(f"{'='*55}")

    grafica_barras_comparativa(resumen, CARPETA_SALIDA)
    grafica_radar(resumen, CARPETA_SALIDA)
    grafica_boxplot(datos_limpios, CARPETA_SALIDA)
    grafica_heatmap(resumen, CARPETA_SALIDA)
    grafica_desviaciones(resumen, CARPETA_SALIDA)
    grafica_mediciones_validas(resumen, total_por_persona, CARPETA_SALIDA)
    tabla_resumen_imagen(resumen, CARPETA_SALIDA)

    print(f"\n{'='*55}")
    print(f"  ✅ PROCESO COMPLETADO")
    print(f"  📁 Resultados en: {CARPETA_SALIDA}")
    print(f"{'='*55}\n")

    # Criterios de limpieza
    print("CRITERIOS DE LIMPIEZA APLICADOS:")
    print("-" * 40)
    print(f"  BPM     : {RANGOS['bpm'][0]} – {RANGOS['bpm'][1]}")
    print(f"  Temp    : {RANGOS['temp'][0]} – {RANGOS['temp'][1]} °C")
    print(f"  PPG     : {RANGOS['ppg'][0]} – {RANGOS['ppg'][1]}")
    print(f"  ECG     : {RANGOS['ecg'][0]} – {RANGOS['ecg'][1]} mV")
    print(f"  Posture : {RANGOS['posture'][0]} – {RANGOS['posture'][1]}")


if __name__ == "__main__":
    main()
