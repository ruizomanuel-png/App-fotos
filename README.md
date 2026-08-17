# Editor HDR inmobiliario

Convierte carpetas de brackets RAW en JPEG terminados, sin pasar por Lightroom.
Sustituye el ciclo de *combinar HDR → editar → pintar máscaras para bajar
ventanas y subir sombras → borrar lo que sobra*, que es exactamente lo repetitivo.

Corre entero en tu MacBook. Sin servidor, sin cuentas, sin coste.

---

## Instalación (macOS)

Una sola vez, en la app **Terminal**:

```bash
cd ~/Documents
git clone https://github.com/ruizomanuel-png/App-fotos.git
cd App-fotos
git checkout claude/hdr-real-estate-photo-app-wu0b0c
```

A partir de ahí, **doble clic en `abrir-app.command`** dentro de la carpeta.
La primera vez prepara el entorno solo (un par de minutos) y luego abre el
navegador. Las siguientes, arranca en segundos.

Si macOS avisa de que no puede verificar el desarrollador: clic derecho sobre
el archivo → Abrir → Abrir. Solo hace falta la primera vez.

Requiere Python 3.10 o superior. Si no lo tienes, el lanzador te lo dice y te
indica cómo instalarlo (`xcode-select --install`, gratis).

### Borrado de objetos (opcional)

Para activar el borrado automático de personas, mascotas y coches (opcional,
descarga ~2 GB y usa la GPU del M3 vía MPS):

```bash
pip install -e ".[ai]"
```

Sin este extra todo lo demás funciona igual; simplemente no borra objetos y lo
anota en el informe.

## Uso

**App web** — la vía normal: doble clic en `abrir-app.command`, o desde la
terminal con el entorno activado:

```bash
source .venv/bin/activate
hdrpipe serve      # abre http://127.0.0.1:8000
```

Pega la ruta de la carpeta con los RAW y pulsa Procesar. Para lotes grandes es
mejor pegar la ruta que arrastrar los archivos: lee la carpeta en su sitio, sin
copiar gigas de un lado a otro.

**Línea de comandos** — para automatizar:

```bash
hdrpipe run ~/Fotos/CalleMayor12 ~/Entregas/CalleMayor12
hdrpipe run ~/Fotos/CalleMayor12 /tmp/prueba --rapido --limite 3   # ver el look sin esperar
```

### Qué sale

```
Entregas/CalleMayor12/
    salon.jpg
    cocina.jpg
    revisar/
        dormitorio2.jpg     <- no pasó el control de calidad
    informe.json            <- qué se hizo en cada foto y por qué
```

Las fotos dudosas no se descartan en silencio: van a `revisar/`, fuera del ZIP
principal, con el motivo anotado en el informe. Perder una toma sin enterarte
sale más caro que mirar una carpeta.

## Calibrar tu look

Este es el paso que más cambia el resultado. Los valores de fábrica son un
punto de partida razonable; tus pares antes/después los convierten en *tu* look.

```
calibracion/
    raw/      los brackets originales (se agrupan solos)
    final/    tus JPEG exportados de Lightroom, mismo nombre que la escena
```

```bash
hdrpipe calibrate ~/calibracion
```

Con 20–30 pares variados (salones, baños, contraluces, exteriores) mide cuatro
cosas y las escribe en `config/learned.yaml`:

| Qué aprende | Dónde acaba |
|---|---|
| Cuánto de clara dejas una foto | `tone.target_midtone` |
| Tu curva de tono, punto por punto | `grade.curve_lut` |
| Cuánta calidez conservas | `white_balance.keep_warmth` |
| Cuánto color metes | `grade.saturation` |

No entrena ningún modelo. Revela tus RAW con los parámetros actuales, compara
el resultado con tu export, corrige y repite unas cuantas vueltas. Se queda con
la mejor iteración medida, así que calibrar nunca deja el look peor que antes.
Borra `config/learned.yaml` para volver a los valores de fábrica.

## Qué hace, en orden

| Etapa | Qué resuelve |
|---|---|
| **Revelado** | Sony ARW y Apple ProRAW DNG a radiancia lineal, con el mismo balance de blancos en las tres tomas |
| **Agrupación** | Separa la carpeta en escenas por hora de disparo |
| **Alineación** | Corrige el desplazamiento residual del trípode, que produce bordes dobles en los marcos |
| **Fusión** | Combina las tres exposiciones en un mapa de radiancia sin techo |
| **Balance de blancos** | Neutraliza la dominante y trata por separado el interior cálido y la ventana fría |
| **Window pull** | Baja las ventanas con máscara automática y guided filter, sin halos |
| **Sombras** | Levanta las zonas oscuras con la misma técnica |
| **Mapeo de tono** | Curva fílmica con desaturación de altas luces |
| **Borrado** | Personas, mascotas y coches (requiere el extra `ai`) |
| **Geometría** | Verticales rectas, horizonte, distorsión de lente y reencuadre |
| **Acabado** | Tu curva, vibrance, césped y agua |
| **Consistencia** | Acerca cada foto al consenso del lote |
| **Detalle** | Ruido de color y enfoque limitado a los bordes |
| **Control** | Nitidez, recorte y quemados; lo dudoso va a `revisar/` |

### Decisiones que conviene conocer

**La fusión es física, no cosmética.** Se usan los tiempos de exposición del
EXIF para llevar las tres tomas a la misma escala de radiancia. El resultado no
tiene techo: el sofá vale 0,03 y la ventana puede valer 60. Por eso el window
pull no se inventa nada — baja información que ya estaba medida. Si el EXIF
viene incompleto, las escalas se deducen de los propios píxeles.

**Las máscaras van en paradas, no en percentiles.** Una ventana se define como
lo que está entre 2 y 3,5 paradas por encima del nivel del interior. Un umbral
por percentil describe cuánta superficie se selecciona, y eso no sirve: una
ventana puede ocupar el 2% de una foto y el 30% de la siguiente.

**Las correcciones se calculan por foto.** Los valores del YAML son topes, no
cantidades fijas. Una ventana con el jardín apenas quemado recibe menos paradas
que una a contraluz directo.

**El enderezado reencuadra dentro del cuadrilátero corregido**, no contra el
lienzo de partida. Enderezar es una rotación de cámara, y una rotación mueve el
encuadre entero: recortar contra el lienzo original tira contenido válido. En
una escena de prueba con 6,6° de inclinación, la diferencia era 34% de pérdida
frente al 5,8% actual.

## Configuración

Todo lo ajustable está en [`config/default.yaml`](config/default.yaml), comentado
parámetro a parámetro. `config/learned.yaml` (que escribe `calibrate`) se aplica
encima.

Los ajustes que más se tocan:

```yaml
tone:
  target_midtone: 0.46        # cómo de clara sale una foto
  window_pull:
    ev: 2.4                   # tope de recuperación de ventanas
batch:
  consistency: 0.45           # 0 = cada foto por su cuenta, 1 = todas iguales
output:
  max_long_side: 3600
  quality: 94
retention:
  days: 7                     # los lotes se borran solos pasada una semana
```

### Distorsión de lente

Sin perfil, la distorsión no se corrige (el ProRAW del iPhone ya viene
corregido de fábrica). Para un objetivo Sony se calibra una vez, a ojo sobre
una foto con líneas rectas cerca del borde:

```yaml
geometry:
  lens_profiles:
    "FE 16-35mm F2.8 GM":     # se busca dentro del campo LensModel del EXIF
      k1: -0.08
      k2: 0.01
```

## Desarrollo

```bash
pip install -e ".[dev]"
pytest
```

Las pruebas trabajan sobre interiores sintéticos (`tests/scenes.py`) que
reproducen el caso difícil: penumbra, rincón oscuro y ventana cinco paradas por
encima.

## Estado y límites conocidos

Lo que **no** está en esta versión, por decisión:

- **Reemplazo de cielo.** Es la parte más frágil; primero conviene tener el
  resto asentado.
- **Borrado manual con clic.** Cables, enchufes, mangueras y reflejos del
  trípode en espejos no se detectan solos de forma fiable — no hay modelo que
  sepa qué "sobra" en tu foto. El borrado automático cubre personas, mascotas y
  coches; el resto necesita el editor de la v2.
- **Integración con Google Drive.** De momento, ruta local o arrastrar y soltar.
- **Sliders y reedición.** Sube y descarga.

Lo que hay que verificar contra material real, y no se ha podido comprobar aquí:

- **El revelado de ARW y DNG.** El código usa LibRaw por la ruta estándar, pero
  las pruebas corren sobre TIFF sintéticos: no se pueden fabricar archivos RAW
  válidos. La primera carpeta real es la prueba de verdad.
- **El borrado de objetos.** El extra `ai` no está instalado en el entorno donde
  se desarrolló, así que esa ruta está escrita pero sin ejecutar. La degradación
  cuando falta el modelo sí está comprobada.
- **Los valores por defecto de tono y color.** Están puestos a criterio sobre
  escenas sintéticas. Calibra con tus pares antes de juzgar el look.
