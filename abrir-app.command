#!/bin/bash
#
# Lanzador para macOS. Doble clic en Finder y listo.
#
# La primera vez prepara el entorno (tarda un par de minutos); a partir de
# ahi arranca en segundos. Para cerrar la app, cierra esta ventana o pulsa
# Control+C.

cd "$(dirname "$0")" || exit 1

PUERTO=8000
VERDE=$'\033[32m'; ROJO=$'\033[31m'; GRIS=$'\033[90m'; FIN=$'\033[0m'

echo ""
echo "  Editor HDR inmobiliario"
echo "  ${GRIS}$(pwd)${FIN}"
echo ""

# --- Python -----------------------------------------------------------------

if ! command -v python3 >/dev/null 2>&1; then
    echo "${ROJO}No hay Python instalado.${FIN}"
    echo ""
    echo "Instalalo con las herramientas de desarrollo de Apple (son gratis):"
    echo "  1. Abre la app Terminal"
    echo "  2. Escribe:  xcode-select --install"
    echo "  3. Acepta y espera a que termine"
    echo "  4. Vuelve a hacer doble clic aqui"
    echo ""
    read -r -p "Pulsa Enter para cerrar."
    exit 1
fi

# --- Entorno ----------------------------------------------------------------

if [ ! -d .venv ]; then
    echo "Primera vez: preparando el entorno. Esto tarda un par de minutos..."
    echo ""
    python3 -m venv .venv || { echo "${ROJO}No se pudo crear el entorno.${FIN}"; read -r; exit 1; }
fi

# Todo se invoca por ruta absoluta, sin depender del PATH ni de `activate`.
PYTHON="$(pwd)/.venv/bin/python"
HDRPIPE="$(pwd)/.venv/bin/hdrpipe"

# Se comprueba que exista el ejecutable, no que el paquete se pueda importar.
# El import enganaba: estamos dentro de la carpeta del proyecto, donde hay un
# directorio llamado `hdrpipe/`, asi que `import hdrpipe` funciona aunque no se
# haya instalado nada -- y luego el comando no existe.
if [ ! -x "$HDRPIPE" ]; then
    echo "Instalando dependencias. La primera vez tarda un par de minutos..."
    "$PYTHON" -m pip install --quiet --upgrade pip
    if ! "$PYTHON" -m pip install --quiet -e .; then
        echo ""
        echo "${ROJO}Fallo la instalacion.${FIN} El motivo esta en las lineas de arriba."
        read -r -p "Pulsa Enter para cerrar."
        exit 1
    fi
    if [ ! -x "$HDRPIPE" ]; then
        echo ""
        echo "${ROJO}La instalacion termino pero falta el comando hdrpipe.${FIN}"
        echo "Prueba a borrar la carpeta .venv y volver a abrir la app:"
        echo "  rm -rf '$(pwd)/.venv'"
        read -r -p "Pulsa Enter para cerrar."
        exit 1
    fi
    echo "${VERDE}Listo.${FIN}"
    echo ""
fi

# --- Puerto libre -----------------------------------------------------------

# Si ya hay una copia corriendo, no se arranca otra: se abre la que hay.
if lsof -nP -iTCP:"$PUERTO" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "La app ya estaba abierta. Llevandote a ella..."
    open "http://127.0.0.1:${PUERTO}"
    echo ""
    echo "${GRIS}Puedes cerrar esta ventana.${FIN}"
    sleep 3
    exit 0
fi

# --- Arranque ---------------------------------------------------------------

DIRECCION="http://127.0.0.1:${PUERTO}"

echo "  Direccion de la app:  ${VERDE}${DIRECCION}${FIN}"
echo "  ${GRIS}Si el navegador no se abre solo, copia esa direccion en Safari o Chrome.${FIN}"
echo "  ${GRIS}Para cerrar la app: cierra esta ventana o pulsa Control+C${FIN}"
echo ""
echo -n "Arrancando"

# Se espera a que el servidor responda antes de abrir el navegador, para no
# encontrarse un "no se puede conectar" en la primera carga. El primer arranque
# carga OpenCV y LibRaw, que en frio tardan lo suyo, de ahi el margen amplio.
(
    for _ in $(seq 1 120); do
        if curl -s -o /dev/null --max-time 2 "${DIRECCION}/"; then
            echo ""
            echo "${VERDE}Lista.${FIN} Abriendo el navegador..."
            open "${DIRECCION}" || {
                echo "${ROJO}No se pudo abrir el navegador solo.${FIN}"
                echo "Entra tu a mano en:  ${DIRECCION}"
            }
            exit 0
        fi
        echo -n "."
        sleep 1
    done
    echo ""
    echo "${ROJO}El servidor no ha respondido en dos minutos.${FIN}"
    echo "Mira si hay algun error mas abajo en esta misma ventana."
) &
ESPERA=$!

# Sin `exec`: si el arranque falla, el shell sigue vivo para poder contarlo.
# Con `exec`, la ventana de Terminal se cerraria de golpe sin mostrar nada.
"$HDRPIPE" serve --puerto "$PUERTO"
CODIGO=$?

kill "$ESPERA" 2>/dev/null

# Control+C devuelve 130: eso es un cierre normal, no un fallo.
if [ "$CODIGO" -ne 0 ] && [ "$CODIGO" -ne 130 ]; then
    echo ""
    echo "${ROJO}La app se ha cerrado con un error (codigo $CODIGO).${FIN}"
    echo "El motivo esta en las lineas de arriba."
    echo ""
    read -r -p "Pulsa Enter para cerrar esta ventana."
fi
