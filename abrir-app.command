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

# shellcheck disable=SC1091
source .venv/bin/activate

if ! python -c "import hdrpipe" >/dev/null 2>&1; then
    echo "Instalando dependencias..."
    pip install --quiet --upgrade pip
    if ! pip install --quiet -e .; then
        echo ""
        echo "${ROJO}Fallo la instalacion.${FIN} Arriba tienes el motivo."
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

# La direccion la imprime `hdrpipe serve`; aqui solo se recuerda como salir.
echo "${GRIS}Para cerrar la app: cierra esta ventana o pulsa Control+C${FIN}"
echo ""

# Se espera a que el servidor responda antes de abrir el navegador, para no
# encontrarse un "no se puede conectar" en la primera carga.
(
    for _ in $(seq 1 40); do
        if curl -s -o /dev/null "http://127.0.0.1:${PUERTO}/"; then
            open "http://127.0.0.1:${PUERTO}"
            exit 0
        fi
        sleep 0.5
    done
) &

exec hdrpipe serve --puerto "$PUERTO"
