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

MINIMO="3.10"

sirve() {
    [ -x "$1" ] && "$1" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' \
        >/dev/null 2>&1
}

# `python3` a secas suele apuntar al Python de sistema de Apple, que se queda
# anos atras. Se buscan primero las versiones modernas por nombre y en las
# rutas donde dejan el interprete el instalador de python.org y Homebrew.
buscar_python() {
    local candidatos=(python3.14 python3.13 python3.12 python3.11 python3.10)
    local version ruta candidato
    for version in 3.14 3.13 3.12 3.11 3.10; do
        candidatos+=("/Library/Frameworks/Python.framework/Versions/${version}/bin/python3")
        candidatos+=("/opt/homebrew/bin/python${version}")
        candidatos+=("/usr/local/bin/python${version}")
    done
    candidatos+=(python3)

    for candidato in "${candidatos[@]}"; do
        ruta="$(command -v "$candidato" 2>/dev/null)"
        [ -z "$ruta" ] && ruta="$candidato"
        if sirve "$ruta"; then
            echo "$ruta"
            return 0
        fi
    done
    return 1
}

PY3="$(buscar_python)"

if [ -z "$PY3" ]; then
    ACTUAL="$(python3 -V 2>&1 || echo 'ninguno')"
    echo "${ROJO}Hace falta Python ${MINIMO} o superior.${FIN}"
    echo "  Lo que hay ahora en este Mac: ${ACTUAL}"
    echo "  ${GRIS}(es el Python que trae macOS de serie, y se queda anticuado)${FIN}"
    echo ""
    echo "Se instala en dos minutos y es gratis:"
    echo "  1. Entra en  ${VERDE}https://www.python.org/downloads/macos/${FIN}"
    echo "  2. Descarga el instalador mas reciente (macOS 64-bit universal2)"
    echo "  3. Abrelo y dale a Continuar hasta el final"
    echo "  4. Vuelve a hacer doble clic en esta app"
    echo ""
    echo "${GRIS}No sustituye al Python de macOS, se instala al lado. No rompe nada.${FIN}"
    echo ""
    read -r -p "Pulsa Enter para cerrar."
    exit 1
fi

# --- Entorno ----------------------------------------------------------------

# Un entorno creado con un Python antiguo se queda con esa version para
# siempre. Si estaba hecho con el Python de sistema, se rehace.
if [ -d .venv ] && ! sirve "$(pwd)/.venv/bin/python"; then
    echo "El entorno se creo con una version antigua de Python. Rehaciendolo..."
    rm -rf .venv
fi

if [ ! -d .venv ]; then
    echo "Primera vez: preparando el entorno con $("$PY3" -V 2>&1)..."
    echo ""
    "$PY3" -m venv .venv || {
        echo "${ROJO}No se pudo crear el entorno.${FIN}"
        read -r -p "Pulsa Enter para cerrar."
        exit 1
    }
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
