# Sourced by the scripts in this directory. The local servers listen only on Unix
# sockets in CAPX_SOCKET_DIR (capx/utils/unix_http.py); a socket is named by its file
# name there, e.g. model.sock or sam3.sock.

export CAPX_SOCKET_DIR="${CAPX_SOCKET_DIR:-${HOME}/.capx/sockets}"

# Succeeds if a server answers HTTP on socket $1, at path $2 (default /), with any status.
sock_up() {
    curl -s -o /dev/null --max-time 3 --unix-socket "${CAPX_SOCKET_DIR}/$1" "http://localhost${2:-/}"
}

# Succeeds if a server answers socket $1, path $2, with a 2xx status.
sock_ok() {
    curl -sf -o /dev/null --max-time 3 --unix-socket "${CAPX_SOCKET_DIR}/$1" "http://localhost${2:-/}"
}
