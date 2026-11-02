#!/usr/bin/env bash
# Adapted from PINNs/ssh_tunnel.sh: argv arrays, forwarding and keepalive options.
# OpenSSH control sockets isolate start/status/stop to this app's own tunnel.
set -euo pipefail
command=${1:-help}
[[ $# -gt 0 ]] && shift
ssh_user=""; ssh_host=""; ssh_key=""; ssh_port=22; remote_port=9000; local_port=9000
while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
    case "$1" in
        -u|--user) ssh_user=$2 ;;
        -H|--host) ssh_host=$2 ;;
        -k|--key) ssh_key=$2 ;;
        -P|--ssh-port) ssh_port=$2 ;;
        -p|--port) remote_port=$2 ;;
        -l|--local) local_port=$2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift 2
done
for port in "$ssh_port" "$remote_port" "$local_port"; do
    [[ $port =~ ^[0-9]+$ ]] && (( port > 0 && port <= 65535 )) || { echo "Invalid port" >&2; exit 2; }
done
state_dir="${TMPDIR:-/tmp}/lidarpoc-${UID}"
control_socket="${state_dir}/tunnel-${local_port}"
case "$command" in
    start)
        [[ -n $ssh_host && $ssh_host != -* && $ssh_user != -* ]] || { echo "Supply -H host and optionally -u user" >&2; exit 2; }
        mkdir -p "$state_dir"
        chmod 700 "$state_dir"
        options=(-p "$ssh_port" -o ConnectTimeout=10 -o BatchMode=yes
                 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes)
        [[ -z $ssh_key ]] || options+=(-i "$ssh_key")
        [[ -z $ssh_user ]] || options+=(-l "$ssh_user")
        ssh -M -S "$control_socket" -fNT "${options[@]}" \
            -L "127.0.0.2:${local_port}:127.0.0.1:${remote_port}" "$ssh_host"
        echo "API tunnel ready: http://127.0.0.2:${local_port}"
        ;;
    status) ssh -S "$control_socket" -O check localhost ;;
    stop) ssh -S "$control_socket" -O exit localhost ;;
    *) echo "Usage: bash ssh_tunnel.sh {start|status|stop} [-H host] [-u user] [-k key] [-P ssh_port] [-p remote_port] [-l local_port]" ;;
esac
