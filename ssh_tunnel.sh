#!/usr/bin/env bash
# Adapted from PINNs/ssh_tunnel.sh; reuse ControlMaster connections when available.
set -euo pipefail
command=${1:-help}
[[ $# -gt 0 ]] && shift
ssh_user=""; ssh_host=""; ssh_key=""; ssh_port=22; remote_port=9000; local_port=9000; existing_socket=""
while [[ $# -gt 0 ]]; do
    [[ $# -ge 2 ]] || { echo "Missing value for $1" >&2; exit 2; }
    case "$1" in
        -u|--user) ssh_user=$2 ;;
        -H|--host) ssh_host=$2 ;;
        -k|--key) ssh_key=$2 ;;
        -P|--ssh-port) ssh_port=$2 ;;
        -p|--port) remote_port=$2 ;;
        -l|--local) local_port=$2 ;;
        -S|--control-socket) existing_socket=$2 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
    shift 2
done
for port in "$ssh_port" "$remote_port" "$local_port"; do
    [[ $port =~ ^[0-9]+$ ]] && (( port > 0 && port <= 65535 )) || { echo "Invalid port" >&2; exit 2; }
done
state_dir="${TMPDIR:-/tmp}/lidarpoc-${UID}"
state_file="${state_dir}/forward-${local_port}.state"
app_socket="${state_dir}/tunnel-${local_port}"
forward="127.0.0.2:${local_port}:127.0.0.1:${remote_port}"
saved=()
[[ ! -f $state_file ]] || mapfile -t saved < "$state_file"
control_socket=${saved[0]:-${existing_socket:-$app_socket}}
master_alive() { ssh -S "$1" -O check localhost >/dev/null 2>&1; }
case "$command" in
    start)
        [[ -n $ssh_host && $ssh_host != -* && $ssh_user != -* ]] || { echo "Supply -H host and optionally -u user" >&2; exit 2; }
        mkdir -p "$state_dir"
        chmod 700 "$state_dir"
        identity="${ssh_user}@${ssh_host}:${ssh_port}"
        if [[ ${#saved[@]} -gt 0 ]] && master_alive "${saved[0]}"; then
            [[ ${saved[1]} == "$forward" && ${saved[3]} == "$identity" ]] || {
                echo "A different tunnel is registered on this local port; stop it first" >&2; exit 1;
            }
            echo "Reusing active API tunnel: http://127.0.0.2:${local_port}"
            exit 0
        fi
        options=(-p "$ssh_port" -o ConnectTimeout=10 -o BatchMode=yes
                 -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -o ExitOnForwardFailure=yes)
        [[ -z $ssh_key ]] || options+=(-i "$ssh_key")
        [[ -z $ssh_user ]] || options+=(-l "$ssh_user")
        [[ -z $existing_socket ]] || options+=(-S "$existing_socket")
        configured=$(ssh -G "${options[@]}" "$ssh_host" 2>/dev/null |
                     awk '$1 == "controlpath" {sub(/^[^[:space:]]+[[:space:]]+/, ""); print}')
        owned=0
        if [[ -n $configured && $configured != none ]] && master_alive "$configured"; then
            control_socket=$configured
            ssh -S "$control_socket" -O forward -L "$forward" "$ssh_host"
            echo "Reused existing SSH connection: $control_socket"
        elif [[ -n $existing_socket ]]; then
            echo "No existing SSH master at $existing_socket; no new connection was opened" >&2
            exit 1
        else
            control_socket=$app_socket
            owned=1
            ssh -M -S "$control_socket" -fNT "${options[@]}" -L "$forward" "$ssh_host"
            echo "Opened SSH connection: $control_socket"
        fi
        printf '%s\n' "$control_socket" "$forward" "$owned" "$identity" > "$state_file"
        echo "API tunnel ready: http://127.0.0.2:${local_port}"
        ;;
    status) ssh -S "$control_socket" -O check localhost ;;
    stop)
        ssh -S "$control_socket" -O cancel -L "${saved[1]:-$forward}" localhost
        if [[ ${saved[2]:-0} == 1 ]]; then
            ssh -S "$control_socket" -O exit localhost
        fi
        rm -f "$state_file"
        echo "Closed API forwarding; shared SSH connections are preserved"
        ;;
    *) echo "Usage: bash ssh_tunnel.sh {start|status|stop} [-H host] [-u user] [-k key] [-P ssh_port] [-p remote_port] [-l local_port] [-S existing_control_socket]" ;;
esac
