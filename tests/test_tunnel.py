import json
import os
from pathlib import Path
import subprocess
import tomllib
import pytest


@pytest.fixture
def ssh(tmp_path):
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / 'ssh'
    arguments = tmp_path / 'ssh-arguments.jsonl'
    binary.write_text('''#!/usr/bin/python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
with open(os.environ['SSH_ARGUMENTS'], 'a') as output:
    output.write(json.dumps(args) + '\\n')
socket = args[args.index('-S') + 1] if '-S' in args else os.environ.get('SSH_CONFIG_SOCKET', 'none')
alive = Path(socket + '.alive')
if '-G' in args:
    print('controlpath ' + socket)
elif '-O' in args:
    command = args[args.index('-O') + 1]
    if command == 'check': sys.exit(0 if alive.exists() else 1)
    if command == 'exit': alive.unlink()
elif '-M' in args:
    alive.touch()
''')
    binary.chmod(0o755)
    environment = {**os.environ, 'PATH': f'{tmp_path}{os.pathsep}{os.environ["PATH"]}',
                   'TMPDIR': str(tmp_path), 'SSH_ARGUMENTS': str(arguments)}

    def execute(*args, configured='none'):
        return subprocess.run(['bash', str(root / 'ssh_tunnel.sh'), *args], capture_output=True, text=True,
                              env={**environment, 'SSH_CONFIG_SOCKET': configured})

    def calls():
        return [json.loads(line) for line in arguments.read_text().splitlines()]

    return execute, calls


def test_ssh_key_and_port_9000_use_a_separate_client_loopback_address(tmp_path, ssh):
    execute, calls = ssh
    result = execute('start', '-H', 'example.amazonaws.com', '-u', 'ubuntu', '-k', '/client/key.pem')
    assert result.returncode == 0, result.stderr
    args = calls()[-1]
    assert args[args.index('-L') + 1] == '127.0.0.2:9000:127.0.0.1:9000'
    assert args[args.index('-i') + 1] == '/client/key.pem'
    assert args[args.index('-p') + 1] == '22'
    assert args[-1] == 'example.amazonaws.com'
    result = execute('stop')
    assert result.returncode == 0, result.stderr
    assert calls()[-1][calls()[-1].index('-O') + 1] == 'exit'
    root = Path(__file__).resolve().parents[1]
    config = tomllib.loads((root / '.streamlit/config.toml').read_text())['server']
    assert config == {'address': '127.0.0.1', 'port': 9000}


@pytest.mark.parametrize('explicit', [True, False])
def test_existing_master_is_reused_and_preserved_when_forwarding_closes(tmp_path, ssh, explicit):
    execute, calls = ssh
    socket = tmp_path / 'existing-control'
    alive = Path(str(socket) + '.alive')
    alive.touch()
    options = ['-S', str(socket)] if explicit else []
    result = execute('start', '-H', 'example.amazonaws.com', '-u', 'ubuntu', *options,
                     configured='none' if explicit else str(socket))
    assert result.returncode == 0, result.stderr
    assert 'Reused existing SSH connection' in result.stdout
    assert not any('-M' in call for call in calls())
    forward = next(call for call in calls() if '-L' in call)
    assert forward[forward.index('-O') + 1] == 'forward'
    assert forward[forward.index('-L') + 1] == '127.0.0.2:9000:127.0.0.1:9000'
    result = execute('stop')
    assert result.returncode == 0, result.stderr
    assert alive.exists()
    assert not any('exit' in call for call in calls())
    assert any('cancel' in call for call in calls())


def test_explicit_missing_master_does_not_open_a_new_connection(tmp_path, ssh):
    execute, calls = ssh
    result = execute('start', '-H', 'example.amazonaws.com', '-S', str(tmp_path / 'missing'))
    assert result.returncode != 0
    assert 'no new connection was opened' in result.stderr
    assert not any('-M' in call or '-fNT' in call for call in calls())


def test_repeated_start_reuses_registered_forward_and_rejects_different_host(ssh):
    execute, calls = ssh
    for _ in range(2):
        result = execute('start', '-H', 'example.amazonaws.com', '-u', 'ubuntu')
        assert result.returncode == 0, result.stderr
    assert sum('-M' in call for call in calls()) == 1
    result = execute('start', '-H', 'different.amazonaws.com', '-u', 'ubuntu')
    assert result.returncode != 0
    assert sum('-M' in call for call in calls()) == 1
