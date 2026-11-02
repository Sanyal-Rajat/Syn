import json
import os
from pathlib import Path
import subprocess
import tomllib


def test_ssh_key_and_port_9000_use_a_separate_client_loopback_address(tmp_path):
    root = Path(__file__).resolve().parents[1]
    binary = tmp_path / 'ssh'
    arguments = tmp_path / 'ssh-arguments.json'
    binary.write_text('#!/usr/bin/python3\nimport json, os, sys\n'
                      'with open(os.environ["SSH_ARGUMENTS"], "w") as output: json.dump(sys.argv[1:], output)\n')
    binary.chmod(0o755)
    result = subprocess.run(['bash', str(root / 'ssh_tunnel.sh'), 'start', '-H', 'example.amazonaws.com',
                             '-u', 'ubuntu', '-k', '/client/key.pem'], capture_output=True, text=True,
                            env={**os.environ, 'PATH': f'{tmp_path}{os.pathsep}{os.environ["PATH"]}',
                                 'TMPDIR': str(tmp_path), 'SSH_ARGUMENTS': str(arguments)})
    assert result.returncode == 0, result.stderr
    args = json.loads(arguments.read_text())
    assert args[args.index('-L') + 1] == '127.0.0.2:9000:127.0.0.1:9000'
    assert args[args.index('-i') + 1] == '/client/key.pem'
    assert args[args.index('-p') + 1] == '22'
    assert args[-1] == 'example.amazonaws.com'
    config = tomllib.loads((root / '.streamlit/config.toml').read_text())['server']
    assert config == {'address': '127.0.0.1', 'port': 9000}
