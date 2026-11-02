"""Small logging helpers shared by workers, the API and the thin client."""
import codecs
import logging
from pathlib import Path
import sys

from .io import write_json


def get_logger():
    logger = logging.getLogger('lidarpoc')
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger


def progress(directory, stage, completed=0, total=1):
    write_json(Path(directory) / 'progress.json', {'stage': stage, 'completed': completed, 'total': total})
    get_logger().info('%s [%s/%s]', stage, completed, total)


def relay_output(process, path):
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    with Path(path).open('wb') as log, process.stdout:
        for chunk in iter(lambda: process.stdout.read1(8192), b''):
            log.write(chunk)
            log.flush()
            print(decoder.decode(chunk), end='', flush=True)
        print(decoder.decode(b'', final=True), end='', flush=True)
    return process.wait()


def read_log(path, offset=0, limit=65536):
    path = Path(path)
    if not path.is_file():
        return {'text': '', 'offset': 0, 'reset': offset != 0}
    with path.open('rb') as log:
        reset = offset > path.stat().st_size
        log.seek(0 if reset else offset)
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        text = decoder.decode(log.read(limit))
        return {'text': text, 'offset': log.tell() - len(decoder.getstate()[0]), 'reset': reset}


def sync_remote_log(api, job_id, cursor):
    chunk = api('GET', f'/jobs/{job_id}/logs', params={'offset': cursor.get('offset', 0)}).json()
    if chunk['reset']:
        cursor['text'] = ''
    text = chunk['text']
    if text:
        print(f'[remote {job_id[:8]}] {text}', end='' if text.endswith('\n') else '\n', flush=True)
    cursor['offset'] = chunk['offset']
    cursor['text'] = (cursor.get('text', '') + text.replace('\r', '\n'))[-65536:]
    return cursor['text']
