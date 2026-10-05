"""Check offline snapshot completeness without loading a model into RAM."""
import json
from pathlib import Path


def model_ready(root, model):
    snapshots = Path(root) / '.models/hub' / ('models--' + model.replace('/', '--')) / 'snapshots'
    for snapshot in snapshots.glob('*'):
        if not (snapshot / 'config.json').is_file() or not (snapshot / 'tokenizer_config.json').is_file():
            continue
        try:
            index = snapshot / 'model.safetensors.index.json'
            names = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
            for name in names:
                path = snapshot / name
                with path.open('rb') as stream:
                    header_size = int.from_bytes(stream.read(8), 'little')
                    if not 0 < header_size < 100_000_000:
                        raise ValueError('Invalid weight header')
                    header = json.loads(stream.read(header_size))
                end = max(v['data_offsets'][1] for k, v in header.items() if k != '__metadata__')
                if path.stat().st_size != 8 + header_size + end:
                    raise ValueError('Incomplete weight file')
            return True
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return False
