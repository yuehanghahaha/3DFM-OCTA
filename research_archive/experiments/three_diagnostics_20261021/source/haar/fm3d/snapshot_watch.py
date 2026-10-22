"""Save five-epoch snapshots for a training process that is already running.

The training loop writes history only after latest.pt and best.pt have been
atomically saved. Watching history therefore preserves the selected weights
without interrupting the process or changing its in-memory configuration.
"""
import argparse
import json
import shutil
import time
from pathlib import Path

import torch


def completed_records(path):
    text = path.read_text(encoding='utf-8')
    return [json.loads(line) for line in text.rsplit('\n', 1)[0].splitlines()]


def copy_atomic(source, destination):
    if destination.exists():
        raise FileExistsError(f'Will not replace an existing snapshot: {destination}')
    temporary = destination.with_suffix('.tmp')
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def snapshot_epoch(output, epoch, records):
    record = next((row for row in records if row['epoch'] == epoch), None)
    if record is None:
        return False
    if 'validation' not in record:
        raise ValueError(f'Epoch {epoch} has no validation result')
    latest = torch.load(output / 'latest.pt', map_location='cpu', weights_only=False)
    if latest['epoch'] < epoch:
        return False
    if latest['epoch'] > epoch:
        raise RuntimeError(f'Missed epoch {epoch}; latest.pt already contains epoch {latest["epoch"]}')
    validations = (row for row in records if row['epoch'] <= epoch and 'validation' in row)
    selected = min(validations, key=lambda row: row['validation']['mae'])
    best = torch.load(output / 'best.pt', map_location='cpu', weights_only=False)
    if best['epoch'] != selected['epoch'] or best['best_mae'] != selected['validation']['mae']:
        raise RuntimeError(f'best.pt does not match the best validation model through epoch {epoch}')
    if latest['best_mae'] != best['best_mae']:
        raise RuntimeError(f'latest.pt and best.pt disagree at epoch {epoch}')
    current_path = output / f'epoch_{epoch:04d}.pt'
    best_path = output / f'best_through_{epoch:04d}.pt'
    if current_path.exists() or best_path.exists():
        if not (current_path.exists() and best_path.exists()):
            raise FileExistsError(f'Incomplete snapshot for epoch {epoch}')
        current = torch.load(current_path, map_location='cpu', weights_only=False)
        selected_snapshot = torch.load(best_path, map_location='cpu', weights_only=False)
        if current['epoch'] != epoch or (selected_snapshot['epoch'], selected_snapshot['best_mae']) != (
                best['epoch'], best['best_mae']):
            raise RuntimeError(f'Existing snapshot for epoch {epoch} does not match history')
        print(f'epoch={epoch} snapshots already saved (selected epoch={best["epoch"]})', flush=True)
        return True
    copy_atomic(output / 'latest.pt', current_path)
    copy_atomic(output / 'best.pt', best_path)
    print(f'epoch={epoch} saved current and best-through (selected epoch={best["epoch"]})', flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--first', type=int, required=True)
    parser.add_argument('--last', type=int, required=True)
    parser.add_argument('--interval', type=int, default=5)
    parser.add_argument('--poll-seconds', type=float, default=2)
    args = parser.parse_args()
    if args.interval < 1 or args.first < 1 or args.last < args.first or args.poll_seconds <= 0:
        parser.error('Invalid epoch range, interval or poll period')
    if args.first % args.interval or args.last % args.interval:
        parser.error('First and last epochs must be multiples of interval')
    for epoch in range(args.first, args.last + 1, args.interval):
        while not snapshot_epoch(args.output, epoch, completed_records(args.output / 'history.jsonl')):
            time.sleep(args.poll_seconds)
    print('All requested snapshots saved.', flush=True)


if __name__ == '__main__':
    main()
