import argparse
from pathlib import Path
import sys
import traceback
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT/'source'))
from engine import train, load_samples, read_json, write_json, sha256
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--arm', default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--stop-after', type=int)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read_json(ROOT/'protocol.json')
    assert sha256(ROOT/'data/schedule.npz') == config['schedule_sha256']
    assert sha256(ROOT/'data/selection.json') == config['selection_sha256']
    _, samples = load_samples(ROOT)
    names = list(config['arms']) if args.arm == 'all' else [args.arm]
    try:
        for name in names:
            if name not in config['arms']:
                raise ValueError('Unknown arm')
            train(ROOT, name, config, samples, torch.device(args.device), args.resume, args.stop_after)
        write_json(ROOT/'status.json', dict(status='checkpointed' if args.stop_after else 'complete', arms=names))
    except Exception as error:
        write_json(ROOT/'status.json', dict(status='failed', error=str(error), traceback=traceback.format_exc()))
        raise


if __name__ == '__main__':
    main()

