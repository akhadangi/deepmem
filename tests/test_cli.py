from pathlib import Path
from deepmem.train import parser


def test_train_cli_options():
    args = parser().parse_args(['--config', 'configs/tiny_smoke.json',
                                '--data-files', 'examples/sample.txt',
                                '--output-dir', 'runs/test', '--batch-size', '1'])
    assert args.batch_size == 1 and args.config.endswith('tiny_smoke.json')
