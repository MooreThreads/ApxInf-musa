"""Derive the MP22 encode down M-fast image source from the vendored tile."""
import argparse
import hashlib
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='output ASM file')
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[1] / 'kernels/asm/qint8_256x128.asm'
    if hashlib.sha256(source.read_bytes()).hexdigest() != (
        '223a1cf05ae582bdf5b8696d8911b73c976225b43e8273e5783b7c48d6b8da0b'
    ):
        parser.error('unreviewed source revision; review the block-index mapping first')
    if args.out.resolve() == source:
        parser.error('output must not overwrite the original tile')
    text = source.read_text()
    for old, new in [('R20, LM4', 'R20, LM5'), ('R21, LM5', 'R21, LM4')]:
        old, new = 'MOV.LD.LM.B32 ' + old, 'MOV.LD.LM.B32 ' + new
        assert text.count(old) == 1
        text = text.replace(old, new)
    symbol = 'musa_asm_s8mma_nhwc_conv_KRSC_256_256x128_lin_ol_outbf16'
    text = text.replace(symbol, symbol + '_mfast')
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)


if __name__ == '__main__':
    main()
