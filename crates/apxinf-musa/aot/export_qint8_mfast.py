"""Derive the MP22 encode down M-fast image source from the vendored tile."""
import argparse
import hashlib
from pathlib import Path

SYMBOL = 'musa_asm_s8mma_nhwc_conv_KRSC_256_256x128_lin_ol_outbf16'
SYMBOL_OCCURRENCES = 10


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True, help='output ASM file')
    parser.add_argument('--check', action='store_true',
                        help='verify --out instead of writing it')
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
        count = text.count(old)
        if count != 1:
            parser.error(f'expected exactly one {old!r}, found {count}')
        text = text.replace(old, new)
    count = text.count(SYMBOL)
    if count != SYMBOL_OCCURRENCES:
        parser.error(
            f'expected {SYMBOL_OCCURRENCES} occurrences of {SYMBOL}, found {count}'
        )
    text = text.replace(SYMBOL, SYMBOL + '_mfast')
    if args.check:
        if not args.out.is_file() or args.out.read_text() != text:
            parser.error(f'{args.out} is not the derived image')
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)


if __name__ == '__main__':
    main()
