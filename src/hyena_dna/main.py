from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List

from embed import HyenaDNAModule
from tqdm import tqdm


@dataclass
class Args:
    inputs: List[Path]
    outdir: Path
    checkpoint: Path

    @classmethod
    def from_namespace(cls, ns: Namespace):
        kwargs = {field.name: getattr(ns, field.name) for field in fields(cls)}

        return cls(**kwargs)


def parse_args() -> Args:
    parser = ArgumentParser(
        description="Generate HYENA DNA embeddings for genomes using the large 1M sequence length model"
    )

    parser.add_argument(
        "-i",
        "--inputs",
        nargs="+",
        metavar="FILE(s)",
        type=Path,
        required=True,
        help="Input FASTA genome file(s) to embed.",
    )

    parser.add_argument(
        "-o",
        "--outdir",
        metavar="DIR",
        type=Path,
        default=Path("hyena-DNA_output"),
        help="Output directory with .h5 embedding files (default: %(default)s)",
    )

    parser.add_argument(
        "-c",
        "--checkpoint",
        metavar="DIR",
        type=Path,
        default=Path("checkpoints"),
        help="Hugging face model checkpoint dir (default: %(default)s)",
    )

    return Args.from_namespace(parser.parse_args())


def per_file_main(
    module: HyenaDNAModule, file: Path, outdir: Path, verbose: bool = True
):
    basename = f"{file.stem}_HYENA-DNA.h5"
    output = outdir.joinpath(basename)
    module.predict_loop(file, output, verbose=verbose)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    module = HyenaDNAModule(checkpoint=args.checkpoint)

    if len(args.inputs) == 1:
        per_file_main(module, args.inputs[0], args.outdir, verbose=True)
    else:
        for file in tqdm(args.inputs, desc="FASTA files"):
            per_file_main(module, file, args.outdir, verbose=False)


if __name__ == "__main__":
    main()
