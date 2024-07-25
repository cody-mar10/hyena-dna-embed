from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, fields
from pathlib import Path
from typing import List

from hyena_dna.embed import HyenaDNAModule
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


def for_each_file(module: HyenaDNAModule, files: List[Path], outdir: Path):
    for file in tqdm(files, desc="FASTA files"):
        per_file_main(module, file, outdir, verbose=False)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    module = HyenaDNAModule(checkpoint=args.checkpoint)

    if len(args.inputs) == 1:
        maybe_file = args.inputs[0]

        if maybe_file.is_dir():
            files = list(maybe_file.glob("*.fna"))
            for_each_file(module, files, args.outdir)
        else:
            per_file_main(module, maybe_file, args.outdir, verbose=True)
    else:
        for_each_file(module, args.inputs, args.outdir)


if __name__ == "__main__":
    main()
