import re
from math import ceil
from pathlib import Path
from typing import Union

import einops
import tables as tb
import torch
from fastatools import FastaFile
from huggingface import HyenaDNAPreTrainedModel
from more_itertools import ichunked
from tqdm import tqdm

from standalone_hyenadna import CharacterTokenizer

FilePath = Union[str, Path]


class HyenaDNAModule:
    MODEL_NAME = "hyenadna-large-1m-seqlen"
    MAX_LEN = 1_000_000
    AMBIGUOUS_CHAR = "N"
    AMBIGUOUS_CHAR_PATTERN = re.compile(r"[^ACGTN]")

    def __init__(self, checkpoint: FilePath):
        # account for beginning and end of sequence tokens
        self.eff_max_len = HyenaDNAModule.MAX_LEN - 2

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = (
            HyenaDNAPreTrainedModel.from_pretrained(
                path=checkpoint,
                model_name=HyenaDNAModule.MODEL_NAME,
                download=False,
                config=None,
                use_head=False,
            )
            .to(self.device)
            .eval()
        )

        self.hidden_dim = 256

        self.tokenizer = CharacterTokenizer(
            characters=list("ACGTN"),
            model_max_length=HyenaDNAModule.MAX_LEN,
        )

    def tokenize(self, sequence: str) -> torch.Tensor:
        return (
            torch.tensor(self.tokenizer(sequence)["input_ids"])
            .unsqueeze(0)
            .to(self.device)
        )

    @torch.inference_mode()
    def forward(self, sequence: str) -> torch.Tensor:
        token_ids = self.tokenize(sequence)
        # skip the beginning and end of sequence tokens
        output: torch.Tensor = self.model(token_ids)[:, 1:-1, :]
        return einops.reduce(output, "1 nt dim -> 1 dim", reduction="mean")

    def predict_loop(
        self, fasta_file: FilePath, output: FilePath, verbose: bool = True
    ):
        fasta = FastaFile(fasta_file)
        n_seqs = len(fasta)

        with tb.File(output, "w") as fp:
            storage = fp.create_earray(
                fp.root,
                "data",
                atom=tb.Float32Atom(),
                shape=(0, self.hidden_dim),
                expectedrows=n_seqs,
                filters=tb.Filters(complib="blosc:lz4", complevel=4),
            )

            for record in tqdm(
                fasta.parse(), total=n_seqs, disable=not verbose, desc="Sequences"
            ):
                sequence = HyenaDNAModule.AMBIGUOUS_CHAR_PATTERN.sub(
                    HyenaDNAModule.AMBIGUOUS_CHAR, record.sequence
                )

                n_fragments = ceil(len(sequence) / self.eff_max_len)
                if n_fragments == 1:
                    embedding = self.forward(sequence)
                else:
                    # average over all fragments
                    embedding = torch.zeros(1, self.hidden_dim, device=self.device)

                    for fragment in ichunked(sequence, self.eff_max_len):
                        subseq = "".join(fragment)
                        embedding += self.forward(subseq)

                    embedding /= n_fragments

                storage.append(embedding.cpu().numpy())
