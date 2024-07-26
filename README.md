# hyena-dna-embed

Generating [HyenaDNA](https://github.com/HazyResearch/hyena-dna) genome embeddings using the "large-1m-seqlen" model, trained on up to 1M nucleotide contexts.

## Disclaimer

**This is a wrapper repository for HyenaDNA. For any problems specific to HyenaDNA, please contact the above repository.** We did not develop HyenaDNA, so we make no guarantees about it.

We have, however, simplified the dependencies to those minimally required. It is possible that additional dependencies listed on the [HyenaDNA](https://github.com/HazyResearch/hyena-dna) could significantly speed up inference, but we did not find them necessary.

## Installation

### Without GPUs

```bash
# setup torch first -- conda does this so much better than pip
mamba create -n hyena-dna -c pytorch -c conda-forge 'pytorch>=2.0' torchvision cpuonly 'python<3.12'

mamba activate hyena-dna-test

# install latest updates from this repository
pip install git+https://github.com/cody-mar10/hyena-dna-embed.git
```

### With GPUs

```bash
# setup torch first -- conda does this so much better than pip
mamba create -n hyena-dna -c pytorch -c nvidia -c conda-forge 'pytorch>=2.0' torchvision pytorch-cuda=11.8 'python<3.12'

mamba activate hyena-dna-test

# install latest updates from this repository
pip install git+https://github.com/cody-mar10/hyena-dna-embed.git
```

This will install the `hyena-dna` executable into your environment.

## Usage

The minimum command needed  for genome FASTA files with non-`ACGTN` characters removed is:

```bash
hyena-dna -i FASTAFILE
```

This will automatically download the model weights for you into `checkpoints`, but you can configure this with the `-c` flag.

You can additionally control the output directory location with the `-o` flag.

The output `.h5` file has a single `data` field, which contains the HyenaDNA genome embeddings for each genome in the input FASTA file (**in the same order**).

### Large or fragmented genomes?

For genomes larger than 1M nucleotides, we chunk these sequences into non-overlapping units of 1M nucleotides each. We embed each chunk individually and then average the embeddings to get a final genome embedding. Similarly, for fragmented genomes, we individually embed each scaffold, but report each scaffold embedding individually. For downstream analyses, we averaged scaffold embeddings for a final genome embedding.
