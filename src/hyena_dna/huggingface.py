import json
import os
import re
import subprocess

import torch
from transformers import PreTrainedModel

from hyena_dna.standalone_hyenadna import HyenaDNAModel


def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string


def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict."""

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if "backbone" in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = "model." + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except KeyError as e:
                raise KeyError("key mismatch in the state dicts!") from e

    # scratch_dict has been updated
    return scratch_dict


def setup_git_lfs():
    """Setup git-lfs on the user system if they haven't already"""

    command = "git lfs install"
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as err:
        if "git: 'lfs' is not a git command." in err.cmd:
            message = """Could not initialize git-lfs. Please install git-lfs: https://github.com/git-lfs/git-lfs
            
            Alternatively, download the model weights manually from https://huggingface.co/LongSafari/hyenadna-large-1m-seqlen
            """

            raise RuntimeError(message) from err
        raise err

    print("git-lfs setup ok!")


class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)  # type: ignore

    @classmethod
    def from_pretrained(
        cls,
        path,
        model_name,
        device="cpu",
        use_head=False,
        n_classes=2,
    ):
        # first check if it is a local path
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path):  # checks if exists also
            config_file = os.path.join(pretrained_model_name_or_path, "config.json")

        else:
            hf_url = f"https://huggingface.co/LongSafari/{model_name}"

            # setup git-lfs if not already
            setup_git_lfs()

            command = f"mkdir -p {path} && cd {path} && git clone {hf_url}"
            subprocess.run(command, shell=True)

            config_file = os.path.join(pretrained_model_name_or_path, "config.json")

            print(f"Downloaded model from {hf_url} to {pretrained_model_name_or_path}")

        config = json.load(open(config_file))

        scratch_model = HyenaDNAModel(
            **config, use_head=use_head, n_classes=n_classes
        )  # the new model format

        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, "weights.ckpt"),
            map_location=torch.device(device),
            weights_only=False,
        )

        # need to load weights slightly different if using gradient checkpointing
        checkpointing = bool(config.get("checkpoint_mixer", False))

        # grab state dict from both and load weights
        state_dict = load_weights(
            scratch_model.state_dict(),
            loaded_ckpt["state_dict"],
            checkpointing=checkpointing,
        )

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        print("Loaded pretrained weights ok!")
        return scratch_model
