import os
from typing import Tuple

import numpy as np
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import torch
from nvidia.dali import pipeline_def
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_sharding():
    """Return (shard_id, num_shards) for DDP. Falls back to single-GPU."""
    if torch.distributed.is_initialized():
        return torch.distributed.get_rank(), torch.distributed.get_world_size()
    return 0, 1


def _get_device_id():
    """Return local GPU index (respects LOCAL_RANK for multi-node DDP)."""
    if torch.distributed.is_initialized():
        return int(os.environ.get("LOCAL_RANK", 0))
    return 0


def _parse_cls(label_bytes):
    """Convert raw .cls bytes (text integer) → int64 numpy scalar.

    HuggingFace webdataset stores class labels as UTF-8 text integers.
    This runs on CPU, but the cost is negligible for a single int per sample.
    """
    text = label_bytes.tobytes().decode("utf-8").strip()
    return np.array([int(text)], dtype=np.int64)


# ---------------------------------------------------------------------------
# DALI pipelines
# ---------------------------------------------------------------------------

@pipeline_def
def _train_pipeline(
    tar_paths, idx_paths, image_size, shard_id, num_shards, reader_seed, aug_seed
):
    img_raw, label = fn.readers.webdataset(
        paths=tar_paths,
        index_paths=idx_paths,
        ext=["jpg", "cls"],
        random_shuffle=True,
        initial_fill=5000,
        seed=reader_seed,
        shard_id=shard_id,
        num_shards=num_shards,
        name="Reader",
        missing_component_behavior="error",
    )
    label = fn.python_function(label, function=_parse_cls)

    # Fused JPEG decode + random crop on GPU (avoids decoding full res)
    img = fn.decoders.image_random_crop(
        img_raw,
        device="mixed",
        output_type=types.RGB,
        random_aspect_ratio=[0.75, 4.0 / 3.0],
        random_area=[0.08, 1.0],
        seed=aug_seed,
    )
    img = fn.resize(img, device="gpu", size=image_size)
    coin = fn.random.coin_flip(probability=0.2, seed=aug_seed + 1)
    img = fn.crop_mirror_normalize(
        img,
        device="gpu",
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=coin,
    )
    return img, label


@pipeline_def
def _val_pipeline(tar_paths, idx_paths, image_size, shard_id, num_shards, reader_seed):
    img_raw, label = fn.readers.webdataset(
        paths=tar_paths,
        index_paths=idx_paths,
        ext=["jpg", "cls"],
        random_shuffle=False,
        seed=reader_seed,
        shard_id=shard_id,
        num_shards=num_shards,
        name="Reader",
        missing_component_behavior="error",
    )
    label = fn.python_function(label, function=_parse_cls)

    img = fn.decoders.image(img_raw, device="mixed", output_type=types.RGB)
    # Exact resize to match your original v2.Resize(image_size)
    img = fn.resize(img, device="gpu", size=image_size)
    img = fn.crop_mirror_normalize(
        img,
        device="gpu",
        dtype=types.FLOAT,
        output_layout="CHW",
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return img, label


# ---------------------------------------------------------------------------
# Wrapper so the iterator yields (images, labels) like your old dataloader
# ---------------------------------------------------------------------------

class DALILoader:
    """Thin wrapper around DALIGenericIterator that yields (images, labels)."""

    def __init__(self, dali_iterator):
        self._iterator = dali_iterator

    def __iter__(self):
        for batch in self._iterator:
            images = batch[0]["images"]           # already a CUDA tensor, CHW
            labels = batch[0]["labels"].squeeze(-1).long()
            yield images, labels

    def __len__(self):
        return len(self._iterator)


# ---------------------------------------------------------------------------
# Builder (replaces create_dataset / create_dataloader)
# ---------------------------------------------------------------------------

def _build_loader(
    pipeline_fn,
    tar_paths,
    idx_paths,
    batch_size,
    image_size,
    is_train,
    num_threads,
    seed,
):
    shard_id, num_shards = _get_sharding()
    device_id = _get_device_id()
    reader_seed = int(seed) + shard_id
    aug_seed = int(seed) + 10_000 + shard_id

    pipe_kwargs = dict(
        tar_paths=tar_paths,
        idx_paths=idx_paths,
        image_size=image_size,
        shard_id=shard_id,
        num_shards=num_shards,
        batch_size=batch_size,
        num_threads=num_threads,
        device_id=device_id,
        seed=reader_seed,  # pipeline-level RNG state
    )
    if is_train:
        pipe_kwargs["reader_seed"] = reader_seed
        pipe_kwargs["aug_seed"] = aug_seed
    else:
        pipe_kwargs["reader_seed"] = reader_seed
    pipe = pipeline_fn(**pipe_kwargs)
    pipe.build()

    iterator = DALIGenericIterator(
        pipe,
        ["images", "labels"],
        reader_name="Reader",                     # enables proper epoch tracking
        last_batch_policy=(
            LastBatchPolicy.DROP if is_train else LastBatchPolicy.PARTIAL
        ),
        auto_reset=True,                          # resets automatically each epoch
    )
    return DALILoader(iterator)


# ---------------------------------------------------------------------------
# Public API  (same signature as before)
# ---------------------------------------------------------------------------

def create_imagenet_train_val(
    batch_size: int,
    image_size: Tuple[int, int],
    mode: str,
    seed: int = 0,
    num_train_threads: int = 12,
    num_val_threads: int = 2,
):
    image_size = list(image_size)
    base = os.environ.get(
        "LOCAL_DATA_PATH",
        "/blue/weishao/chojnowski.h/data/imagenet-1k",
    )
    print(f"Using base path: {base}")

    shard_ranges = {
        "full": {"train": range(79), "val": range(4)},
        "mini": {"train": range(2), "val": range(2)},
    }
    assert mode in shard_ranges, f"{mode} invalid mode."
    ranges = shard_ranges[mode]

    def _paths(split, r):
        tars = [f"{base}/{split}/imagenet-1k_{i:05d}.tar" for i in r]
        idxs = [f"{base}/{split}/imagenet-1k_{i:05d}.idx" for i in r]
        return tars, idxs

    train_tars, train_idxs = _paths("train", ranges["train"])
    val_tars, val_idxs = _paths("validation", ranges["val"])

    train_loader = _build_loader(
        _train_pipeline, train_tars, train_idxs,
        batch_size,
        image_size,
        is_train=True,
        num_threads=num_train_threads,
        seed=seed,
    )
    val_loader = _build_loader(
        _val_pipeline, val_tars, val_idxs,
        batch_size,
        image_size,
        is_train=False,
        num_threads=num_val_threads,
        seed=seed + 1_000_000,
    )
    return train_loader, val_loader
