from typing import Tuple

import torch
import torchvision.transforms.v2 as v2
import webdataset as wds


def create_dataset(
    batch_size: int, is_train: bool, image_size: Tuple[int, int], url: str
):
    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.RandomHorizontalFlip(0.25) if is_train else v2.Identity(),
            v2.RandomResizedCrop(image_size) if is_train else v2.Resize(image_size),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    dataset = (
        wds.WebDataset(
            url,
            shardshuffle=300,  # just a number higher than the largest number of shards
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            resampled=True if is_train else False,
        )
        .shuffle(5000)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(transform, lambda y: y)
        .batched(batch_size, partial=False)
    )
    if is_train:
        dataset = dataset.with_epoch(1_281_167 // batch_size)  # pyrefly:ignore
    return dataset


def create_dataloader(
    batch_size: int,
    is_train: bool,
    image_size: Tuple[int, int],
    url: str,
    num_workers: int,
    epoch_batches: int | None = None,
):
    dataset = create_dataset(batch_size, is_train, image_size, url)
    dataloader = wds.WebLoader(dataset, num_workers=num_workers, batch_size=None)
    if is_train and epoch_batches is not None:
        # Keep resampled training data, but force each epoch to a fixed number of batches.
        dataloader = dataloader.with_epoch(epoch_batches).with_length(epoch_batches)
    return dataloader


def create_imagenet_train_val(batch_size: int, image_size: Tuple[int, int], mode: str):
    urls_full = {
        "train": "/home/chojnowski.h/weishao/chojnowski.h/iSSL/core/data/imagenet-1k/train/imagenet-1k_{00000..00078}.tar",
        "val": "/home/chojnowski.h/weishao/chojnowski.h/iSSL/core/data/imagenet-1k/validation/imagenet-1k_{00000..00003}.tar",
    }
    urls_mini = {
        "train": "/home/chojnowski.h/weishao/chojnowski.h/iSSL/core/data/imagenet-1k/train/imagenet-1k_{00000..00001}.tar",
        "val": "/home/chojnowski.h/weishao/chojnowski.h/iSSL/core/data/imagenet-1k/validation/imagenet-1k_{00000..00001}.tar",
    }
    assert mode in ["full", "mini"], f"{mode} invalid mode."
    urls = urls_mini if mode == "mini" else urls_full
    train_epoch_batches = 1_281_167 // batch_size
    train_dataloader = create_dataloader(
        batch_size,
        True,
        image_size,
        urls["train"],
        num_workers=12,
        epoch_batches=train_epoch_batches,
    )
    val_dataloader = create_dataloader(
        batch_size,
        False,
        image_size,
        urls["val"],
        num_workers=2,
    )
    return train_dataloader, val_dataloader
