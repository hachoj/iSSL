import torch
import torchvision.transforms.v2 as v2
import webdataset as wds


def create_dataset(batch_size: int, mode: str = "train", image_size: int = 256):
    urls = {
        "train": "/home/chojnowski.h/weishao/chojnowski.h/iSSL/imagenet-1k/train/imagenet-1k_{00000..00208}.tar",
        "val": "/home/chojnowski.h/weishao/chojnowski.h/iSSL/imagenet-1k/val/imagenet-1k_{00000..00005}.tar",
        "test": "/home/chojnowski.h/weishao/chojnowski.h/iSSL/imagenet-1k/test/imagenet-1k_{00000..00007}.tar",
    }[mode]

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((image_size, image_size)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225),
            ),
        ]
    )

    dataset = (
        wds.WebDataset(
            urls,
            shardshuffle=300,  # just a number higher than the largest number of shards
            nodesplitter=wds.split_by_node,
            workersplitter=wds.split_by_worker,
            resampled=True if mode == "train" else False
        )
        .shuffle(5000)
        .decode("pil")
        .to_tuple("jpg", "cls")
        .map_tuple(transform, lambda y: y)
        .batched(batch_size, partial=False)
    )
    if mode == 'train':
        dataset = dataset.with_epoch(1_281_167 // batch_size)
    return dataset


def create_dataloader(batch_size: int, mode: str = "train", image_size: int = 256):
    dataset = create_dataset(batch_size, mode, image_size)
    dataloader = wds.WebLoader(dataset, num_workers=2, batch_size=None)
    return dataloader
