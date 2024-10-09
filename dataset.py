from typing import List, Union
import math
import itertools
import webdataset as wds
from braceexpand import braceexpand

from torchvision import transforms
import torchvision.transforms.functional as TF
from diffusers.training_utils import resolve_interpolation_mode

from common_utils import tarfile_to_samples_nothrow, filter_keys, default_collate


class SDText2ImageDataset:
    def __init__(
        self,
        train_shards_path_or_url: Union[str, List[str]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        resolution: int = 512,
        interpolation_type: str = "bilinear",
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        work_on_latent: bool = False,
    ):
        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [
                list(braceexpand(urls)) for urls in train_shards_path_or_url
            ]
            # flatten list using itertools
            train_shards_path_or_url = list(
                itertools.chain.from_iterable(train_shards_path_or_url)
            )

        if work_on_latent:
            processing_pipeline = [
                wds.decode("l", handler=wds.ignore_and_continue),
                wds.rename(
                    param="npy",
                    text="txt",
                    handler=wds.warn_and_continue,
                ),
                wds.map(filter_keys({"param", "text"})),
                wds.to_tuple("param", "text"),
            ]
        else:
            interpolation_mode = resolve_interpolation_mode(interpolation_type)

            def transform(example):
                # resize image
                image = example["image"]
                image = TF.resize(image, resolution, interpolation=interpolation_mode)

                # get crop coordinates and crop image
                c_top, c_left, _, _ = transforms.RandomCrop.get_params(
                    image, output_size=(resolution, resolution)
                )
                image = TF.crop(image, c_top, c_left, resolution, resolution)
                image = TF.to_tensor(image)
                image = TF.normalize(image, [0.5], [0.5])

                example["image"] = image
                return example

            processing_pipeline = [
                wds.decode("pil", handler=wds.ignore_and_continue),
                wds.rename(
                    image="jpg;png;jpeg;webp",
                    text="text;txt;caption",
                    handler=wds.warn_and_continue,
                ),
                wds.map(filter_keys({"image", "text"})),
                wds.map(transform),
                wds.to_tuple("image", "text"),
            ]

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(
                per_gpu_batch_size, partial=False, collation_fn=default_collate
            ),
        ]

        num_worker_batches = math.ceil(
            num_train_examples / (global_batch_size * num_workers)
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader


class Text2ImageDataset:
    def __init__(
        self,
        train_shards_path_or_url: Union[str, List[str]],
        num_train_examples: int,
        per_gpu_batch_size: int,
        global_batch_size: int,
        num_workers: int,
        resolution: int = 1024,
        shuffle_buffer_size: int = 1000,
        pin_memory: bool = False,
        persistent_workers: bool = False,
        use_fix_crop_and_size: bool = False,
    ):
        if not isinstance(train_shards_path_or_url, str):
            train_shards_path_or_url = [
                list(braceexpand(urls)) for urls in train_shards_path_or_url
            ]
            # flatten list using itertools
            train_shards_path_or_url = list(
                itertools.chain.from_iterable(train_shards_path_or_url)
            )

        def get_orig_size(json):
            if use_fix_crop_and_size:
                return (resolution, resolution)
            else:
                return (
                    int(json.get("original_width", 0.0)),
                    int(json.get("original_height", 0.0)),
                )

        def transform(example):
            # resize image
            image = example["image"]
            image = TF.resize(
                image, resolution, interpolation=transforms.InterpolationMode.BILINEAR
            )

            # get crop coordinates and crop image
            c_top, c_left, _, _ = transforms.RandomCrop.get_params(
                image, output_size=(resolution, resolution)
            )
            image = TF.crop(image, c_top, c_left, resolution, resolution)
            image = TF.to_tensor(image)
            image = TF.normalize(image, [0.5], [0.5])

            example["image"] = image
            example["crop_coords"] = (
                (c_top, c_left) if not use_fix_crop_and_size else (0, 0)
            )
            return example

        processing_pipeline = [
            wds.decode("pil", handler=wds.ignore_and_continue),
            wds.rename(
                image="jpg;png;jpeg;webp",
                text="text;txt;caption",
                orig_size="json",
                handler=wds.warn_and_continue,
            ),
            wds.map(filter_keys({"image", "text", "orig_size"})),
            wds.map_dict(orig_size=get_orig_size),
            wds.map(transform),
            wds.to_tuple("image", "text", "orig_size", "crop_coords"),
        ]

        # Create train dataset and loader
        pipeline = [
            wds.ResampledShards(train_shards_path_or_url),
            tarfile_to_samples_nothrow,
            wds.shuffle(shuffle_buffer_size),
            *processing_pipeline,
            wds.batched(
                per_gpu_batch_size, partial=False, collation_fn=default_collate
            ),
        ]

        num_worker_batches = math.ceil(
            num_train_examples / (global_batch_size * num_workers)
        )  # per dataloader worker
        num_batches = num_worker_batches * num_workers
        num_samples = num_batches * global_batch_size

        # each worker is iterating over this
        self._train_dataset = wds.DataPipeline(*pipeline).with_epoch(num_worker_batches)
        self._train_dataloader = wds.WebLoader(
            self._train_dataset,
            batch_size=None,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        # add meta-data to dataloader instance for convenience
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader
