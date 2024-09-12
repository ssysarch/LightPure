import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageNet
from .GTdata import GTSRB
import pathlib


def get_dataset(
    dataset: str,
    data_dir: str,
    image_size: int,
    mode: str = "train",
):
    assert data_dir is not None, "Data directory must be specified"
    data_dir = pathlib.Path(data_dir, dataset)

    if dataset == "cifar10":
        assert image_size <= 32, "CIFAR-10 image size must be less than or equal to 32"
        assert mode in ["train", "test"], "Mode must be either 'train' or 'test'"
        dataset = CIFAR10(
            data_dir,
            train=(mode == "train"),
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
            download=True,
        )

    elif dataset == "imagenet":
        assert mode in ["train", "val"], "Mode must be either 'train' or 'val'"
        assert (
            data_dir.exists()
        ), f"Data directory {data_dir} does not exist. Please download the dataset first."

        dataset = ImageNet(
            data_dir,
            split=mode,
            transform=transforms.Compose(
                [
                    transforms.Resize(image_size),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                    ),
                ]
            ),
        )
    elif dataset == "GTSRB":
        dataset = GTSRB(
            root_dir=data_dir,
            train=(mode == "train"),
            transform=transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            ),
        )
    else:
        raise NotImplementedError(f"Dataset {dataset} not supported")

    return dataset
