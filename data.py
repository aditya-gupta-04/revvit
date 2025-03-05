import torch
import torchvision
import torchvision.transforms as transforms
import os
import shutil


def get_data_loader(args):

    print("==> Preparing data..")

    if args.dataset == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        # Will downloaded and save the dataset if needed
        trainset = torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.bs, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.bs, shuffle=False, num_workers=2
        )
        
        num_classes = len(trainset.classes)  
        images, labels = next(iter(trainloader))
        print(f"Train samples: {len(trainset)}, Test samples: {len(testset)}, Num Classes : {num_classes}, Batch shape: {images.shape}")

        return trainloader, testloader
    
    elif args.dataset == "CIFAR100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
        ])

        # Will downloaded and save the dataset if needed
        trainset = torchvision.datasets.CIFAR100(
            root="./data", train=True, download=True, transform=transform_train
        )
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=args.bs, shuffle=True, num_workers=2
        )

        testset = torchvision.datasets.CIFAR100(
            root="./data", train=False, download=True, transform=transform_test
        )
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=args.bs, shuffle=False, num_workers=2
        )

        num_classes = len(trainset.classes)  
        images, labels = next(iter(trainloader))
        print(f"Train samples: {len(trainset)}, Test samples: {len(testset)}, Num Classes : {num_classes}, Batch shape: {images.shape}")

        return trainloader, testloader
    
    elif args.dataset == "TinyImageNet":

        if os.path.isdir("./data/tiny-imagenet-200"):
            print("data/tiny-imagenet-200 directory already exists!")
            print("Using Existing Image Folder Directory")
        else:
            print("Downloading Tiny ImageNet")
            
            if not os.path.isdir("./data"):
                os.makedirs("./data")
            os.system("wget http://cs231n.stanford.edu/tiny-imagenet-200.zip")
            os.system("unzip -q tiny-imagenet-200.zip -d data/")
            os.system(f"rm -rf tiny-imagenet-200.zip")

            with open("./data/tiny-imagenet-200/val/val_annotations.txt", "r") as f:
                data = f.readlines()

            for line in data:
                image_name, class_id, *_ = line.split("\t")

                if not os.path.exists(f"./data/tiny-imagenet-200/val/{class_id}"):
                    os.makedirs(f"./data/tiny-imagenet-200/val/{class_id}")

                shutil.move(f"./data/tiny-imagenet-200/val/images/{image_name}", f"./data/tiny-imagenet-200/val/{class_id}/{image_name}")

            shutil.rmtree("./data/tiny-imagenet-200/val/images/")
            os.system("rm -rf ./data/tiny-imagenet-200/val/val_annotations.txt")

            for dir in os.listdir("./data/tiny-imagenet-200/train/"):
                for img in os.listdir(f"./data/tiny-imagenet-200/train/{dir}/images/"):
                    shutil.move(f"./data/tiny-imagenet-200/train/{dir}/images/{img}", f"./data/tiny-imagenet-200/train/{dir}/{img}")

                shutil.rmtree(f"./data/tiny-imagenet-200/train/{dir}/images/")
                os.system(f"rm -rf ./data/tiny-imagenet-200/train/{dir}/*.txt")

        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(64),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        transform_test = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Define dataset paths
        train_dir = "./data/tiny-imagenet-200/train"
        val_dir = "./data/tiny-imagenet-200/val"

        # Load dataset using ImageFolder
        trainset = torchvision.datasets.ImageFolder(root=train_dir, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2)

        testset = torchvision.datasets.ImageFolder(root=val_dir, transform=transform_test)
        testloader = torch.utils.data.DataLoader(testset, batch_size=args.bs, shuffle=False, num_workers=2)

        num_classes = len(trainset.classes)  
        images, labels = next(iter(trainloader))
        print(f"Train samples: {len(trainset)}, Test samples: {len(testset)}, Num Classes : {num_classes}, Batch shape: {images.shape}")

        return trainloader, testloader

    else:
        print(f"{args.dataset} not known")