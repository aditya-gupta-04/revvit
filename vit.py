import torch
from torch import nn

# Needed to implement custom backward pass
from torch.autograd import Function as Function

from rev import *
from fast_rev import *
from tokenmixers import *

def build_model(args):
    if args.model == "vit":
        if args.pareprop:
            rev_arch = FastRevViT
        else:
            rev_arch = RevViT

        model = rev_arch(
            embed_dim=args.embed_dim,
            n_head=args.n_head,
            depth=args.depth,
            drop_path_rate=(0.1 if args.deit_scheme else 0.0),
            patch_size=args.patch_size,
            image_size=args.image_size,
            num_classes=args.num_classes,
            enable_amp=args.amp,
            token_mixer=args.token_mixer,
            pool_size=args.pool_size,
        )
    elif args.model == "swin":
        model = RevSwin(
            img_size=args.image_size,
            patch_size=args.patch_size,
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            depths=[args.depth // 2, args.depth // 2],
            num_heads=[args.n_head, args.n_head * 2],
            window_size=4,
            fast_backprop=args.pareprop,
        )
    elif args.model == "mvit":
        model = RevMViT(
            img_size=args.image_size,
            patch_kernel=(3, 3),
            patch_stride=(2, 2),
            patch_padding=(1, 1),
            num_classes=args.num_classes,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.n_head,  # doubles every stage
            last_block_indexes=[0, 2],
            qkv_pool_kernel=(3, 3),
            adaptive_kv_stride=2,
            adaptive_window_size=16,
            fast_backprop=args.pareprop,
        )
    elif args.model == "vit-og":
        model = ViT_OG(
            embed_dim=args.embed_dim,
            n_head=args.n_head,
            depth=args.depth,
            patch_size=args.patch_size,
            image_size=args.image_size,
            num_classes=args.num_classes,
            enable_amp=args.amp,
            token_mixer=args.token_mixer,
            pool_size=args.pool_size,
        )
    elif args.model == "vit-small":
        print("Warning : vit-small is not configured to its native 224x224 image and 16x16 patch size")
        print("Warning : vit-small modified to not use class tokens and classifier head is uses average pool of output sequence")
        model = timm.create_model("vit_small_patch16_224", pretrained=False, 
                                    num_classes=args.num_classes, img_size=args.image_size, patch_size=args.patch_size,
                                    class_token=False, global_pool='avg', drop_rate=0.0, drop_path_rate=(0.1 if args.deit_scheme else 0.0))
    else:
        raise NotImplementedError(f"Model {args.model} not supported.")
    
    print(f"\nNumber of model parameters: {sum(p.numel() for p in model.parameters())}\n")

    # Whether to use memory-efficient reversible backpropagation or vanilla backpropagation
    # Note that in both cases, the model is reversible.
    # For Swin, this requires iterating through the layers.
    model.no_custom_backward = args.vanilla_bp
    
    return model

class ViT_OG(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        n_head=8,
        depth=8,
        patch_size=(
            2,
            2,
        ),  # this patch size is used for CIFAR-10
        # --> (32 // 2)**2 = 256 sequence length
        image_size=(32, 32),  # CIFAR-10 image size
        num_classes=10,
        enable_amp=False,
        token_mixer="attention",
        pool_size=3
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.n_head = n_head
        self.depth = depth
        self.patch_size = patch_size

        num_patches = (image_size[0] // self.patch_size[0]) * (
            image_size[1] // self.patch_size[1]
        )

        patches_shape = (image_size[0] // self.patch_size[0], image_size[1] // self.patch_size[1])

        # Reversible blocks can be treated same as vanilla blocks,
        # any special treatment needed for reversible bacpropagation
        # is contrained inside the block code and not exposed.
        self.layers = nn.ModuleList(
            [
                Block(
                    dim=self.embed_dim,
                    num_heads=self.n_head,
                    enable_amp=enable_amp,
                    token_mixer=token_mixer,
                    pool_size=pool_size,
                    patches_shape=patches_shape,
                )
                for _ in range(self.depth)
            ]
        )

        # Standard Patchification and absolute positional embeddings as in ViT
        self.patch_embed = nn.Conv2d(
            3, self.embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.pos_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, self.embed_dim)
        )
        # What kind of a shit initialization is this? Could have used randn * 0.02 like how its done in timm

        # The two streams are concatenated and passed through a linear
        # layer for final projection. This is the only part of RevViT
        # that uses different parameters/FLOPs than a standard ViT model.
        # Note that fusion can be done in several ways, including
        # more expressive methods like in paper, but we use
        # linear layer + LN for simplicity.
        self.head = nn.Linear(self.embed_dim, num_classes, bias=True)
        self.norm = nn.LayerNorm(self.embed_dim)

    def forward(self, x):
        # patchification using conv and flattening
        # + abolsute positional embeddings
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x += self.pos_embeddings

        # the two streams X_1 and X_2 are initialized identically with x and
        # concatenated along the last dimension to pass into the reversible blocks
        # x = torch.cat([x, x], dim=-1)

        for _, layer in enumerate(self.layers):
            x = layer(x)
        

        # aggregate across sequence length
        x = x.mean(1)

        # head pre-norm
        x = self.norm(x) 

        # pre-softmax logits
        x = self.head(x)

        # return pre-softmax logits
        return x
    

class Block(nn.Module):

    def __init__(self, dim, num_heads, enable_amp,  token_mixer, pool_size, patches_shape):
        super().__init__()
        # F and G can be arbitrary functions, here we use
        # simple attwntion and MLP sub-blocks using vanilla attention.
        if token_mixer == "attention":
            self.F = AttentionSubBlock(
                dim=dim, num_heads=num_heads, enable_amp=enable_amp
            )
            print("Using attention token mixer")
        elif token_mixer == "pooling":
            self.F = PoolingFBlock(
                dim=dim, pool_size=pool_size, patches_shape=patches_shape, enable_amp=enable_amp
            )
            print(f"Using pooling token mixer with pool_size : {pool_size}")
        elif token_mixer == "spatial_mlp":
            self.F = SpatialMLPFBlock(
                dim=dim, patches_shape=patches_shape, enable_amp=enable_amp
            )
            print("Using spatial_mlp token mixer")
        else:
            print(f"Unsupported Token Mixer {token_mixer}")
            quit()

        self.G = MLPSubblock(dim=dim, enable_amp=enable_amp)

    def forward(self, X):
        X = self.F(X) + X
        X = self.G(X) + X
        return X


