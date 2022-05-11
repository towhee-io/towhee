# original code from https://github.com/SwinTransformer/Video-Swin-Transformer
# modified by Zilliz.

from torch import nn
import torch
from einops import rearrange


from towhee.models.layers.patch_embed3d import PatchEmbed3D
from towhee.models.layers.patch_merging3d import PatchMerging3D
from towhee.models.video_swin_transformer.video_swin_transformer_block import VideoSwinTransformerBlock


class VideoSwinTransformer(nn.Module):
    """
    Video Swin Transformer.
    Ze Liu, Jia Ning, Yue Cao, Yixuan Wei, Zheng Zhang, Stephen Lin, Han Hu
    https://arxiv.org/pdf/2106.13230.pdf

    Args:
        pretrained (`str`):
            Load pretrained weights. Default: None
        pretrained2d (`bool`):
            Load image pretrained weights. Default: True
        patch_size (`tuple[int]`):
            Patch size. Default: (4,4,4).
        in_chans (`int)`:
            Number of input image channels. Default: 3.
        embed_dim (`int`):
            Number of linear projection output channels. Default: 96.
        depths (`tuple[int]`):
            Depths of each Swin Transformer stage.
        num_heads (`tuple[int]`):
            Number of attention head of each stage.
        window_size (`int`):
            Window size. Default: 7.
        mlp_ratio (`float`):
            Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (`bool`):
            If True, add a learnable bias to query, key, value. Default: True
        qk_scale (`float`):
            Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (`float`):
            Dropout rate.
        attn_drop_rate (`float`):
            Attention dropout rate. Default: 0.
        drop_path_rate (`float`):
            Stochastic depth rate. Default: 0.2.
        norm_layer (`nn.Module`):
            Normalization layer. Default: nn.LayerNorm.
        patch_norm (`bool`):
            If True, add normalization after patch embedding. Default: False.
        frozen_stages (`int`):
            Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (`bool`):
            Use checkpoint.
        stride (`tuple[int]`):
            stride size for patch embed3d.
    """

    def __init__(self,
                 pretrained=None,
                 pretrained2d=True,
                 patch_size=(4, 4, 4),
                 in_chans=3,
                 embed_dim=96,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 window_size=(2, 7, 7),
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 patch_norm=False,
                 frozen_stages=-1,
                 use_checkpoint=False,
                 depth_mode=None,
                 depth_patch_embed_separate_params=True,
                 stride=None
                 ):
        super().__init__()

        self.pretrained = pretrained
        self.pretrained2d = pretrained2d
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.frozen_stages = frozen_stages
        self.window_size = window_size
        self.patch_size = patch_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed3D(
            patch_size=patch_size, c=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None, stride=stride)

        if depth_mode is not None:
            assert depth_mode in ["separate_d_tokens", "summed_rgb_d_tokens", "rgbd"]
            if depth_mode in ["separate_d_tokens", "summed_rgb_d_tokens"]:
                depth_chans = 1
                assert (
                    depth_patch_embed_separate_params
                ), "separate tokenization needs separate parameters"
                if depth_mode == "separate_d_tokens":
                    raise NotImplementedError()
            else:
                assert depth_mode == "rgbd"
                depth_chans = 4

            self.depth_patch_embed_separate_params = depth_patch_embed_separate_params

            if depth_patch_embed_separate_params:
                self.depth_patch_embed = PatchEmbed3D(
                    patch_size=patch_size,
                    c=depth_chans,
                    embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None,
                    stride=stride
                )
            else:
                # share parameters with patch_embed
                # delete the layer we built above
                del self.patch_embed
                assert depth_chans == 4
                self.patch_embed = PatchEmbed3D(
                    patch_size=patch_size,
                    c=3,
                    embed_dim=embed_dim,
                    additional_variable_channels=[1],
                    norm_layer=norm_layer if self.patch_norm else None,
                    stride=stride
                )

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = VideoSwinTransformerBlock(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=norm_layer,
                downsample=PatchMerging3D if i_layer < self.num_layers-1 else None,
                use_checkpoint=use_checkpoint)
            self.layers.append(layer)

        self.num_features = int(embed_dim * 2**(self.num_layers-1))

        # add a norm layer for each output
        self.norm = norm_layer(self.num_features)

        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.frozen_stages >= 1:
            self.pos_drop.eval()
            for i in range(0, self.frozen_stages):
                m = self.layers[i]
                m.eval()
                for param in m.parameters():
                    param.requires_grad = False

    def inflate_weights(self, logger):
        """
        Inflate the swin2d parameters to swin3d.
        The differences between swin3d and swin2d mainly lie in an extra
        axis. To utilize the pretrained parameters in 2d model,
        the weight of swin2d models should be inflated to fit in the shapes of
        the 3d counterpart.
        Args:
            logger (`logging.Logger`):
                The logger used to print debugging infomation.
        """
        checkpoint = torch.load(self.pretrained, map_location="cpu")
        state_dict = checkpoint["model"]

        # delete relative_position_index since we always re-init it
        relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
        for k in relative_position_index_keys:
            del state_dict[k]

        # delete attn_mask since we always re-init it
        attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
        for k in attn_mask_keys:
            del state_dict[k]

        state_dict["patch_embed.proj.weight"] =\
            state_dict["patch_embed.proj.weight"].unsqueeze(2).repeat(1, 1, self.patch_size[0], 1, 1) \
            / self.patch_size[0]

        # bicubic interpolate relative_position_bias_table if not match
        relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
        for k in relative_position_bias_table_keys:
            relative_position_bias_table_pretrained = state_dict[k]
            # pylint: disable=E1136
            relative_position_bias_table_current = self.state_dict()[k]
            l1, nh1 = relative_position_bias_table_pretrained.size()
            l2, nh2 = relative_position_bias_table_current.size()
            l2 = (2*self.window_size[1]-1) * (2*self.window_size[2]-1)
            wd = self.window_size[0]
            if nh1 != nh2:
                logger.warning(f"Error in loading {k}, passing")
            else:
                if l1 != l2:
                    s1 = int(l1 ** 0.5)
                    relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
                        relative_position_bias_table_pretrained.permute(1, 0).view(1, nh1, s1, s1),
                        size=(2*self.window_size[1]-1, 2*self.window_size[2]-1),
                        mode="bicubic")
                    relative_position_bias_table_pretrained =\
                        relative_position_bias_table_pretrained_resized.view(nh2, l2).permute(1, 0)
            state_dict[k] = relative_position_bias_table_pretrained.repeat(2*wd-1, 1)

        msg = self.load_state_dict(state_dict, strict=False)
        logger.info(msg)
        logger.info(f"=> loaded successfully '{self.pretrained}'")
        del checkpoint
        torch.cuda.empty_cache()

    def get_patch_embedding(self, x):
        # x: B x C x T x H x W
        assert x.ndim == 5
        has_depth = x.shape[1] == 4

        if has_depth:
            if self.depth_mode in ["summed_rgb_d_tokens"]:
                x_rgb = x[:, :3, ...]
                x_d = x[:, 3:, ...]
                x_d = self.depth_patch_embed(x_d)
                x_rgb = self.patch_embed(x_rgb)
                # sum the two sets of tokens
                x = x_rgb + x_d
            elif self.depth_mode == "rgbd":
                if self.depth_patch_embed_separate_params:
                    x = self.depth_patch_embed(x)
                else:
                    x = self.patch_embed(x)
            else:
                raise NotImplementedError()
        else:
            x = self.patch_embed(x)
        return x

    def forward(self, x):
        x = self.get_patch_embedding(x)

        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x.contiguous())

        x = rearrange(x, "n c d h w -> n d h w c")
        x = self.norm(x)
        x = rearrange(x, "n d h w c -> n c d h w")

        return x
