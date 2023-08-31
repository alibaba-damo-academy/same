import os

import numpy as np
import SAMReg.cores.functionals as sF
import torch
import torch.nn as nn
import torch.nn.functional as F
from SAMReg.cores.functionals import spatial_transformer, compose
from SAMReg.cores.layers import ConvBlock, VecInt

from torch.distributions.normal import Normal

def default_unet_features():
    nb_features = [
        [16, 32, 32, 32],             # encoder
        [32, 32, 32, 32, 32, 16, 16]  # decoder
    ]
    return nb_features

class Unet(nn.Module):
    """
    A unet architecture. Layer features can be specified directly as a list of encoder and decoder
    features or as a single integer along with a number of unet levels. The default network features
    per layer (when no options are specified) are:

        encoder: [16, 32, 32, 32]
        decoder: [32, 32, 32, 32, 32, 16, 16]
    """

    def __init__(self,
                 inshape=None,
                 infeats=None,
                 nb_features=None,
                 nb_levels=None,
                 max_pool=2,
                 feat_mult=1,
                 nb_conv_per_level=1,
                 half_res=False,
                 sam_feature_level=1,
                 sam_feature_dim=256):
        """
        Parameters:
            inshape: Input shape. e.g. (192, 192, 192)
            infeats: Number of input features.
            nb_features: Unet convolutional features. Can be specified via a list of lists with
                the form [[encoder feats], [decoder feats]], or as a single integer. 
                If None (default), the unet features are defined by the default config described in 
                the class documentation.
            nb_levels: Number of levels in unet. Only used when nb_features is an integer. 
                Default is None.
            feat_mult: Per-level feature multiplier. Only used when nb_features is an integer. 
                Default is 1.
            nb_conv_per_level: Number of convolutions per unet level. Default is 1.
            half_res: Skip the last decoder upsampling. Default is False.
        """

        super().__init__()

        # ensure correct dimensionality
        ndims = len(inshape)
        assert ndims in [1, 2, 3], 'ndims should be one of 1, 2, or 3. found: %d' % ndims

        # cache some parameters
        self.half_res = half_res

        # default encoder and decoder layer features if nothing provided
        if nb_features is None:
            nb_features = default_unet_features()

        # build feature list automatically
        if isinstance(nb_features, int):
            if nb_levels is None:
                raise ValueError('must provide unet nb_levels if nb_features is an integer')
            feats = np.round(nb_features * feat_mult ** np.arange(nb_levels)).astype(int)
            nb_features = [
                np.repeat(feats[:-1], nb_conv_per_level),
                np.repeat(np.flip(feats), nb_conv_per_level)
            ]
        elif nb_levels is not None:
            raise ValueError('cannot use nb_levels if nb_features is not an integer')

        # extract any surplus (full resolution) decoder convolutions
        enc_nf, dec_nf = nb_features
        nb_dec_convs = len(enc_nf)
        final_convs = dec_nf[nb_dec_convs:]
        dec_nf = dec_nf[:nb_dec_convs]
        self.nb_levels = int(nb_dec_convs / nb_conv_per_level) + 1

        if isinstance(max_pool, int):
            max_pool = [max_pool] * self.nb_levels

        # cache downsampling / upsampling operations
        MaxPooling = getattr(nn, 'MaxPool%dd' % ndims)
        self.pooling = [MaxPooling(s) for s in max_pool]
        self.upsampling = [nn.Upsample(scale_factor=s, mode='nearest') for s in max_pool]

        # configure encoder (down-sampling path)
        prev_nf = infeats
        encoder_nfs = [prev_nf]
        self.encoder = nn.ModuleList()
        add_sam = False
        for level in range(self.nb_levels - 1):
            if level == sam_feature_level:
                add_sam = True
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = enc_nf[level * nb_conv_per_level + conv]
                if add_sam and conv==0:
                    prev_nf += sam_feature_dim
                    add_sam = False
                convs.append(ConvBlock(prev_nf, nf, ndims=ndims))
                prev_nf = nf
            self.encoder.append(convs)
            encoder_nfs.append(prev_nf)

        # configure decoder (up-sampling path)
        encoder_nfs = np.flip(encoder_nfs)
        self.decoder = nn.ModuleList()
        
        for level in range(self.nb_levels - 1):
            convs = nn.ModuleList()
            for conv in range(nb_conv_per_level):
                nf = dec_nf[level * nb_conv_per_level + conv]
                convs.append(ConvBlock(prev_nf, nf, ndims=ndims))
                prev_nf = nf
            self.decoder.append(convs)
            if not half_res or level < (self.nb_levels - 2):
                prev_nf += encoder_nfs[level]

        # now we take care of any remaining convolutions
        self.remaining = nn.ModuleList()
        for num, nf in enumerate(final_convs):
            self.remaining.append(ConvBlock(prev_nf, nf, ndims=ndims))
            prev_nf = nf

        # cache final number of features
        self.final_nf = prev_nf

        self.sam_feature_level = sam_feature_level


    def forward(self, x, features):

        # encoder forward pass
        x_history = [x]
        for level, convs in enumerate(self.encoder):
            if level == self.sam_feature_level:
                x = torch.cat([x, features], dim=1)
            for conv in convs:
                x = conv(x)
            x_history.append(x)
            x = self.pooling[level](x)

        # decoder forward pass with upsampling and concatenation
        for level, convs in enumerate(self.decoder):
            for conv in convs:
                x = conv(x)
            if not self.half_res or level < (self.nb_levels - 2):
                x = self.upsampling[level](x)
                x = torch.cat([x, x_history.pop()], dim=1)

        # remaining convs at full resolution
        for conv in self.remaining:
            x = conv(x)

        return x


class VoxelMorph(nn.Module):
    def __init__(self, inshape, embed,
                 nb_unet_features=None,
                 nb_unet_levels=None,
                 unet_feat_mult=1,
                 nb_unet_conv_per_level=1,
                 src_feats=1,
                 trg_feats=1,
                 unet_half_res=False,
                 sam_feature_level=1,
                 sam_feature_dim=256,
                 sam_corr=None,
                 integrate_step=1):
        super(VoxelMorph, self).__init__()
        
        self.embed = embed
        ndims = len(inshape)
        self.inshape = inshape
        self.sam_feature_level = sam_feature_level
        self.sam_corr = sam_corr
        self.register_buffer("identity", 
                F.affine_grid(
                    torch.eye(3, 4).unsqueeze(0),
                    size=(1,1) + tuple(inshape),
                    align_corners=True,
                ).permute(0,4,1,2,3).flip(1), persistent=False)
        
        # configure core unet model
        if sam_corr is not None:
            # If we are using correlation layer, then the input dimension 
            # would be the kernal volume.
            sam_feature_dim = (sam_corr.kernel_size*2+1)**3
        else:
            # If we do not use correlation layer, then we are concatenating
            # the coarse and fine feature from source and target. Thus, we
            # double the dimension of the input feature.
            sam_feature_dim *= 2

        self.unet_model = Unet(
            inshape,
            infeats=(src_feats + trg_feats),
            nb_features=nb_unet_features,
            nb_levels=nb_unet_levels,
            feat_mult=unet_feat_mult,
            nb_conv_per_level=nb_unet_conv_per_level,
            half_res=unet_half_res,
            sam_feature_level=sam_feature_level,
            sam_feature_dim=sam_feature_dim
        )

        # configure unet to flow field layer
        Conv = getattr(nn, 'Conv%dd' % ndims)
        self.flow = Conv(self.unet_model.final_nf, ndims, kernel_size=3, padding=1)

        # init flow layer with small weights and bias
        self.flow.weight = nn.Parameter(Normal(0, 1e-5).sample(self.flow.weight.shape))
        # self.flow.weight = nn.Parameter(torch.zeros(self.flow.weight.shape))
        self.flow.bias = nn.Parameter(torch.zeros(self.flow.bias.shape))

        if integrate_step > 1:
            self.integrate = VecInt(integrate_step)
        else:
            self.integrate = None
        
    
    def forward(self, source, target, source_label=None, pre_align=None, registration=False):
        '''
        Parameters:
            source: Source image tensor. Bx2xDxWxH. The firsts chanel is for SAM and second is for the registration network.
            target: Target image tensor. Bx2xDxWxH. The firsts chanel is for SAM and second is for the registration network.
            registration: Return transformed image and flow. Default is False.
        '''

        # If pre_align is available
        with torch.no_grad():
            if pre_align is not None:
                pre_align = F.interpolate(
                        pre_align, 
                        size=source.shape[2:], mode='trilinear', align_corners=True)
                source_  = spatial_transformer(
                    source, 
                    pre_align, mode='bilinear', padding_mode="background")
            else:
                source_ = source

            if self.sam_feature_level >= 0:
                # compute SAM cost volume
                source_SAM = self.embed.extract_feat(source_[:, :1])
                target_SAM = self.embed.extract_feat(target[:, :1])

                if self.sam_corr is not None:
                    # we should compute the correlation
                    sam_feat = self.sam_corr(source_SAM[0], target_SAM[0]) + F.interpolate(self.sam_corr(source_SAM[1], target_SAM[1]), size=source_SAM[0].shape[2:], mode="trilinear", align_corners=True)
                    sam_feat = sam_feat.float()
                else:
                    # source_SAM_scaled = (
                    #         source_SAM[0] + F.normalize(F.interpolate(source_SAM[1], size=source_SAM[0].shape[2:], mode="trilinear", align_corners=True), dim=1)).float()
                    # target_SAM_scaled = (
                    #         target_SAM[0] + F.normalize(F.interpolate(target_SAM[1], size=target_SAM[0].shape[2:], mode="trilinear", align_corners=True), dim=1)).float()
                    # sam_feat = torch.cat([source_SAM_scaled, target_SAM_scaled], dim=1)
                    source_SAM_coarse = F.normalize(F.interpolate(source_SAM[1], size=source_SAM[0].shape[2:], mode="trilinear", align_corners=True), dim=1).float()
                    target_SAM_coarse = F.normalize(F.interpolate(target_SAM[1], size=target_SAM[0].shape[2:], mode="trilinear", align_corners=True), dim=1).float()
                    sam_feat = torch.cat([source_SAM[0], source_SAM_coarse, target_SAM[0], target_SAM_coarse], dim=1)

        # concatenate inputs and propagate unet
        x = torch.cat([source_[:, 1:2], target[:, 1:2]], dim=1)
        
        if self.sam_feature_level >= 0:
            x = self.unet_model(x, sam_feat)
        else:
            x = self.unet_model(x, None)

        # transform into flow field
        flow_field = self.flow(x)
        if self.integrate is not None:
            post_field = self.integrate(flow_field)
        else:
            post_field = flow_field

        if pre_align is not None:
            phi = self.identity + compose(pre_align-self.identity, post_field)
        else:
            phi = self.identity + post_field

        # warp image with flow field
        y_source = spatial_transformer(source[:, 1:2], phi, padding_mode="background")

        if self.sam_feature_level >= 0:
            phi_scaled = F.interpolate(self.identity + post_field, size=source_SAM[0].shape[2:], mode='trilinear', align_corners=True)
            y_source_SAM = [
                    spatial_transformer(source_SAM[0].float(), phi_scaled, padding_mode="border").half(),
                    spatial_transformer(source_SAM[1].float(), phi_scaled, padding_mode="border").half()
                ]
            target_SAM[1] = F.interpolate(target_SAM[1], size=source_SAM[0].shape[2:], mode='trilinear', align_corners=True)

        if source_label is not None:
            warped_label  = spatial_transformer(
                    F.one_hot(source_label.long())[:,0].permute(0,4,1,2,3).float(), 
                    phi)

        # return non-integrated flow field if training
        if not registration:
            res = [y_source]
            if self.sam_feature_level > 0:
                res.append(y_source_SAM)
                res.append(target_SAM)
            if source_label is not None:
                res.append(warped_label)
            res.append(flow_field)
            return tuple(res)
        else:
            # If not in registraiton mode, return the composed phi (including pre-align if it exist)
            return y_source, phi
