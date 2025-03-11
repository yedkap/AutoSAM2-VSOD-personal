from models.hardnet import HarDNet
from models.base import *
from models.image_encoder_RGBD import ImageEncoderRGB_D


class SmallDecoder(nn.Module):
    def __init__(self, full_features, out):
        super(SmallDecoder, self).__init__()
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0)
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0)
        self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)

    def forward(self, x):
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        out = F.tanh(self.final(z))
        # out = self.final(z)
        return out


class SmallDecoderSimpleDepth(nn.Module):
    def __init__(self, full_features, out):
        super(SmallDecoderSimpleDepth, self).__init__()
        self.reduction_layers = nn.ModuleList([
            nn.Conv2d(full_features[ii] * 2, full_features[ii], kernel_size=1, stride=1, bias=False)
            for ii in (1, 2, 3)
        ])
        self.up1 = UpBlockSkip(full_features[3] + full_features[2], full_features[2],
                               func='relu', drop=0)
        self.up2 = UpBlockSkip(full_features[2] + full_features[1], full_features[1],
                               func='relu', drop=0)
        self.final = CNNBlock(full_features[1], out, kernel_size=3, drop=0)

    def forward(self, x):
        for ii, idx in enumerate((1, 2, 3)):
            x[idx] = self.reduction_layers[ii](x[idx])
        z = self.up1(x[3], x[2])
        z = self.up2(z, x[1])
        out = F.tanh(self.final(z))
        # out = self.final(z)
        return out


class ModelEmb(nn.Module):
    def __init__(self, args, size_out=64, train_decoder_only=False):
        super(ModelEmb, self).__init__()
        print('using HarDNet backbone')
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = SmallDecoder(d, out=256)
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.size_out = size_out
        self.train_decoder_only = train_decoder_only

    def forward(self, img):
        if self.train_decoder_only:
            with torch.no_grad():
                z = self.backbone(img)
        else:
            z = self.backbone(img)
        dense_embeddings = self.decoder(z)
        dense_embeddings = F.interpolate(dense_embeddings, (self.size_out, self.size_out), mode='bilinear', align_corners=True)
        return dense_embeddings


class ModelEmbSimpleDepth(nn.Module):
    def __init__(self, args, size_out=64, train_decoder_only=False):
        super(ModelEmbSimpleDepth, self).__init__()
        print('using simple depth integration as greyscale input')
        print('using HarDNet backbone')
        # self.depth_conv = nn.Conv2d(1, 3, kernel_size=1, stride=1, bias=False)
        self.backbone = HarDNet(depth_wise=bool(int(args['depth_wise'])), arch=int(args['order']), args=args)
        d, f = self.backbone.full_features, self.backbone.features
        self.decoder = SmallDecoderSimpleDepth(d, out=256)
        for param in self.backbone.parameters():
            param.requires_grad = True
        self.size_out = size_out
        self.train_decoder_only = train_decoder_only

    def forward(self, img, depth_image):
        depth_input = depth_image.repeat(1, 3, 1, 1)
        # depth_input = self.depth_conv(depth_image)
        if self.train_decoder_only:
            with torch.no_grad():
                z_img = self.backbone(img)
                z_depth = self.backbone(depth_input)
        else:
            z_img = self.backbone(depth_input)
            z_depth = self.backbone(depth_image)
        z = [torch.cat((z_img_res, z_depth_res), dim=1) for z_img_res, z_depth_res in zip(z_img, z_depth)]
        dense_embeddings = self.decoder(z)
        dense_embeddings = F.interpolate(dense_embeddings, (self.size_out, self.size_out), mode='bilinear', align_corners=True)
        return dense_embeddings


class ModelEmbESA(nn.Module):
    def __init__(self, args,size_out=64, train_decoder_only=None)
        super(ModelEmbESA, self).__init__()
        print('using ESA RGBD backbone')
        self.backbone = ImageEncoderRGB_D(
            pretrained=True
        )

        d = [1,self.backbone.encoder_rgb.down_4_channels_out ,self.backbone.encoder_rgb.down_8_channels_out,self.backbone.encoder_rgb.down_16_channels_out]

        self.decoder = SmallDecoder(d, out=256)
        #for param in self.backbone.parameters(): #fix these
            #param.requires_grad = True
        self.train_decoder_only = train_decoder_only

    def forward(self, img, depth_image, size=None):
        # Convert RGB to grayscale using the luminance formula
        #gray_img = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
        #gray_img = (gray_img - gray_img.min()) / (gray_img.max() - gray_img.min())
        #gray_img = gray_img.unsqueeze(1)
        #print(gray_img.shape)
        if self.train_decoder_only:
            with torch.no_grad():
                z = self.backbone(img, depth_image)
        else:
            z = self.backbone(img, depth_image)
        dense_embeddings = self.decoder(z)
        dense_embeddings = F.interpolate(dense_embeddings, (64, 64), mode='bilinear', align_corners=True)
        return dense_embeddings



class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class ModelH(nn.Module):
    def __init__(self):
        super(ModelH, self).__init__()
        self.conv1 = nn.ConvTranspose2d(256, 64, 3, stride=2, padding=1)
        self.norm1 = LayerNorm2d(64)
        self.gelu = nn.GELU()
        self.conv2 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.norm2 = LayerNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, mask):
        z = self.conv1(mask, output_size=(128, 128))
        z = self.norm1(z)
        z = self.gelu(z)
        z = self.conv2(z, output_size=(256, 256))
        z = self.norm2(z)
        z = self.gelu(z)
        z = self.conv3(z)
        return z
