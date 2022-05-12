# original code from https://github.com/SwinTransformer/Video-Swin-Transformer
# modified by Zilliz.
import torch
try:
    import torchvision
except ModuleNotFoundError:
    os.system("pip install torchvision")
    import torchvision
try:
    import transformers
except ModuleNotFoundError:
    os.system("pip install transformers")
    import transformers
from towhee.models.video_swin_transformer.video_swin_transformer import VideoSwinTransformer
# from video_swin import SwinTransformer3D


class EncImg(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.swin = VideoSwinTransformer(patch_size=(2, 4, 4), patch_norm=True, depths=[2, 2, 18, 2],
                                         window_size=(8, 7, 7), num_heads=[3, 6, 12, 24], mlp_ratio=4.,
                                         drop_path_rate=0.2, stride=(1, 4, 4))
        #self.swin = SwinTransformer3D()
        # self.swin.load_state_dict(torch.load('./_snapshot/ckpt_video-swin.pt', map_location='cpu'))

        self.emb_cls = torch.nn.Parameter(0.02 * torch.randn(1, 1, 1, 768))
        self.emb_pos = torch.nn.Parameter(0.02 * torch.randn(1, 1, 1 + 14 ** 2, 768))
        self.emb_len = torch.nn.Parameter(0.02 * torch.randn(1, 6, 1, 768))
        self.norm = torch.nn.LayerNorm(768)

    def forward(self, img):
        _B, _T, _C, _H, _W = img.shape
        _h, _w = _H // 32, _W // 32

        img = torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                      [0.229, 0.224, 0.225])(img)

        f_img = self.swin(img.transpose(1, 2)).transpose(1, 2)

        f_img = f_img.permute(0, 1, 3, 4, 2).view([_B, _T, _h * _w, 768])
        f_img = torch.cat([self.emb_cls.expand([_B, _T, -1, -1]), f_img], dim=2)
        f_img += self.emb_pos.expand([_B, _T, -1, -1])[:, :, :1 + _h * _w, :] + self.emb_len.expand(
            [_B, -1, 1 + _h * _w, -1])[:, :_T, :, :]
        f_img = self.norm(f_img).view([_B, _T * (1 + _h * _w), -1])

        m_img = torch.ones(1 + _h * _w).long().unsqueeze(0).unsqueeze(0)
        m_img = m_img.expand([_B, _T, -1]).contiguous().view([_B, _T * (1 + _h * _w)])

        return f_img, m_img


class EncTxt(torch.nn.Module):
    def __init__(self):
        super().__init__()

        bert = transformers.BertModel.from_pretrained('bert-base-uncased')
        self.emb_txt = bert.embeddings

    def forward(self, txt):
        f_txt = self.emb_txt(txt)

        return f_txt


class VioletBase(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.enc_img, self.enc_txt = EncImg(), EncTxt()
        bert = transformers.BertForMaskedLM.from_pretrained('bert-base-uncased')
        self.mask_ext, self.trsfr = bert.get_extended_attention_mask, bert.bert.encoder

    def go_feat(self, img, txt, mask):
        feat_img, mask_img = self.enc_img(img)
        feat_txt, mask_txt = self.enc_txt(txt), mask
        return feat_img, mask_img, feat_txt, mask_txt

    def go_cross(self, feat_img, mask_img, feat_txt, mask_txt):
        feat, mask = torch.cat([feat_img, feat_txt], dim=1), torchvision.cat([mask_img, mask_txt], dim=1)
        mask = self.mask_ext(mask, mask.shape, mask.device)
        out = self.trsfr(feat, mask, output_attentions=True)
        return out['last_hidden_state'], out['attentions']

    def load_ckpt(self, ckpt):
        if ckpt == '':
            print('===== Init VIOLET =====')
            return

        ckpt_new, ckpt_old = torch.load(ckpt, map_location='cpu'), self.state_dict()
        key_old = set(ckpt_old.keys())
        for k in ckpt_new:
            if k in ckpt_old and ckpt_new[k].shape == ckpt_old[k].shape:
                ckpt_old[k] = ckpt_new[k]
                key_old.remove(k)
        self.load_state_dict(ckpt_old)
        print('===== Not Load:', key_old, '=====')
