import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return x

    def _initialize_weights(self, clip_model, i):
        self.ln_1 = clip_model.visual.transformer.resblocks[i].ln_1.to(torch.float32)
        self.attn = clip_model.visual.transformer.resblocks[i].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[i].mlp.to(torch.float32)
        self.ln_2 = clip_model.visual.transformer.resblocks[i].ln_2.to(torch.float32)
        for p in self.parameters():
            p.requires_grad = False

class LastResidualAttentionBlock(nn.Module):
    def __init__(self, clip_model, d_model, n_head):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, batch_first=True)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", nn.GELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = nn.LayerNorm(d_model)
        self._initialize_weights(clip_model)

    def forward(self, x: torch.Tensor):
        y = self.ln_1(x)
        y_proj = F.linear(y, self.attn.in_proj_weight, self.attn.in_proj_bias)
        N, L, C = y_proj.shape
        y_proj = y_proj.view(N, L, 3, C // 3).permute(2, 0, 1, 3).reshape(3 * N, L, C // 3)
        y_proj = F.linear(y_proj, self.attn.out_proj.weight, self.attn.out_proj.bias)
        q, k, v = y_proj.tensor_split(3, dim=0)

        v += x
        v = v + self.mlp(self.ln_2(v))

        x = x + self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln_2(x))
        return [x, q, k, v]

    def _initialize_weights(self, clip_model):
        self.ln_1 = clip_model.visual.transformer.resblocks[11].ln_1.to(torch.float32)
        self.attn = clip_model.visual.transformer.resblocks[11].attn.to(torch.float32)
        self.attn.batch_first = True
        self.mlp = clip_model.visual.transformer.resblocks[11].mlp.to(torch.float32)
        self.ln_2 = clip_model.visual.transformer.resblocks[11].ln_2.to(torch.float32)
        for p in self.parameters():
            p.requires_grad = False

class VisionTransformer(nn.Module):
    def __init__(self, clip_model, input_resolution=224, patch_size=16, width=768, layers=12, heads=12):
        super().__init__()
        self.patch_size = patch_size
        self.input_resolution = input_resolution
        self.conv1 = clip_model.visual.conv1.to(torch.float32)
        self.cls_token = torch.load('utils/cls_token.pt', map_location='cpu').to(torch.float32)
        self.positional_embedding = nn.Parameter(clip_model.visual.positional_embedding.to(torch.float32))
        self.ln_pre = clip_model.visual.ln_pre.to(torch.float32)

        resblocks = []
        for i in range(layers - 1):
            blk = ResidualAttentionBlock(width, heads)
            blk._initialize_weights(clip_model, i)
            resblocks.append(blk)

        last_blk = LastResidualAttentionBlock(clip_model, width, heads)
        resblocks.append(last_blk)

        self.transformer = nn.Sequential(*resblocks)
        self.ln_post = clip_model.visual.ln_post.to(torch.float32)
        self.proj = clip_model.visual.proj.to(torch.float32)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        input_h, input_w = x.size()[-2:]
        output_h, output_w = math.ceil(input_h / self.patch_size), math.ceil(input_w / self.patch_size)

        x = F.pad(x, [0, max(output_w * self.patch_size - input_w, 0), 0, max(output_h * self.patch_size - input_h, 0)])
        x = self.conv1(x).flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1).to(x.device, dtype=torch.float32)
        x = torch.cat((cls_tokens, x), dim=1)

        pos_h = pos_w = self.input_resolution // self.patch_size
        pos_embed_weight = self.positional_embedding[1:].reshape(1, pos_h, pos_w, -1).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight, size=(output_h, output_w), mode='bicubic', align_corners=False)
        pos_embed_weight = pos_embed_weight.flatten(2).transpose(1, 2)
        pos_embedding = torch.cat((self.positional_embedding[0:1].unsqueeze(0), pos_embed_weight), dim=1)

        x = self.ln_pre(x + pos_embedding)

        for i, blk in enumerate(self.transformer):
            res = blk(x)
            if isinstance(res, list):
                x, q, k, v = res
            else:
                x = res

        v = self.ln_post(v)
        v_2d = v[:, 1:].reshape(B, output_h, output_w, -1).permute(0, 3, 1, 2).contiguous()
        return v_2d, (output_h, output_w), (x[:, 0] @ self.proj), k, pos_embedding[:, 1:]

class TextEncoder(nn.Module):
    def __init__(self, clip_model, training=False, cfg=None, device=None):
        super().__init__()
        self.transformer = clip_model.transformer.to(torch.float32)
        self.token_embedding = clip_model.token_embedding.to(torch.float32)
        self.positional_embedding = clip_model.positional_embedding.to(torch.float32)
        self.ln_final = clip_model.ln_final.to(torch.float32)
        self.text_projection = clip_model.text_projection.to(torch.float32)
        self.device = device

        for p in self.parameters():
            p.requires_grad = False

        if training:
            token = torch.zeros((1, 73), dtype=torch.int).to(self.device)
            self.prompt_token = nn.Parameter(self.token_embedding(token).detach().clone())
            self.prompt_token.requires_grad = True
        else:
            ckpt = torch.load(cfg.LOAD_PATH, map_location="cpu")
            prompt_weight = ckpt.get('module.text_encoder.prompt_token', ckpt.get('prompt_token'))
            self.prompt_token = nn.Parameter(prompt_weight.to(torch.float32), requires_grad=False)

    def forward(self, cls_name_token):
        B = cls_name_token.shape[0]
        prompt_token = self.prompt_token.repeat(B, 1, 1).to(self.device)
        start_token = self.token_embedding(torch.tensor(49406, device=self.device)).repeat(B, 1, 1)
        cls_token = self.token_embedding(cls_name_token.to(self.device))
        x = torch.cat([start_token, prompt_token, cls_token], dim=1)
        x = (x + self.positional_embedding.type(torch.float32)).permute(1, 0, 2)
        x = self.transformer(x).permute(1, 0, 2)
        x = self.ln_final(x).type(torch.float32)
        return x[torch.arange(B), 74 + cls_name_token.argmax(dim=-1)] @ self.text_projection

class MSFAModule(nn.Module):
    def __init__(self, channels=512):
        super(MSFAModule, self).__init__()

        self.high_freq_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels, bias=False)
        laplacian = torch.tensor([[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]])
        self.high_freq_conv.weight.data = laplacian.view(1, 1, 3, 3).repeat(channels, 1, 1, 1)
        self.high_freq_conv.weight.requires_grad = False


        self.spectrum_gate = nn.Sequential(
            nn.Conv2d(channels, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.spectrum_gate[2].weight, 0)

    def forward(self, x):
        feat_hf = self.high_freq_conv(x)
        gate = self.spectrum_gate(x)
        return feat_hf, gate

class SAPModule(nn.Module):
    def __init__(self, threshold=0.05):
        super(SAPModule, self).__init__()
        self.threshold = threshold

    def forward(self, x):
        B, C_f, H_f, W_f = x.shape
        feat_flat = x.view(B, C_f, -1)

        affinity = torch.bmm(feat_flat.transpose(1, 2), feat_flat) / (C_f ** 0.5)
        affinity = torch.where(affinity < self.threshold, torch.tensor(-1e9).to(x.device), affinity)
        affinity_weights = F.softmax(affinity, dim=-1)


        feat_sap = torch.bmm(feat_flat, affinity_weights.transpose(1, 2)).view(B, C_f, H_f, W_f)
        return feat_sap


class FASGSeg(nn.Module):
    def __init__(self, cfg, clip_model, rank):
        super(FASGSeg, self).__init__()
        self.device = rank
        self.cnum = cfg.DATASET.NUM_CLASSES
        self.logit_scale = clip_model.logit_scale

        self.vit = VisionTransformer(clip_model=clip_model, input_resolution=224, patch_size=16, width=768, layers=12,
                                     heads=12)
        visual_channel = cfg.MODEL.VISUAL_CHANNEL
        text_channel = cfg.MODEL.TEXT_CHANNEL
        self.proj = nn.Conv2d(visual_channel, text_channel, 1, bias=False)
        self.proj.weight = nn.Parameter(clip_model.visual.proj[:, :, None, None].permute(1, 0, 2, 3).to(torch.float32),
                                        requires_grad=False)


        self.text_encoder = TextEncoder(clip_model, training=cfg.MODEL.TRAINING, cfg=cfg, device=rank)
        self.pe_proj = nn.Conv2d(768, 512, kernel_size=1)

        for p in self.parameters():
            p.requires_grad = False


        self.msfa = MSFAModule(channels=512)
        self.sap = SAPModule(threshold=0.05)

        if cfg.MODEL.TRAINING:
            self.decoder_conv2 = nn.Conv2d(512 + self.cnum, self.cnum, kernel_size=5, padding=2, stride=1)
            nn.init.kaiming_normal_(self.decoder_conv2.weight, a=0, mode='fan_out', nonlinearity='relu')
            self.decoder_norm2 = nn.BatchNorm2d(self.cnum)
            nn.init.constant_(self.decoder_norm2.weight, 1)
            nn.init.constant_(self.decoder_norm2.bias, 0)
        else:
            self.decoder_conv2 = nn.Conv2d(self.cnum + 512, self.cnum, kernel_size=5, padding=2, stride=1)
            self.decoder_norm2 = nn.BatchNorm2d(self.cnum)

    def forward(self, image, pseudo_labels, base_text_features, cls_name_token=None, training=False, img_metas=None,
                use_msfa=True, use_sap=True, **kwargs):
        device = self.device
        gt_text_embeddings = base_text_features.to(device)
        batch_size = image.shape[0]
        image = image.to(device)


        v, shape, _, _, positional_embedding = self.vit(image)
        positional_embedding = positional_embedding.reshape(1, shape[0], shape[1], -1).permute(0, 3, 1, 2)
        feat_raw = F.normalize(self.proj(v), dim=1)


        global_feat = feat_raw.mean(dim=(2, 3), keepdim=True)
        feat_raw = F.normalize(feat_raw + 0.1 * global_feat, dim=1)


        feat_hf, gate = self.msfa(feat_raw)
        feat_sap = self.sap(feat_raw)

        feat_fused = feat_raw.clone()
        if use_msfa:
            feat_fused = feat_fused + (gate * feat_hf * 0.5)
        if use_sap:
            feat_fused = feat_fused + (feat_sap * 0.2)
        feat = F.normalize(feat_fused, dim=1)


        output_q = F.conv2d(feat, gt_text_embeddings[:, :, None, None]).permute(0, 2, 3, 1).reshape(batch_size, -1,
                                                                                                    self.cnum)
        prompt = F.normalize(self.text_encoder(cls_name_token), dim=1)
        pe = self.pe_proj(positional_embedding).permute(0, 2, 3, 1).reshape(1, shape[0] * shape[1], -1)
        bias_logits = pe @ prompt.t()
        output_base = torch.sub(output_q, bias_logits).permute(0, 2, 1).reshape(batch_size, -1, shape[0], shape[1])


        feature = torch.cat((feat, output_base), dim=1)
        feature = self.decoder_conv2(feature)
        output = self.decoder_norm2(feature)

        if not training:
            return output


        logit_scale = self.logit_scale.exp()
        gate_resized = F.interpolate(gate, size=(shape[0], shape[1]), mode='bilinear', align_corners=False)
        gate_flat = gate_resized.reshape(batch_size, 1, -1).permute(0, 2, 1)

        dynamic_scale = 100.0 * (1.0 + gate_flat)
        output_scale = output.reshape(batch_size, self.cnum, -1).permute(0, 2, 1) * dynamic_scale
        output_gumbel = F.gumbel_softmax(output_scale, tau=1, hard=True, dim=2).reshape(batch_size, shape[0], shape[1],
                                                                                        -1)

        loss = 0
        for j in range(batch_size):
            if len(pseudo_labels[j]) == 0: continue
            masked_image_features = []
            for i in pseudo_labels[j]:
                mask = output_gumbel[j, :, :, i].unsqueeze(dim=0)
                masked_image_feature = torch.mul(feat[j].unsqueeze(dim=0), mask)
                feature_pool = nn.AdaptiveAvgPool2d((1, 1))(masked_image_feature).reshape(1, 512)
                masked_image_features.append(feature_pool)

            masked_image_features = torch.stack(masked_image_features, dim=0).squeeze(dim=1)
            similarity_img = logit_scale * masked_image_features @ gt_text_embeddings.t()
            labels = torch.tensor(pseudo_labels[j]).to(device)
            loss += F.cross_entropy(similarity_img, labels)

        loss_bias_reg = torch.mean(bias_logits ** 2)
        return output, (loss / batch_size) + 0.1 * loss_bias_reg