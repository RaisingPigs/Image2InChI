import math
import numpy as np
import timm
from torch import nn
from typing import Dict, Optional
from fairseq.models import FairseqIncrementalDecoder
from fairseq.modules import TransformerDecoderLayer
import torch
from torch import Tensor


class Namespace(object):
    def __init__(self, adict):
        self.__dict__.update(adict)


class TransformerDecode(FairseqIncrementalDecoder):
    def __init__(self, dim, ff_dim, num_head, num_layer):
        super().__init__({})

        self.layer = nn.ModuleList([
            TransformerDecoderLayer(Namespace({
                'decoder_embed_dim': dim,
                'decoder_attention_heads': num_head,
                'attention_dropout': 0.1,
                'dropout': 0.1,
                'decoder_normalize_before': True,
                'decoder_ffn_embed_dim': ff_dim,
                # 'decoder_learned_pos': True,
                # 'cross_self_attention': True,
                # 'activation-fn': 'gelu',
            })) for i in range(num_layer)
        ])
        self.layer_norm = nn.LayerNorm(dim)

    def forward(self, x, mem, x_mask):
        # print('my TransformerDecode forward()')
        for layer in self.layer:
            x = layer(x, mem, self_attn_mask=x_mask)[0]
        x = self.layer_norm(x)
        return x  # T x B x C

    # def forward_one(self, x, mem, incremental_state):
    def forward_one(self,
                    x: torch.Tensor,
                    mem: torch.Tensor,
                    incremental_state: Optional[Dict[str, Dict[str, Optional[torch.Tensor]]]]
                    ) -> torch.Tensor:
        x = x[-1:]
        for layer in self.layer:
            x = layer(x, mem, incremental_state=incremental_state)[0]
        x = self.layer_norm(x)
        return x


class PositionEncode1D(torch.nn.Module):
    def __init__(self, dim, max_length):
        super().__init__()
        assert (dim % 2 == 0)
        self.max_length = max_length

        d = torch.exp(torch.arange(0., dim, 2) * (-math.log(10000.0) / dim))
        position = torch.arange(0., max_length).unsqueeze(1)
        pos = torch.zeros(1, max_length, dim)
        pos[0, :, 0::2] = torch.sin(position * d)
        pos[0, :, 1::2] = torch.cos(position * d)
        self.register_buffer('pos', pos)

    def forward(self, x):
        batch_size, T, dim = x.shape
        x = x + self.pos[:, :T]
        return x


class Encoder(nn.Module):
    """编码器类
    这里是使用 tit 作为编码器
    """

    def __init__(self, encoder_dim):
        super(Encoder, self).__init__()

        # 生成网络加载预训练权重
        self.swin_t = timm.create_model('swin_small_patch4_window7_224')
        self.swin_t.head = nn.Identity()
        # 因为这里的swin_t初始的conv的out_channels=128, 所以计算到最后c为1024
        # print(self.swin_t)
        self.fc = nn.Linear(768, encoder_dim)

    def forward(self, x):
        """
        :param x: shape=[b,7,7,1024] 
        """
        x = self.swin_t(x)

        b, h, w, c = x.shape
        x = x.reshape(b, -1, c)
        return self.fc(x)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_length,
                 num_head=8, ff_dim=1024, num_layer=3):
        super().__init__()

        self.embed_dim = embed_dim

        # 解码器
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.pos = PositionEncode1D(embed_dim, max_length)

        self.text_decode = TransformerDecode(embed_dim, ff_dim, num_head, num_layer)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, encoder_out, seq):
        """训练阶段：前向传播
        """
        device = seq.device
        b = seq.size(0)
        encoder_out = encoder_out.permute(1, 0, 2).contiguous()

        text_embed = self.embed(seq)
        text_embed = self.pos(text_embed).permute(1, 0, 2).contiguous()

        text_mask = np.triu(np.ones((seq.size(-1), seq.size(-1))), k=1).astype(np.uint8)
        text_mask = torch.autograd.Variable(torch.from_numpy(text_mask) == 1).to(device)

        #
        x = self.text_decode(text_embed, encoder_out, text_mask)
        x = x.permute(1, 0, 2).contiguous()

        return self.fc(x)

    def predict(self, encoder_out, max_len, start_token, end_token, pad_token):
        """预测阶段：前向传播
        """
        device = encoder_out.device
        batch_size = len(encoder_out)

        # 同上
        image_embed = encoder_out.permute(1, 0, 2).contiguous()

        # b*n 填充 <pad>
        token = torch.full((batch_size, max_len), pad_token, dtype=torch.long).to(device)
        # 获取输出向量的位置向量
        text_pos = self.pos.pos
        # 第一个设置为 <sos>
        token[:, 0] = start_token

        incremental_state = torch.jit.annotate(
            Dict[str, Dict[str, Optional[Tensor]]],
            torch.jit.annotate(Dict[str, Dict[str, Optional[Tensor]]], {}),
        )
        for t in range(max_len - 1):
            # 目前 token 的最后一个值
            last_token = token[:, t]
            # 向量嵌入
            text_embed = self.embed(last_token)
            # 加上位置向量
            text_embed = text_embed + text_pos[:, t]  #
            # b*text_dim -> 1*b*text_dim
            text_embed = text_embed.reshape(1, batch_size, self.embed_dim)
            # 得到下一个向量 1*b*text_dim
            x = self.text_decode.forward_one(text_embed, image_embed, incremental_state)
            # b*text_dim
            x = x.reshape(batch_size, self.embed_dim)
            # b*vocab_size
            l = self.fc(x)
            # 以最大的作为预测
            k = torch.argmax(l, -1)
            token[:, t + 1] = k

            # 遇到 <eos> 和 <pad> 停止预测
            if ((k == end_token) | (k == pad_token)).all():
                break

        # 返回除了 <sos> 之外的序列
        predict = token[:, 1:]
        return predict


class InCHImgAnalyzer(nn.Module):
    def __init__(self, encoder_dim, vocab_size, embed_dim, max_length,
                 num_head=8, ff_dim=1024, num_layer=3):
        super().__init__()
        self.encoder = Encoder(encoder_dim)
        self.decoder = Decoder(
            vocab_size, embed_dim, max_length,
            num_head, ff_dim, num_layer
        )

    def forward(self, img, seq):
        encoder_out = self.encoder(img)
        return self.decoder(encoder_out, seq)

    def predict(self, img, max_len, start_token, end_token, pad_token):
        encoder_out = self.encoder(img)
        return self.decoder.predict(encoder_out, max_len, start_token, end_token, pad_token)
