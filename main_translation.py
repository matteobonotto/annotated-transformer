import torch
from torch import nn, Tensor
import torch.functional as F
from typing import Union, Callable, Optional, Any
import lightning as L

#%% 
#%% Model
model = nn.Transformer()

class PositionalEncoding(nn.Module):
    def __init__(self):
        super.__init__(TransformerModel,self)
        pass

class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        pass

class TransformerModel(nn.Module):
    def __init__(
            self,
            d_model: int = 512, 
            nhead: int = 8, 
            num_encoder_layers: int = 6,
            num_decoder_layers: int = 6, 
            dim_feedforward: int = 2048
            ):
        super.__init__(TransformerModel,self)

        ## Encoder part
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward,
            )
        encoder_norm = nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers = num_encoder_layers,
            norm=encoder_norm
            )

        ## Decoder part
        decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward,
            )
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerEncoder(
            encoder_layer=decoder_layer,
            num_layers = num_decoder_layers,
            norm=decoder_norm
            )

    def forward(
            self,
            src: Tensor, 
            tgt: Tensor, 
            src_mask: Optional[Tensor] = None, 
            tgt_mask: Optional[Tensor] = None
            ):
        memory = self.encoder(src, mask=src_mask)
        output = self.decoder(tgt, memory, tgt_mask=tgt_mask)
        return output
    

#%%
## Data



















