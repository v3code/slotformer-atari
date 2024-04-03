# import matplotlib.pyplot as plt
# from skimage.draw import circle_perimeter_aa
# import numpy as np
# from PIL import Image
# img = np.zeros((220, 220, 3), dtype=np.uint8)
# rr, cc, val = circle_perimeter_aa(40, 40, 30)
# img[rr, cc] = val

# Image.fromarray(img).show()
# from pathlib import Path
# import pickle

# import numpy as np

# for file in Path('/code/data/pong/train').glob('./**/actions.pkl'):
#     lst = np.load(file)
#     assert len(lst) == 50, f'File {file} is broken'
import torch
import torch.nn.functional as F
from slotformer.base_slots.models.mlp import MLPDecoder
from slotformer.base_slots.models.slotmixer import MLP, SlotMixerDecoder
from slotformer.base_slots.models.slotmixer_transformer import TransformerEncoder

from slotformer.base_slots.models.vq_vae.vqvae import VQVAE
# a = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10], [11, 12, 13, 14, 15]])
# print(a[:, :2][:, -2:])

# a = torch.rand(3, 4, 5)
# b = torch.rand(3, 4, 5)

# print(a.flatten(end_dim=-2).shape)

# print(torch.sum(F.cosine_similarity(a.flatten(end_dim=-2), b.flatten(end_dim=-2))**2))

vocab_size = 16
resolution = (64, 64)
slot_size=64
num_patches = 16*16
num_slots=4


enc_dec_dict = dict(
        resolution=resolution[0],
        in_channels=3,
        z_channels=3,
        ch=64,  # base_channel
        ch_mult=[1, 2, 4],  # num_down = len(ch_mult)-1
        num_res_blocks=2,
        attn_resolutions=[],
        out_ch=3,
        dropout=0.0,
    )
vq_dict = dict(
        n_embed=vocab_size,  # vocab_size
        embed_dim=enc_dec_dict['z_channels'],
        percept_loss_w=1.0,
    )

model = VQVAE(
    enc_dec_dict=enc_dec_dict,
    vq_dict=vq_dict,
    use_loss=False,
)

# test_img = torch.randint(0, 30, size=(1, 3, 64, 64))
test_img = torch.rand(1, 3, 64, 64)
# print(model.encode_quantize(test_img)[2].shape)
quant, _, a = model.encode_quantize(test_img)
print(a)

# print(model.quantize.get_codebook_entry(test_img, None))


# dec_dict = dict(
#         allocator = dict(
#             dim=slot_size,
#             memory_dim=slot_size,
#             n_blocks=2,
#             n_heads=4,
#         ),
#         renderer=dict(
#             inp_dim=slot_size,
#             outp_dim=vocab_size,
#             # outp_dim=vocab_size,
#             hidden_dims=[vocab_size*2]*3,
#             final_activation=False,
#         ),
#         model_conf=dict(
#             inp_dim=slot_size,
#             embed_dim=slot_size,
#             outp_dim=num_patches,
#             n_patches=num_patches,
#             use_layer_norms=True,
#             pos_embed_mode='add',
#             renderer_dim=vocab_size,
#         ),
#     )


# dec_dict = dict(
#         inp_dim = slot_size,
#         outp_dim = vocab_size,
#         n_patches = num_patches,
#         hidden_dims= [vocab_size*4]*4,
# )
# decoder = MLPDecoder(
#     **dec_dict
# )

# slots = torch.rand(2, num_slots, slot_size)

# out, masks = decoder(slots)
# print(masks.shape)