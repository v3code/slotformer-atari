import torch
import torch.nn as nn

# from slotformer.base_slots.models import dVAE
# from slotformer.base_slots.models.steve import SlotAttentionWMask
# from slotformer.base_slots.models.steve_transformer import STEVETransformerDecoder

state = torch.load("slate_navigation5x5.pth")["ocr_module_state_dict"]

# state = torch.load("/code/pretrained/steve_shapes.pth")
# print(state.keys())
# raise Exception("stop")
max_len = (64 // 4)**2


# model = STEVETransformerDecoder(32, use_bos_token_in_tf=False, slot_size=64, use_proj_bias=False, n_head=4, d_model=128, num_layers=4, num_slots=6, max_len=max_len)


def process_dvae_key(key):
    key = key.replace("_dvae._", "")
    return f"dvae.{key}"

dvae_sate = {process_dvae_key(k): v for k, v in state.items() if k.startswith("_dvae")}

# torch.save(dvae_sate, "pretrained/dvae_shapes.pth")

# print(torch.load("/code/checkpoint/steve_pong_params/models/model_224325.pth")["state_dict"].keys())

# print("___________")
# print("___________")
# print("___________")
# print("___________")
# print("___________")


# print(state.keys())


# transformer_decoder

tf_dec_state = {k.replace("_tfdec", "tf_dec"): v for k,v in state.items() if k.startswith("_tfdec")}
# print(tf_dec_state.keys())
pos_embed_state = {k.replace("_z_pos", "pos_emb"): v for k,v in state.items() if k.startswith("_z_pos")}
bos_token = state["_bos_token._bos_token"]
tok_emb = state["_dict.dictionary.weight"]

tok_emb = torch.cat([tok_emb, bos_token.squeeze(0)], dim=0)


in_proj = {k.replace("_slotproj", "in_proj"): v for k,v in state.items() if k.startswith("_slotproj")}
head = {k.replace("_out", "head"): v for k,v in state.items() if k.startswith("_out")}

transformer_decoder = dict(
    **tf_dec_state, 
    **pos_embed_state,  
    **in_proj, 
    **head)

transformer_decoder["tok_emb.weight"] = tok_emb
# print(transformer_decoder.keys())
transformer_decoder = {f"trans_decoder.{k}": v for k,v in transformer_decoder.items()}


# model.load_state_dict(transformer_decoder)


# model.tf_dec.load_state_dict(tf_dec_state)

# torch.save(decoder_sate, "pretrained/decoder_shapes.pth")


sa_state_old = {k.replace("_slotattn.", ""): v for k, v in state.items() if k.startswith("_slotattn.")}
sa_mu = sa_state_old["slot_mu"]
sa_logsigma = sa_state_old["slot_log_sigma"]

enc = {k: v for k, v in state.items() if k.startswith("_enc")}
# enc_pos = {k: v for k, v in state.items() if k.startswith("_enc_pos")}

sa_state = {f"slot_attention.{k}": v for k, v in sa_state_old.items() if k not in ["slot_mu", "slot_log_sigma"]}

new_state = dict(**sa_state, **transformer_decoder, **dvae_sate, **enc)
new_state['slot_mu'] = sa_mu
new_state["slot_log_sigma"] = sa_logsigma

# new_state = {k: v.to(torch.float16) for k,v in new_state.items()}

torch.save({"state_dict": new_state}, "pretrained/steve_shapes.pth")


# mlp = {k.replace("_mlp.", ""): v for k, v in sa_state.items() if k.startswith("_mlp.")}

# print(sa_state.keys())
# print(sa.state_dict().keys())