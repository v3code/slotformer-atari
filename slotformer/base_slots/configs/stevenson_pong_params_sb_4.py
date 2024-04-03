from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    # STEVE on 128x128 resolution is very memory-consuming
    # the setting here is for training on 4 RTX6000 GPUs with 24GB memory each
    gpus = 1
    max_epochs = 70  # ~460k steps
    save_interval = 0.05  # training is very slow, save every 0.05 epoch
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 4  # Physion has 8 scenarios

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    # weight_decay=0.05    
    lr = 1e-4  # 1e-4 for the main STEVE model
    dec_lr = 3e-4  # 3e-4 for the Transformer decoder
    clip_grad = 0.08  # following the paper
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps

    # data settings
    dataset = 'pong'
    data_root = './data/pong'
    tasks = ['all']  # train on all 8 scenarios
    n_sample_frames = 6  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    video_len = 30  
    train_batch_size = 64
    val_batch_size = 64
    num_workers = 8

    # model configs
    model = 'STEVENSON'
    resolution = (64, 64)
    input_frames = n_sample_frames

    # Slot Attention
    slot_size = 16
    slot_dict = dict(
        # the objects are harder to define in Physion than in e.g. CLEVRER
        # e.g. should a stack of 6 boxes be considered as 1 or 6 objects?
        #      among these boxes, some move together, some usually fall, etc.
        # we don't ablate this `num_slots`, so we are not sure if it will make
        # a huge difference to the results
        # qualitatively, 6 achieves a reasonable scene decomposition result
        num_slots=4,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 4,
        num_iterations=3,
        slots_init='param',
        truncate='none',
        sigma=1
    )

    # dVAE tokenizer
    
    vocab_size = 16
    num_patches = 16*16
    
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
    
    vae_dict = dict(
        enc_dec_dict=enc_dec_dict,
        vq_dict=vq_dict,
        ckp_path='/code/checkpoint/vqvae_pong_params_3/models/epoch/model_20.pth',
    )

    # CNN Encoder
    enc_dict = dict(
        enc_channels=(3, 64, 64, 64, 64),
        enc_ks=5,
        enc_out_channels=slot_size,
        enc_norm='',
    )

    # TransformerDecoder
    # dec_dict = dict(
    #     inp_dim = slot_size,
    #     outp_dim = vocab_size,
    #     n_patches = num_patches,
    #     hidden_dims= [vocab_size*4]*4,
        # allocator = dict(
        #     dim=slot_size,
        #     memory_dim=slot_size,
        #     n_blocks=2,
        #     n_heads=4,
        # ),
        # renderer=dict(
        #     dim=slot_size,
        #     outp_dim=vocab_size,
        #     hidden_dims=[vocab_size*2]*3,
        #     final_activation=False,
        # ),
        # model_conf=dict(
        #     inp_dim=slot_size,
        #     embed_dim=slot_size,
        #     outp_dim=num_patches,
        #     n_patches=num_patches,
        #     use_layer_norms=True,
        #     pos_embed_mode='add',
        #     renderer_dim=vocab_size,
        # ),
    # )
    
    # TransformerDecoder
    dec_dict = dict(
        dec_num_layers=4,
        dec_num_heads=4,
        dec_d_model=slot_size,
        atten_type='linear'
    )

    # Predictor
    pred_dict = dict(
        pred_type='transformer',
        pred_rnn=True,
        pred_norm_first=True,
        pred_num_layers=2,
        pred_num_heads=1,
        pred_ffn_dim=slot_size * 4,
        pred_sg_every=None,
    )

    # loss settings
    loss_dict = dict(
        use_img_recon_loss=False,  # additional img recon loss via dVAE decoder
        use_slots_correlation_loss=False,
        use_cossine_similarity_loss=False
    )

    token_recon_loss_w = 1.
    slots_correlation_loss_w = 5.
    img_recon_loss_w = 1.
    cossine_similarity_loss_w = 5.
