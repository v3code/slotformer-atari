from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    # STEVE on 128x128 resolution is very memory-consuming
    # the setting here is for training on 4 RTX6000 GPUs with 24GB memory each
    gpus = 1
    max_epochs = 40  # ~460k steps
    save_interval = 0.05  # training is very slow, save every 0.05 epoch
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 4  # Physion has 8 scenarios

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    # weight_decay=1.5 
    lr = 1e-4  # 1e-4 for the main STEVE model
    dec_lr = 3e-4  # 3e-4 for the Transformer decoder
    clip_grad = 0.05  # following the paper
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps

    # data settings
    dataset = 'navigation'
    data_root = './data/nav5x5'
    tasks = ['all']  # train on all 8 scenarios
    n_sample_frames = 6  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    video_len = 50  
    train_batch_size = 32
    val_batch_size = 32
    num_workers = 4


    # model configs
    model = 'STEVE'
    resolution = (64, 64)
    input_frames = n_sample_frames

    # Slot Attention
    slot_size = 64
    slot_dict = dict(
        # the objects are harder to define in Physion than in e.g. CLEVRER
        # e.g. should a stack of 6 boxes be considered as 1 or 6 objects?
        #      among these boxes, some move together, some usually fall, etc.
        # we don't ablate this `num_slots`, so we are not sure if it will make
        # a huge difference to the results
        # qualitatively, 6 achieves a reasonable scene decomposition result
        num_slots=6,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 4,
        num_iterations=5,
        slots_init='shared_gaussian',
        truncate='none',
        sigma=1,
        use_ocr_sa = True,
        use_dvae_encodings = False
    )

    vocab_size = 32
    # dVAE tokenizer
    dvae_dict = dict(
        down_factor=4,
        vocab_size=vocab_size,
        dvae_ckp_path='/code/pretrained/dvae_shapes.pth',
        use_ocr_version=True,
    )

    # CNN Encoder
    enc_dict = dict(
        enc_channels=(3, 64, 64, 64, 64),
        enc_ks=5,
        enc_out_channels=slot_size,
        enc_norm='',
        use_ocr_encoder=True,
    )

    # TransformerDecoder
    dec_dict = dict(
        dec_type='slate',
        use_bos_token_in_tf=False,
        use_proj_bias=False,
        dec_num_layers=4,
        dec_num_heads=4,
        dec_d_model=128,
        slot_size=slot_size,
        atten_type='linear' 
    )

    # Predictor
    pred_dict = dict(
        pred_type='transformer',
        pred_rnn=True,
        pred_norm_first=True,
        pred_num_layers=2,
        pred_num_heads=4,
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
