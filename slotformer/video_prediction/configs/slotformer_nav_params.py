from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 1  # lightweight since we don't use img recon loss
    max_epochs = 30  # ~230k steps
    save_interval = 0.125  # save every 0.125 epoch
    eval_interval = 2  # evaluate every 2 epochs
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 4  # Physion has 8 scenarios
    
    grad_accum_steps=2
    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 2e-4
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps
    # no weight decay, no gradient clipping
    # sample_video = False
    
    # data settings
    dataset = 'navigation_slots'
    data_root = './data/nav5x5'
    slots_root = './data/nav5x5/slots.pkl'
    tasks = ['all']  # train on all 8 scenarios
    n_sample_frames = 4  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    video_len = 50  
    train_batch_size = 64
    val_batch_size = 64
    num_workers = 1
    
    

    # model configs
    model = 'STEVESlotFormer'
    resolution = (64, 64)
    input_frames = 1  # burn-in frames

    num_slots = 6
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
    # Rollouter
    rollout_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
        history_len=input_frames,
        t_pe='sin',  # sine temporal P.E.
        slots_pe='',  # no slots P.E.
        # Transformer-related configs
        d_model=slot_size,
        num_layers=4,
        num_heads=4,
        ffn_dim=slot_size * 4,
        norm_first=True,
        action_conditioning=False,
        actions_dim=8,
        num_actions=16,
        discrete_actions=True,
    )


    # dVAE tokenizer
    dvae_dict = dict(
        down_factor=4,
        vocab_size=32,
        dvae_ckp_path='/code/pretrained/dvae_shapes.pth',
        use_ocr_version=True,
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
        atten_type='linear',
        dec_ckp_path='/code/pretrained/steve_shapes.pth',
    
    )

    # loss configs
    loss_dict = dict(
        rollout_len=n_sample_frames - rollout_dict['history_len'],
        use_img_recon_loss=False,  # STEVE recon img is too memory-intensive
        use_cosine_similarity_loss = False,
        use_directional_loss=False
    )

    slot_recon_loss_w = 1.
    img_recon_loss_w = 1.
