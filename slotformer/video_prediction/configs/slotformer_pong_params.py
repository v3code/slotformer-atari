from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 1  # lightweight since we don't use img recon loss
    max_epochs = 25  # ~230k steps
    save_interval = 0.125  # save every 0.125 epoch
    eval_interval = 2  # evaluate every 2 epochs
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples = 4  # Physion has 8 scenarios

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 2e-4
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps
    # no weight decay, no gradient clipping
    # sample_video = False
    
    # data settings
    dataset = 'pong_slots'
    data_root = './data/pong'
    slots_root = './data/pong/slots.pkl'
    tasks = ['all']  # train on all 8 scenarios
    n_sample_frames = 1 + 8  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    video_len = 50  
    train_batch_size = 64
    val_batch_size = 64
    num_workers = 3
    
    

    # model configs
    model = 'STEVESlotFormer'
    resolution = (64, 64)
    input_frames = 1  # burn-in frames

    num_slots = 4
    slot_size = 16
    slot_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
        slot_mlp_size=slot_size * 4,
        num_iterations=5,
        slots_init='shared_gaussian',
        truncate='fixed-point',
        sigma=1,
        
        use_dvae_encodings = True
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
        action_conditioning=True,
        actions_dim=18,
        discrete_actions=True,
    )


    # dVAE tokenizer
    dvae_dict = dict(
        down_factor=4,
        vocab_size=16,
        dvae_ckp_path='/code/checkpoint/dvae_pong_params/models/epoch/model_20.pth',
    )

    # TransformerDecoder
    dec_dict = dict(
        dec_num_layers=4,
        dec_num_heads=4,
        dec_d_model=slot_size,
        atten_type='linear',
        dec_ckp_path='/code/checkpoint/steve_pong_params/models/epoch/model_16.pth',
    )

    # loss configs
    loss_dict = dict(
        rollout_len=n_sample_frames - rollout_dict['history_len'],
        use_img_recon_loss=False,  # STEVE recon img is too memory-intensive
    )

    slot_recon_loss_w = 1.
    img_recon_loss_w = 1.
