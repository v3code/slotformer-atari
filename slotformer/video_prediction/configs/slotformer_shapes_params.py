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
    lr = 2e-5
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps
    # no weight decay, no gradient clipping
    # grad

    # data settings
    dataset = 'shapes_slots'
    data_root = './data/shapes'
    slots_root = './data/shapes/slots.pkl'
    tasks = ['all']  # train on all 8 scenarios
    n_sample_frames = 15 + 10  # train on video clips of 6 frames
    frame_offset = 1  # no offset
    video_len = 100  
    train_batch_size = 32
    val_batch_size = 32
    num_workers = 1
    
    

    # model configs
    model = 'STEVESlotFormer'
    resolution = (64, 64)
    input_frames = 15  # burn-in frames

    num_slots = 6
    slot_size = 128
    slot_dict = dict(
        num_slots=num_slots,
        slot_size=slot_size,
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
        num_heads=8,
        ffn_dim=slot_size * 4,
        norm_first=True,
    )


    # dVAE tokenizer
    dvae_dict = dict(
        down_factor=4,
        vocab_size=4096,
        dvae_ckp_path='checkpoint/dvae_shapes_params/models/epoch/model_20.pth',
    )

    # TransformerDecoder
    dec_dict = dict(
        dec_num_layers=4,
        dec_num_heads=4,
        dec_d_model=slot_size,
        atten_type='linear',
        dec_ckp_path='/code/checkpoint/steve_shapes_params/models/epoch/model_4.pth',
    )

    # loss configs
    loss_dict = dict(
        rollout_len=n_sample_frames - rollout_dict['history_len'],
        use_img_recon_loss=False,  # STEVE recon img is too memory-intensive
    )

    slot_recon_loss_w = 1.
    img_recon_loss_w = 1.
