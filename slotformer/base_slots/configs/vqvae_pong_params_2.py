from nerv.training import BaseParams


class SlotFormerParams(BaseParams):
    project = 'SlotFormer'

    # training settings
    gpus = 1  # 1 GPU should also be good
    max_epochs = 20  # ~700k steps
    save_interval = 0.25  # save every 0.25 epoch
    save_epoch_end = True  # save ckp at the end of every epoch
    n_samples =  4  # Physion has 8 scenarios

    # optimizer settings
    # Adam optimizer, Cosine decay with Warmup
    optimizer = 'Adam'
    lr = 1e-4
    warmup_steps_pct = 0.05  # warmup in the first 5% of total steps
    # no weight decay, no gradient clipping

    # data settings
    dataset = 'pong'
    data_root = './data/pong'
    tasks = ['all']  # train on all 8 scenarios
    n_sample_frames = 1  # train on single frames
    frame_offset = 1  # no offset
    video_len = 50  # take the first 150 frames of each video
    train_batch_size = 128
    val_batch_size = 128
    num_workers = 8

    # model configs
    model = 'vqVAE'
    vocab_size = 64
    resolution = (64, 64)
    
    enc_dec_dict = dict(
        resolution=resolution[0],
        in_channels=3,
        z_channels=32,
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

    recon_loss_w = 1.
    quant_loss_w = 1.
    percept_loss_w = vq_dict['percept_loss_w']
