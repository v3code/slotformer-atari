import wandb
import numpy as np

import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.utils as vutils

from nerv.training import BaseMethod, CosineAnnealingWarmupRestarts

# from nerv.lion import Lion

from .models import cosine_anneal, get_lr, gumbel_softmax, make_one_hot, \
    to_rgb_from_tensor


def build_method(**kwargs):
    params = kwargs['params']
    if params.model == 'StoSAVi':
        return SAViMethod(**kwargs)
    elif params.model in ('dVAE', 'vqVAE'):
        return VAEMethod(**kwargs)
    elif params.model == 'STEVE':
        return STEVEMethod(**kwargs)
    elif params.model == 'STEVENSON':
        return STEVENSONMethod(**kwargs)
    else:
        raise NotImplementedError(f'{params.model} method is not implemented.')


class SlotBaseMethod(BaseMethod):
    """Base method in this project."""

    @staticmethod
    def _pad_frame(video, target_T):
        """Pad the video to a target length at the end"""
        if video.shape[0] >= target_T:
            return video
        dup_video = torch.stack(
            [video[-1]] * (target_T - video.shape[0]), dim=0)
        return torch.cat([video, dup_video], dim=0)

    @staticmethod
    def _pause_frame(video, N=4):
        """Pause the video on the first frame by duplicating it"""
        dup_video = torch.stack([video[0]] * N, dim=0)
        return torch.cat([dup_video, video], dim=0)

    def _convert_video(self, video, caption=None):
        video = torch.cat(video, dim=2)  # [T, 3, B*H, L*W]
        video = (video * 255.).numpy().astype(np.uint8)
        return wandb.Video(video, fps=self.vis_fps, caption=caption)

    @staticmethod
    def _get_sample_idx(N, dst):
        """Load videos uniformly from the dataset."""
        dst_len = len(dst.files)  # treat each video as a sample
        N = N - 1 if dst_len % N != 0 else N
        sampled_idx = torch.arange(0, dst_len, dst_len // N)
        return sampled_idx

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1, sample_video=True):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        super().validation_epoch(model, san_check_step=san_check_step)
        if self.local_rank != 0:
            return
        # visualization after every epoch
        if sample_video:
            self._sample_video(model)

    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        optimizer = super()._configure_optimizers()[0]

        lr = self.params.lr
        total_steps = self.params.max_epochs * len(self.train_loader)
        warmup_steps = self.params.warmup_steps_pct * total_steps

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=lr,
            min_lr=lr / 100.,
            warmup_steps=warmup_steps,
        )

        return optimizer, (scheduler, 'step')

    @property
    def vis_fps(self):
        # PHYRE
        if 'phyre' in self.params.dataset.lower():
            return 4
        # OBJ3D, CLEVRER, Physion
        else:
            return 8


class SAViMethod(SlotBaseMethod):
    """SAVi model training method."""

    def _make_video_grid(self, imgs, recon_combined, recons, masks):
        """Make a video of grid images showing slot decomposition."""
        # pause the video on the 1st frame in PHYRE
        if 'phyre' in self.params.dataset.lower():
            imgs, recon_combined, recons, masks = [
                self._pause_frame(x)
                for x in [imgs, recon_combined, recons, masks]
            ]
        # in PHYRE if the background is black, we scale the mask differently
        scale = 0. if self.params.get('reverse_color', False) else 1.
        # combine images in a way so we can display all outputs in one grid
        # output rescaled to be between 0 and 1
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    imgs.unsqueeze(1),  # original images
                    recon_combined.unsqueeze(1),  # reconstructions
                    recons * masks + (1. - masks) * scale,  # each slot
                ],
                dim=1,
            ))  # [T, num_slots+2, 3, H, W]
        # stack the slot decomposition in all frames to a video
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
                pad_value=1. - scale,
            ) for i in range(recons.shape[0])
        ])  # [T, 3, H, (num_slots+2)*W]
        return save_video

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results, labels = [], []
        for i in sampled_idx:
            data_dict = dst.get_video(i.item())
            video, label = data_dict['video'].float().to(self.device), \
                data_dict.get('label', None)  # label for PHYRE
            in_dict = {'img': video[None]}
            out_dict = model(in_dict)
            out_dict = {k: v[0] for k, v in out_dict.items()}
            recon_combined, recons, masks = out_dict['post_recon_combined'], \
                out_dict['post_recons'], out_dict['post_masks']
            imgs = video.type_as(recon_combined)
            save_video = self._make_video_grid(imgs, recon_combined, recons,
                                               masks)
            results.append(save_video)
            labels.append(label)

        if all(lbl is not None for lbl in labels):
            caption = '\n'.join(
                ['Success' if lbl == 1 else 'Fail' for lbl in labels])
        else:
            caption = None
        wandb.log({'val/video': self._convert_video(results, caption=caption)},
                  step=self.it)
        torch.cuda.empty_cache()


class VAEMethod(SlotBaseMethod):
    """dVAE and VQVAE model training method."""
    
    @staticmethod
    def _make_video(video, pred_video):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(torch.stack([video, pred_video],
                                             dim=1))  # [T, 2, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, 2*W]
        return save_video

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        results = []
        for i in sampled_idx:
            video = dst.get_video(i.item())['video'].float().to(self.device)
            all_recons, bs = [], 100  # a hack to avoid OOM
            for batch_idx in range(0, video.shape[0], bs):
                data_dict = {
                    'img': video[batch_idx:batch_idx + bs],
                    'tau': 1.,
                    'hard': True,
                }
                recon = model(data_dict)['recon']
                all_recons.append(recon)
                torch.cuda.empty_cache()
            recon_video = torch.cat(all_recons, dim=0)
            save_video = self._make_video(video, recon_video)
            results.append(save_video)

        wandb.log({'val/video': self._convert_video(results)}, step=self.it)
        torch.cuda.empty_cache()
    
    @staticmethod
    def _make_img(imgs, pred_imgs):
        """imgs are of shape [N, C, H, W]"""
        out = torch.stack([
            imgs.cpu(),
            pred_imgs.cpu(),
        ], dim=1).cpu()  # [N, 2, 3, H, W]
        out = vutils.make_grid(
            out.flatten(0, 1),
            nrow=out.shape[1],
            pad_value=-1.,
        )  # [T, 3, H, 2*W]
        return out

    @torch.no_grad()
    def _sample_img(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        collate_fn = self.val_loader.collate_fn
        data_dict = collate_fn([dst[i] for i in sampled_idx])
        data_dict = {k: v.to(model.device) for k, v in data_dict.items()}

        out_dict = model(data_dict)
        recon_imgs = out_dict['recon']
        imgs = data_dict['img']
        images = self._make_img(imgs, recon_imgs)

        wandb.log({'val/imgs': self._convert_img(images)}, step=self.it)
        torch.cuda.empty_cache()

    def _training_step_start(self):
        """Things to do at the beginning of every training step."""
        super()._training_step_start()

        if self.params.model != 'dVAE':
            return

        # dVAE: update the tau (gumbel softmax temperature)
        total_steps = self.params.max_epochs * len(self.train_loader)
        decay_steps = self.params.tau_decay_pct * total_steps

        # decay tau
        self.model.module.tau = cosine_anneal(
            self.it,
            start_value=self.params.init_tau,
            final_value=self.params.final_tau,
            start_step=0,
            final_step=decay_steps,
        )

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.params.model != 'dVAE':
            return

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        # also log the tau
        wandb.log({'train/gumbel_tau': self.model.module.tau}, step=self.it)


class STEVEMethod(SlotBaseMethod):
    """STEVE model training method."""
    


    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        # assert self.params.optimizer.lower() == 'adam'
        # assert self.params.weight_decay <= 0.
        lr = self.params.lr
        dec_lr = self.params.dec_lr
        if self.params.weight_decay is None:
            self.params.weight_decay = 0.

        # STEVE uses different lr for its Transformer decoder and other parts
        sa_params = list(
            filter(
                lambda kv: 'trans_decoder' not in kv[0] and kv[1].
                requires_grad, self.model.named_parameters()))
        dec_params = list(
            filter(lambda kv: 'trans_decoder' in kv[0],
                   self.model.named_parameters()))

        params_list = [
            {
                'params': [kv[1] for kv in sa_params],
            },
            {
                'params': [kv[1] for kv in dec_params],
                'lr': dec_lr,
            },
        ]
        if self.params.optimizer.lower() == 'adam':
            optimizer = optim.Adam(params_list, lr=lr, weight_decay=self.params.weight_decay)
        # if self.params.optimizer.lower() == 'lion':
        #     optimizer = Lion(params_list, lr=lr, weight_decay=self.params.weight_decay)
        else:
            optimizer = optim.AdamW(params_list, lr=lr, weight_decay=self.params.weight_decay)

        total_steps = self.params.max_epochs * len(self.train_loader)
        warmup_steps = self.params.warmup_steps_pct * total_steps

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=(lr, dec_lr),
            min_lr=0.,
            warmup_steps=warmup_steps,
        )

        return optimizer, (scheduler, 'step')

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        # STEVE's Transformer-based decoder autoregressively reconstructs the
        # video, which is super slow
        # therefore, we only visualize scene decomposition results
        # but don't show the video reconstruction
        # change this if you want to see reconstruction anyways
        self.recon_video = True
        super().validation_epoch(model, san_check_step=san_check_step)

    @staticmethod
    def _make_video(video, soft_video, hard_video, history_len=None):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(
            torch.stack(
                [
                    video.cpu(),  # original video
                    soft_video.cpu(),  # dVAE gumbel softmax reconstruction
                    hard_video.cpu(),  # argmax token reconstruction
                ],
                dim=1,
            ))  # [T, 3, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i],
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, 3*W]
        return save_video

    @staticmethod
    def _make_slots_video(video, pred_video):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(
            torch.cat(
                [ 
                    video.unsqueeze(1),  # [T, 1, 3, H, W]
                    pred_video,  # [T, num_slots, 3, H, W]
                ],
                dim=1,
            ))  # [T, num_slots + 1, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, (num_slots+1)*W]
        return save_video
    
    @staticmethod
    def _make_masks_video(video, pred_video):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    video.unsqueeze(1),  # [T, 1, 3, H, W]
                    pred_video.unsqueeze(2).expand(-1,-1,3,-1,-1),  # [T, num_slots, 1, H, W]
                ],
                dim=1,
            ))  # [T, num_slots + 1, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, (num_slots+1)*W]
        return save_video

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        model.testing = True  # we only want the slots
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        num_patches = model.num_patches
        n = int(num_patches**0.5)
        use_dec_masks = False
        results, recon_results, masks_result = [], [], []
        dec_results, dec_masks_results = [], []
        for i in sampled_idx:
            video = dst.get_video(i.item())['video'].float().to(self.device)
            data_dict = {'img': video[None]}
            out_dict = model(data_dict)
            masks = out_dict['masks'][0]  # [T, num_slots, H, W]
            masked_video = video.unsqueeze(1) * masks.unsqueeze(2)
            # [T, num_slots, C, H, W]
            save_video = self._make_slots_video(video, masked_video)
            masks_video = self._make_masks_video(video, masks)
            results.append(save_video)
            masks_result.append(masks_video)
            dec_masks = out_dict.get('dec_masks')
            if dec_masks is not None:
                use_dec_masks = True
                dec_masks = dec_masks[0]
                dec_masked_video = video.unsqueeze(1) * dec_masks.unsqueeze(2)
                # [T, num_slots, C, H, W]
                dec_save_video = self._make_slots_video(video, dec_masked_video)
                dec_results.append(dec_save_video)
                dec_masks_video = self._make_masks_video(video, dec_masks)
                dec_masks_results.append(dec_masks_video)
            if not self.recon_video :
                continue
            
            recon_video, recon_video_hard = self._generate_recon(model, out_dict, num_patches, n)

            save_video = self._make_video(video, recon_video, recon_video_hard)
            recon_results.append(save_video)
            torch.cuda.empty_cache()
            save_video = self._make_video(video, recon_video, recon_video_hard)
            recon_results.append(save_video)
            torch.cuda.empty_cache()

        log_dict = {'val/video': self._convert_video(results),
                    'val/masks_video': self._convert_video(masks_result)}
        if use_dec_masks:
            log_dict.update({
                'val/dec_video': self._convert_video(dec_results),
                'val/dec_masks_video': self._convert_video(dec_masks_results),
            })
        if self.recon_video:
            log_dict['val/recon_video'] = self._convert_video(recon_results)
        wandb.log(log_dict, step=self.it)
        torch.cuda.empty_cache()
        model.testing = False
    
    @torch.no_grad()
    def _generate_recon(self, model, out_dict, num_patches, n):
        slots = out_dict['slots'][0]  # [T, num_slots, slot_size]
        if model.dec_dict['dec_type'] == 'slate':
            all_soft_video, all_hard_video, bs = [], [], 16  # to avoid OOM
            for batch_idx in range(0, slots.shape[0], bs):
                _, logits = model.trans_decoder.generate(
                    slots[batch_idx:batch_idx + bs],
                    steps=num_patches,
                    sample=False,
                )
                # [T, patch_size**2, vocab_size] --> [T, vocab_size, h, w]
                logits = logits.transpose(2, 1).unflatten(
                    -1, (n, n)).contiguous().cuda()
                # 1. use logits after gumbel softmax to reconstruct the video
                z_logits = F.log_softmax(logits, dim=1)
                z = gumbel_softmax(z_logits, 0.1, hard=False, dim=1)
                recon_video = model.dvae.detokenize(z)
                all_soft_video.append(recon_video.cpu())
                del z_logits, z, recon_video
                torch.cuda.empty_cache()
                # 2. SLATE directly use ont-hot token (argmax) as input
                z_hard = make_one_hot(logits, dim=1)
                recon_video_hard = model.dvae.detokenize(z_hard)
                all_hard_video.append(recon_video_hard.cpu())
                del logits, z_hard, recon_video_hard
                torch.cuda.empty_cache()

            recon_video = torch.cat(all_soft_video, dim=0)
            recon_video_hard = torch.cat(all_hard_video, dim=0)
            return recon_video, recon_video_hard
            
            
        elif model.dec_dict['dec_type'] == 'mlp':
            logits = out_dict['pred_token_id']
            h, w = model.h, model.w
            T, L, C = logits.shape
            logits = logits.transpose(1, 2).reshape(-1, C, h, w)
            z_logits = F.log_softmax(logits, dim=1)
            z = gumbel_softmax(z_logits, 0.1, hard=False, dim=1)
            recon_video = model.dvae.detokenize(z)
            # all_soft_video.append(recon_video.cpu())
            del z_logits, z
            torch.cuda.empty_cache()
            z_hard = make_one_hot(logits, dim=1)
            recon_video_hard = model.dvae.detokenize(z_hard)
            # all_hard_video.append(recon_video_hard.cpu())
            del logits, z_hard
            torch.cuda.empty_cache()
            return recon_video, recon_video_hard

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        # log all the lr
        log_dict = {
            'train/lr': get_lr(self.optimizer),
            'train/dec_lr': self.optimizer.param_groups[1]['lr'],
        }
        wandb.log(log_dict, step=self.it)


# class STEVENSONMethod(SlotBaseMethod):
#     """STEVE model training method."""
    


#     def _configure_optimizers(self):
#         """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
#         # assert self.params.optimizer.lower() == 'adam'
#         # assert self.params.weight_decay <= 0.
#         lr = self.params.lr
#         dec_lr = self.params.dec_lr
#         if self.params.weight_decay is None:
#             self.params.weight_decay = 0.

#         # STEVE uses different lr for its Transformer decoder and other parts
#         sa_params = list(
#             filter(
#                 lambda kv: 'decoder' not in kv[0] and kv[1].
#                 requires_grad, self.model.named_parameters()))
#         dec_params = list(
#             filter(lambda kv: 'decoder' in kv[0],
#                    self.model.named_parameters()))

#         params_list = [
#             {
#                 'params': [kv[1] for kv in sa_params],
#             },
#             {
#                 'params': [kv[1] for kv in dec_params],
#                 'lr': dec_lr,
#             },
#         ]
#         if self.params.optimizer.lower() == 'adam':
#             optimizer = optim.Adam(params_list, lr=lr, weight_decay=self.params.weight_decay)
#         # if self.params.optimizer.lower() == 'lion':
#         #     optimizer = Lion(params_list, lr=lr, weight_decay=self.params.weight_decay)
#         else:
#             optimizer = optim.AdamW(params_list, lr=lr, weight_decay=self.params.weight_decay)

#         total_steps = self.params.max_epochs * len(self.train_loader)
#         warmup_steps = self.params.warmup_steps_pct * total_steps

#         scheduler = CosineAnnealingWarmupRestarts(
#             optimizer,
#             total_steps,
#             max_lr=(lr, dec_lr),
#             min_lr=0.,
#             warmup_steps=warmup_steps,
#         )

#         return optimizer, (scheduler, 'step')

#     @torch.no_grad()
#     def validation_epoch(self, model, san_check_step=-1):
#         """Validate one epoch.

#         We aggregate the avg of all statistics and only log once.
#         """
#         # STEVE's Transformer-based decoder autoregressively reconstructs the
#         # video, which is super slow
#         # therefore, we only visualize scene decomposition results
#         # but don't show the video reconstruction
#         # change this if you want to see reconstruction anyways
#         self.recon_video = True
#         super().validation_epoch(model, san_check_step=san_check_step)

#     @staticmethod
#     def _make_video(video, recon_video, history_len=None):
#         """videos are of shape [T, C, H, W]"""
#         out = to_rgb_from_tensor(
#             torch.stack(
#                 [
#                     video.cpu(),  # original video
#                     recon_video.cpu(),  # dVAE gumbel softmax reconstruction
                    
#                 ],
#                 dim=1,
#             ))  # [T, 3, 3, H, W]
#         save_video = torch.stack([
#             vutils.make_grid(
#                 out[i],
#                 nrow=out.shape[1],
#             ) for i in range(video.shape[0])
#         ])  # [T, 3, H, 3*W]
#         return save_video

#     @staticmethod
#     def _make_slots_video(video, pred_video):
#         """videos are of shape [T, C, H, W]"""
#         out = to_rgb_from_tensor(
#             torch.cat(
#                 [
#                     video.unsqueeze(1),  # [T, 1, 3, H, W]
#                     pred_video,  # [T, num_slots, 3, H, W]
#                 ],
#                 dim=1,
#             ))  # [T, num_slots + 1, 3, H, W]
#         save_video = torch.stack([
#             vutils.make_grid(
#                 out[i].cpu(),
#                 nrow=out.shape[1],
#             ) for i in range(video.shape[0])
#         ])  # [T, 3, H, (num_slots+1)*W]
#         return save_video
    
#     @staticmethod
#     def _make_masks_video(video, pred_video):
#         """videos are of shape [T, C, H, W]"""
#         out = to_rgb_from_tensor(
#             torch.cat(
#                 [
#                     video.unsqueeze(1),  # [T, 1, 3, H, W]
#                     pred_video.unsqueeze(2).expand(-1,-1,3,-1,-1),  # [T, num_slots, 1, H, W]
#                 ],
#                 dim=1,
#             ))  # [T, num_slots + 1, 3, H, W]
#         save_video = torch.stack([
#             vutils.make_grid(
#                 out[i].cpu(),
#                 nrow=out.shape[1],
#             ) for i in range(video.shape[0])
#         ])  # [T, 3, H, (num_slots+1)*W]
#         return save_video

#     @torch.no_grad()
#     def _sample_video(self, model):
#         """model is a simple nn.Module, not warpped in e.g. DataParallel."""
#         model.eval()
#         model.testing = True  # we only want the slots
#         dst = self.val_loader.dataset
#         sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
#         num_patches = model.num_patches
#         n = int(num_patches**0.5)
#         results, recon_results,  masks_result, decoder_masks_result = [], [], [], []
#         for i in sampled_idx:
#             video = dst.get_video(i.item())['video'].float().to(self.device)
#             data_dict = {'img': video[None]}
#             out_dict = model(data_dict)
#             masks = out_dict['masks'][0]  # [T, num_slots, H, W]
#             masked_video = video.unsqueeze(1) * masks.unsqueeze(2)
#             # [T, num_slots, C, H, W]
#             # save_video = self._make_slots_video(video, masked_video)
#             masks_video = self._make_masks_video(video, masks)
#             decoder_masks_video = self._make_masks_video(video, out_dict['decoder_masks'][0])
#             # results.append(save_video)
#             masks_result.append(masks_video)
#             decoder_masks_result.append(decoder_masks_video)
#             if not self.recon_video:
#                 continue

#             # reconstruct the video by autoregressively generating patch tokens
#             # using Transformer decoder conditioned on slots
#             logits = out_dict['pred_token_id'] # [T, num_slots, slot_size]
#             h,w = model.visual_resolution
#             T, L, C = logits.shape
#             logits = logits.reshape(-1, h, w, C)
#             # all_soft_video, all_hard_video, bs = [], [], 16  # to avoid OOM
#             soft_logits = F.log_softmax(logits, dim=-1)
#             hard_logits = torch.argmax(soft_logits, dim=-1)
#             recon = model.vqvae.decode_tokens(hard_logits)
            
#             # for batch_idx in range(0, slots.shape[0], bs):
                
#             #     _, logits = model.trans_decoder.generate(
#             #         slots[batch_idx:batch_idx + bs],
#             #     )
#             #     # [T, patch_size**2, vocab_size] --> [T, vocab_size, h, w]
#             #     logits = logits.transpose(2, 1).unflatten(
#             #         -1, (n, n)).contiguous().cuda()
#             #     # 1. use logits after gumbel softmax to reconstruct the video
#             #     z_logits = F.log_softmax(logits, dim=1)
#             #     z = gumbel_softmax(z_logits, 0.1, hard=False, dim=1)
#             #     recon_video = model.dvae.detokenize(z)
#             #     all_soft_video.append(recon_video.cpu())
#             #     del z_logits, z, recon_video
#             #     torch.cuda.empty_cache()
#             #     # 2. SLATE directly use ont-hot token (argmax) as input
#             #     z_hard = make_one_hot(logits, dim=1)
#             #     recon_video_hard = model.dvae.detokenize(z_hard)
#             #     all_hard_video.append(recon_video_hard.cpu())
#             #     del logits, z_hard, recon_video_hard
#             #     torch.cuda.empty_cache()

#             save_video = self._make_video(video, recon)
#             recon_results.append(save_video)
#             torch.cuda.empty_cache()

#         log_dict = {
#                     'val/masks_video': self._convert_video(masks_result),
#                     'val/decoder_masks_video': self._convert_video(decoder_masks_result)
#                     }
#         if self.recon_video:
#             log_dict['val/recon_video'] = self._convert_video(recon_results)
#         wandb.log(log_dict, step=self.it)
#         torch.cuda.empty_cache()
#         model.testing = False

#     def _log_train(self, out_dict):
#         """Log statistics in training to wandb."""
#         super()._log_train(out_dict)

#         if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
#             return

#         # log all the lr
#         log_dict = {
#             'train/lr': get_lr(self.optimizer),
#             'train/dec_lr': self.optimizer.param_groups[1]['lr'],
#         }
#         wandb.log(log_dict, step=self.it)

class STEVENSONMethod(SlotBaseMethod):
    """STEVE model training method."""
    


    def _configure_optimizers(self):
        """Returns an optimizer, a scheduler and its frequency (step/epoch)."""
        # assert self.params.optimizer.lower() == 'adam'
        # assert self.params.weight_decay <= 0.
        lr = self.params.lr
        dec_lr = self.params.dec_lr
        if self.params.weight_decay is None:
            self.params.weight_decay = 0.

        # STEVE uses different lr for its Transformer decoder and other parts
        sa_params = list(
            filter(
                lambda kv: 'trans_decoder' not in kv[0] and kv[1].
                requires_grad, self.model.named_parameters()))
        dec_params = list(
            filter(lambda kv: 'trans_decoder' in kv[0],
                   self.model.named_parameters()))

        params_list = [
            {
                'params': [kv[1] for kv in sa_params],
            },
            {
                'params': [kv[1] for kv in dec_params],
                'lr': dec_lr,
            },
        ]
        if self.params.optimizer.lower() == 'adam':
            optimizer = optim.Adam(params_list, lr=lr, weight_decay=self.params.weight_decay)
        # if self.params.optimizer.lower() == 'lion':
        #     optimizer = Lion(params_list, lr=lr, weight_decay=self.params.weight_decay)
        else:
            optimizer = optim.AdamW(params_list, lr=lr, weight_decay=self.params.weight_decay)

        total_steps = self.params.max_epochs * len(self.train_loader)
        warmup_steps = self.params.warmup_steps_pct * total_steps

        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            total_steps,
            max_lr=(lr, dec_lr),
            min_lr=0.,
            warmup_steps=warmup_steps,
        )

        return optimizer, (scheduler, 'step')

    @torch.no_grad()
    def validation_epoch(self, model, san_check_step=-1):
        """Validate one epoch.

        We aggregate the avg of all statistics and only log once.
        """
        # STEVE's Transformer-based decoder autoregressively reconstructs the
        # video, which is super slow
        # therefore, we only visualize scene decomposition results
        # but don't show the video reconstruction
        # change this if you want to see reconstruction anyways
        self.recon_video = True
        super().validation_epoch(model, san_check_step=san_check_step)

    @staticmethod
    def _make_video(video, recon_video, history_len=None):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(
            torch.stack(
                [
                    video.cpu(),  # original video
                    recon_video.cpu(),  # dVAE gumbel softmax reconstruction
                    
                ],
                dim=1,
            ))  # [T, 3, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i],
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, 3*W]
        return save_video

    @staticmethod
    def _make_slots_video(video, pred_video):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    video.unsqueeze(1),  # [T, 1, 3, H, W]
                    pred_video,  # [T, num_slots, 3, H, W]
                ],
                dim=1,
            ))  # [T, num_slots + 1, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, (num_slots+1)*W]
        return save_video
    
    @staticmethod
    def _make_masks_video(video, pred_video):
        """videos are of shape [T, C, H, W]"""
        out = to_rgb_from_tensor(
            torch.cat(
                [
                    video.unsqueeze(1),  # [T, 1, 3, H, W]
                    pred_video.unsqueeze(2).expand(-1,-1,3,-1,-1),  # [T, num_slots, 1, H, W]
                ],
                dim=1,
            ))  # [T, num_slots + 1, 3, H, W]
        save_video = torch.stack([
            vutils.make_grid(
                out[i].cpu(),
                nrow=out.shape[1],
            ) for i in range(video.shape[0])
        ])  # [T, 3, H, (num_slots+1)*W]
        return save_video

    @torch.no_grad()
    def _sample_video(self, model):
        """model is a simple nn.Module, not warpped in e.g. DataParallel."""
        model.eval()
        model.testing = True  # we only want the slots
        dst = self.val_loader.dataset
        sampled_idx = self._get_sample_idx(self.params.n_samples, dst)
        num_patches = model.num_patches
        n = int(num_patches**0.5)
        results, recon_results,  masks_result = [], [], []
        for i in sampled_idx:
            video = dst.get_video(i.item())['video'].float().to(self.device)
            data_dict = {'img': video[None]}
            out_dict = model(data_dict)
            masks = out_dict['masks'][0]  # [T, num_slots, H, W]
            masked_video = video.unsqueeze(1) * masks.unsqueeze(2)
            # [T, num_slots, C, H, W]
            save_video = self._make_slots_video(video, masked_video)
            masks_video = self._make_masks_video(video, masks)
            results.append(save_video)
            masks_result.append(masks_video)
            if not self.recon_video:
                continue

            # reconstruct the video by autoregressively generating patch tokens
            # using Transformer decoder conditioned on slots
            slots = out_dict['slots'][0]  # [T, num_slots, slot_size]
            all_recon_video, bs = [], 16  # to avoid OOM
            for batch_idx in range(0, slots.shape[0], bs):
                
                _, logits = model.trans_decoder.generate(
                    slots[batch_idx:batch_idx + bs],
                    steps=num_patches,
                    sample=False,
                )
                # [T, patch_size**2, vocab_size] --> [T, vocab_size, h, w]
                logits = logits.transpose(2, 1).unflatten(
                    -1, (n, n)).contiguous().cuda()
                # 1. use logits after gumbel softmax to reconstruct the video
                
                z_logits = F.log_softmax(logits, dim=1)
                z = torch.argmax(z_logits, dim=1)
                recon_video = model.vqvae.decode_tokens(z)
                all_recon_video.append(recon_video)
                
                del logits, z, recon_video
                torch.cuda.empty_cache()

            recon_video = torch.cat(all_recon_video, dim=0)
            # recon_video_hard = torch.cat(all_hard_video, dim=0)
            save_video = self._make_video(video, recon_video)
            recon_results.append(save_video)
            torch.cuda.empty_cache()

        log_dict = {'val/video': self._convert_video(results),
                    'val/masks_video': self._convert_video(masks_result)}
        if self.recon_video:
            log_dict['val/recon_video'] = self._convert_video(recon_results)
        wandb.log(log_dict, step=self.it)
        torch.cuda.empty_cache()
        model.testing = False

    def _log_train(self, out_dict):
        """Log statistics in training to wandb."""
        super()._log_train(out_dict)

        if self.local_rank != 0 or (self.epoch_it + 1) % self.print_iter != 0:
            return

        # log all the lr
        log_dict = {
            'train/lr': get_lr(self.optimizer),
            'train/dec_lr': self.optimizer.param_groups[1]['lr'],
        }
        wandb.log(log_dict, step=self.it)
