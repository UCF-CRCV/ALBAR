import torch
from torch.distributions.beta import Beta
    

class StillMixRandomBlending():
    """
    videos (torch.Tensor): Model input videos, float tensor with the
        shape of (B, N, C, H, W) or (B, N, C, T, H, W).
    label (torch.Tensor): Labels are converted from hard labels to soft labels.
    Hard labels are integer tensors with the shape of (B, 1) and all of the
    elements are in the range [0, num_classes - 1].
    Soft labels (probablity distribution over classes) are float tensors
    with the shape of (B, 1, num_classes) and all of the elements are in
    the range [0, 1].

    kwargs (dict, optional): Other keyword argument to be used to
        blending imgs and labels in a mini-batch.
    
    Args:
        num_classes (int): The number of classes.
        alpha (float): Parameters for Beta distribution.
    """

    def __init__(self, prob_aug=0.5, alpha1=200.0, alpha2=200.0):
        super().__init__()
        self.beta = Beta(alpha1, alpha2)
        self.prob_aug = prob_aug

    def do_blending(self, videos, label, **kwargs):
        assert len(kwargs) == 0, f'unexpected kwargs for mixup {kwargs}'
        device = videos.device

        lam = self.beta.sample()
        indicator = len(videos.shape)
        if indicator == 5: ## 2D recognizer
            batch_size, num_clip, channel, h, w = videos.shape
            tmp_video = videos
            sample_from = num_clip
            num_sample = batch_size
            num_elements = channel*h*w
        elif indicator == 6: ## 3D recognizer
            batch_size, num_clip, channel, clip_len, h, w = videos.shape
            tmp_video = videos.permute(0,1,3,2,4,5).contiguous()
            sample_from = clip_len
            num_sample = batch_size*num_clip
            num_elements = channel*h*w
        else:
            assert False, "Invalid video size."

        if batch_size == 1:
            return videos, label, torch.arange(0, batch_size).to(device)

        ## randomly choose one frame for each video
        # frame_sample_index = torch.stack([torch.randint(0, sample_from, (num_sample,)).unsqueeze(-1)]*num_elements, dim=-1).cuda()
        frame_sample_index = torch.stack([torch.randint(0, sample_from, (num_sample,)).unsqueeze(-1)]*num_elements, dim=-1).to(device)
        frames = tmp_video.view(num_sample, sample_from, num_elements).contiguous().gather(1,frame_sample_index).view(batch_size, num_sample//batch_size, 1, channel, h, w)
        ## randomly choose one frame to mix
        frames_index = torch.stack([torch.arange(0,batch_size)]*batch_size, dim=0)
        frames_index = frames_index.view(batch_size**2)[:-1].view(batch_size-1, batch_size+1)[:,1:].contiguous().view(batch_size, batch_size-1)
        sample_index = torch.randint(0, batch_size-1, (batch_size, 1))
        sampled_frames_index = frames_index.gather(1, sample_index).view(-1)
        sampled_frames = frames[sampled_frames_index].view(batch_size, 1, num_sample//batch_size, 1, channel, h, w).squeeze(1)

        if indicator == 5:
            sampled_frames = sampled_frames.squeeze(1)
        elif indicator == 6:
            sampled_frames = sampled_frames.permute(0,1,3,2,4,5).contiguous()

        ## mix
        mixed_videos = lam * videos + (1 - lam) * sampled_frames
        ## choose samples according to prob
        if self.prob_aug < 1:
            rand_batch = torch.rand(batch_size)
            aug_ind = torch.where(rand_batch < self.prob_aug)
            ori_ind = torch.where(rand_batch >= self.prob_aug)
            new_order = torch.cat([aug_ind[0], ori_ind[0]], dim=0)
            all_videos = torch.cat([mixed_videos[aug_ind], videos[ori_ind]], dim=0).contiguous()
            # all_label = torch.cat([label[aug_ind], label[ori_ind]], dim=0).contiguous()
            all_label = label[new_order]
        else:
            all_videos = mixed_videos
            all_label = label

        return all_videos, all_label, new_order
    
    def __call__(self, imgs, label, **kwargs):
        return self.do_blending(imgs, label, **kwargs)
