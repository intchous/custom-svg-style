import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from diffusers import DDPMScheduler, DDIMScheduler

from svg_ldm.Vecdiffusion.util.diffusion_utils import make_beta_schedule, make_ddim_timesteps, make_ddim_sampling_parameters, extract, q_sample, p_sample_loop_cond, ddim_sample_loop
from svg_ldm.Vecdiffusion.util.backbone import TransformerEncoder


def get_text_embeds(prompt, text_encoder, tokenizer, use_glb=False, device="cuda"):
    # prompt: [str]
    inputs = tokenizer(prompt, padding='max_length',
                       max_length=tokenizer.model_max_length, return_tensors='pt', truncation=True)

    if (use_glb):
        embeddings = text_encoder(inputs.input_ids.to(device))[1]
    else:
        embeddings = text_encoder(inputs.input_ids.to(device))[0]

    return embeddings


def encode_prompt(prompt, text_encoder, tokenizer, negative_prompt=None, do_classifier_free_guidance=True, use_glb=False, device="cuda"):
    # text conditional embed
    prompt_embeds = get_text_embeds(
        prompt=prompt, text_encoder=text_encoder, tokenizer=tokenizer, use_glb=use_glb, device=device)

    if do_classifier_free_guidance:
        if negative_prompt is None:
            uncond_tokens = [""]
        elif isinstance(negative_prompt, str):
            uncond_tokens = [negative_prompt]
        else:
            uncond_tokens = negative_prompt

        # unconditional embed
        negative_prompt_embeds = get_text_embeds(
            prompt=uncond_tokens, text_encoder=text_encoder, tokenizer=tokenizer, use_glb=use_glb, device=device)

        concat_prompt_embeds = torch.cat(
            [negative_prompt_embeds, prompt_embeds])

        return concat_prompt_embeds

    return prompt_embeds


class Diffusion_ddpm(nn.Module):
    def __init__(self, num_timesteps=1000, nhead=8, feature_dim=2048, dim_transformer=512, num_layers=4, max_len=25, seq_dim=32, device='cuda', beta_schedule='cosine', ddim_num_steps=50, condition='None', encoder_hid_dim=None, cross_attention_dim=None, use_norm_out=False):

        super().__init__()

        self.device = device

        self.num_timesteps = num_timesteps
        betas = make_beta_schedule(
            schedule=beta_schedule, num_timesteps=self.num_timesteps, start=0.0001, end=0.02)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_cumprod)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1).to(self.device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coeff_2 = (torch.sqrt(
            alphas) * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = posterior_variance
        self.logvar = betas.log()

        self.condition = condition
        self.seq_dim = seq_dim
        self.num_class = seq_dim - 4

        self.model = TransformerEncoder(num_layers=num_layers, max_len=max_len, dim_seq=seq_dim, dim_transformer=dim_transformer, nhead=nhead, dim_feedforward=feature_dim,
                                        diffusion_step=num_timesteps, encoder_hid_dim=encoder_hid_dim, cross_attention_dim=cross_attention_dim, use_norm_out=use_norm_out, device=device)

        # self.ddim_num_steps = ddim_num_steps
        # self.make_ddim_schedule(ddim_num_steps=ddim_num_steps)

        self.ddpm_scheduler = DDPMScheduler(
            # linear, scaled_linear, squaredcos_cap_v2
            # beta_schedule="squaredcos_cap_v2",

            # beta_schedule="scaled_linear",
            # beta_start=0.00085,
            # beta_end=0.012,
            # steps_offset=1,
            clip_sample=False,
        )

        self.test_ddpm_scheduler = DDPMScheduler(
            clip_sample=False,
        )

        self.test_ddim_scheduler = DDIMScheduler(
            set_alpha_to_one=False,
            clip_sample=False,
        )

    def load_diffusion_net(self, net_state_dict):
        self.model.load_state_dict(net_state_dict, strict=True)

    def make_ddim_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        self.ddim_num_steps = ddim_num_steps
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps, num_ddpm_timesteps=self.num_timesteps)

        assert self.alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        def to_torch(x): return x.clone().detach().to(
            torch.float32).to(self.device)

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=self.alphas_cumprod, ddim_timesteps=self.ddim_timesteps, eta=ddim_eta)

        self.ddim_sigmas = ddim_sigmas
        self.ddim_alphas = ddim_alphas
        self.ddim_alphas_prev = ddim_alphas_prev

    def compute_and_plot_weighing(self):
        # Define parameters
        T = 1000

        # Time steps from 0 to 998
        t_all = torch.arange(0, T - 1)  # [0, 1, ..., 998]
        batch_size = len(t_all)
        x = torch.zeros(batch_size, 1, 1)

        # Compute sqrt_one_minus_alpha_bar_t
        sqrt_one_minus_alpha_bar_t = extract(
            self.one_minus_alphas_bar_sqrt, t_all, x)
        sqrt_alpha_bar_t = torch.sqrt(1 - sqrt_one_minus_alpha_bar_t.square())

        # Compute weighing
        epsilon = 1e-8
        weighing = (
            sqrt_alpha_bar_t / torch.clamp(sqrt_one_minus_alpha_bar_t, min=epsilon)) ** 2
        max_weighing = 1.0 * 1e8
        weighing = torch.clamp(weighing, max=max_weighing)
        weighing = weighing.view(-1).cpu().numpy()
        print("weighing:", weighing)

        # Plot the graph
        plt.figure(figsize=(10, 6))
        plt.plot(t_all.numpy(), weighing)
        plt.xlabel('t')
        plt.ylabel('weighing')
        plt.title('Weighing vs t')
        plt.grid(True)
        plt.savefig('weighing_vs_t.png')
        plt.close()

    def sample_t(self, size=(1,), t_max=None):
        """Samples batches of time steps to use."""
        if t_max is None:
            t_max = int(self.num_timesteps) - 1

        t = torch.randint(low=0, high=t_max, size=size, device=self.device)

        return t.to(self.device)

    def forward_t(self, l_0_batch, t, encoder_hidden_states=None, reparam=False):

        batch_size = l_0_batch.shape[0]
        e = torch.randn_like(l_0_batch).to(l_0_batch.device)

        l_t_noise = q_sample(l_0_batch, self.alphas_bar_sqrt,
                             self.one_minus_alphas_bar_sqrt, t, noise=e)

        l_t_input_all = l_t_noise
        e_all = e
        t_all = t

        eps_theta = self.model(hidden_states=l_t_input_all,
                               encoder_hidden_states=encoder_hidden_states, timesteps=t_all)

        if reparam:
            sqrt_one_minus_alpha_bar_t = extract(
                self.one_minus_alphas_bar_sqrt, t_all, l_t_input_all)

            sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()

            l_0_generate_reparam = 1 / sqrt_alpha_bar_t * \
                (l_t_input_all - eps_theta *
                 sqrt_one_minus_alpha_bar_t).to(self.device)

            # avoid nan
            # weighing = (sqrt_alpha_bar_t / sqrt_one_minus_alpha_bar_t) ** 2
            epsilon = 1e-8
            weighing = (
                sqrt_alpha_bar_t / torch.clamp(sqrt_one_minus_alpha_bar_t, min=epsilon)) ** 2
            max_weighing = 1.0 * 1e8
            weighing = torch.clamp(weighing, max=max_weighing)

            return eps_theta, e_all, l_0_generate_reparam, weighing
        else:
            return eps_theta, e_all, None, None

    def forward_t_ddpm_scheduler(self, l_0_batch, t, encoder_hidden_states=None):
        batch_size = l_0_batch.shape[0]
        e = torch.randn_like(l_0_batch).to(l_0_batch.device)

        l_t_noise = self.ddpm_scheduler.add_noise(
            original_samples=l_0_batch, noise=e, timesteps=t)

        l_t_input_all = l_t_noise
        e_all = e
        t_all = t

        eps_theta = self.model(hidden_states=l_t_input_all,
                               encoder_hidden_states=encoder_hidden_states, timesteps=t_all)

        # 1. compute alphas, betas
        alpha_prod_t = self.ddpm_scheduler.alphas_cumprod[t]
        alpha_prod_t = alpha_prod_t.unsqueeze(1).unsqueeze(1)
        beta_prod_t = 1 - alpha_prod_t

        pred_original_sample = (l_t_noise - beta_prod_t **
                                (0.5) * eps_theta) / alpha_prod_t ** (0.5)

        # avoid nan
        # weighing = alpha_prod_t / beta_prod_t
        epsilon = 1e-8
        weighing = alpha_prod_t / torch.clamp(beta_prod_t, min=epsilon)
        max_weighing = 1.0 * 1e8
        weighing = torch.clamp(weighing, max=max_weighing)

        l_0_generate_reparam = pred_original_sample.to(self.device)

        return eps_theta, e_all, l_0_generate_reparam, weighing

    def reverse_ddpm(self, batch_size=1, encoder_hidden_states=None, only_last_sample=True, stochastic=True, max_len=25, seq_dim=32):

        self.model.eval()

        svg_t_0 = p_sample_loop_cond(model=self.model, batch_size=batch_size, n_steps=self.num_timesteps, alphas=self.alphas, one_minus_alphas_bar_sqrt=self.one_minus_alphas_bar_sqrt,
                                     encoder_hidden_states=encoder_hidden_states, only_last_sample=only_last_sample, stochastic=stochastic, max_len=max_len, seq_dim=seq_dim)

        return svg_t_0

    def reverse_ddpm_lc(self, batch_size=1, encoder_hidden_states=None, stochastic=True, max_len=25, seq_dim=32, guidance_scale=1.0, l_t=None):

        self.model.eval()
        device = self.device

        if l_t is None:
            l_t = stochastic * \
                torch.randn_like(torch.zeros(
                    [batch_size, max_len, seq_dim])).to(device)

        n_steps = self.num_timesteps
        alphas = self.alphas
        one_minus_alphas_bar_sqrt = self.one_minus_alphas_bar_sqrt
        for i_t in reversed(range(0, n_steps-1)):
            # y_{t-1}
            # l_t = p_sample_cond(model, l_t, t, alphas, one_minus_alphas_bar_sqrt, encoder_hidden_states=encoder_hidden_states, stochastic=stochastic)

            """
            Reverse diffusion process sampling -- one time step.
            y: sampled y at time step t, y_t.
            """

            y_t = l_t

            # t = torch.tensor([i_t]).to(device)
            t = torch.full((batch_size,), i_t, device=device,
                           dtype=torch.long).to(device)

            alpha_t = extract(alphas, t, y_t)
            sqrt_one_minus_alpha_bar_t = extract(
                one_minus_alphas_bar_sqrt, t, y_t)

            sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()

            if (guidance_scale > 1.0):
                y_tt = torch.cat([l_t] * 2)
                tt = torch.cat([t] * 2)

                eps_theta = self.model(
                    hidden_states=y_tt, encoder_hidden_states=encoder_hidden_states, timesteps=tt).to(device).detach()

                # perform guidance
                noise_pred_uncond, noise_pred_cond = eps_theta.chunk(2)
                eps_theta = noise_pred_uncond + guidance_scale * \
                    (noise_pred_cond - noise_pred_uncond)

            else:
                eps_theta = self.model(
                    hidden_states=y_t, encoder_hidden_states=encoder_hidden_states, timesteps=t).to(device).detach()

            # y_0 reparameterization
            y_0_reparam = 1 / sqrt_alpha_bar_t * \
                (y_t - eps_theta * sqrt_one_minus_alpha_bar_t).to(device)

            if (i_t == 0):
                y_t_m_1 = y_0_reparam
            else:
                z = stochastic * torch.randn_like(y_t)

                sqrt_one_minus_alpha_bar_t_m_1 = extract(
                    one_minus_alphas_bar_sqrt, t - 1, y_t)
                sqrt_alpha_bar_t_m_1 = (
                    1 - sqrt_one_minus_alpha_bar_t_m_1.square()).sqrt()

                # y_t_m_1 posterior mean component coefficients
                gamma_0 = (1 - alpha_t) * sqrt_alpha_bar_t_m_1 / \
                    (sqrt_one_minus_alpha_bar_t.square())
                gamma_1 = (sqrt_one_minus_alpha_bar_t_m_1.square()) * \
                    (alpha_t.sqrt()) / (sqrt_one_minus_alpha_bar_t.square())

                # posterior mean
                y_t_m_1_hat = gamma_0 * y_0_reparam + gamma_1 * y_t
                # posterior variance
                beta_t_hat = (sqrt_one_minus_alpha_bar_t_m_1.square(
                )) / (sqrt_one_minus_alpha_bar_t.square()) * (1 - alpha_t)

                y_t_m_1 = y_t_m_1_hat.to(
                    device) + beta_t_hat.sqrt().to(device) * z.to(device)

                l_t = y_t_m_1

        # svg_t_0 = p_sample_t_1to0_cond(model=model, y_t=l_t, one_minus_alphas_bar_sqrt=one_minus_alphas_bar_sqrt, encoder_hidden_states=encoder_hidden_states)

        # assert i_t == 0, 'i_t should be 0'
        svg_t_0 = y_t_m_1

        return svg_t_0

    def reverse_ddpm_scheduler(self, batch_size=1, encoder_hidden_states=None, stochastic=True, max_len=25, seq_dim=32, guidance_scale=1.0, l_t=None):

        self.model.eval()
        device = self.device

        self.test_ddpm_scheduler.set_timesteps(self.num_timesteps)
        # self.test_ddpm_scheduler.timesteps tensor([999, 998,..., 1, 0])

        if l_t is None:
            l_t = stochastic * \
                torch.randn_like(torch.zeros(
                    [batch_size, max_len, seq_dim])).to(device)

        for i, t in enumerate(self.test_ddpm_scheduler.timesteps):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.

            t_cuda = torch.tensor([t]).to(device)

            if (guidance_scale > 1.0):
                y_tt = torch.cat([l_t] * 2)
                tt = torch.cat([t_cuda] * 2)

                eps_theta = self.model(
                    hidden_states=y_tt, encoder_hidden_states=encoder_hidden_states, timesteps=tt).to(device).detach()

                # perform guidance
                noise_pred_uncond, noise_pred_cond = eps_theta.chunk(2)
                eps_theta = noise_pred_uncond + guidance_scale * \
                    (noise_pred_cond - noise_pred_uncond)

            else:
                y_t = l_t
                eps_theta = self.model(
                    hidden_states=y_t, encoder_hidden_states=encoder_hidden_states, timesteps=t_cuda).to(device).detach()

            # compute the previous noisy sample x_t -> x_t-1
            # model_output: eps_theta, sample: l_t
            l_t = self.test_ddpm_scheduler.step(eps_theta, t, l_t)[
                'prev_sample']

        svg_t_0 = l_t
        return svg_t_0

    def reverse_ddim(self, batch_size=4, encoder_hidden_states=None, stochastic=True, max_len=25, seq_dim=32):

        self.model.eval()

        svg_t_0, intermediates = ddim_sample_loop(model=self.model, batch_size=batch_size, timesteps=self.ddim_timesteps, ddim_alphas=self.ddim_alphas,
                                                  ddim_alphas_prev=self.ddim_alphas_prev, ddim_sigmas=self.ddim_sigmas, encoder_hidden_states=encoder_hidden_states, stochastic=stochastic, seq_len=max_len, seq_dim=seq_dim)

        return svg_t_0, intermediates

    def reverse_ddim_lc(self, batch_size=1, encoder_hidden_states=None, stochastic=True, max_len=25, seq_dim=32, guidance_scale=1.0, l_t=None):

        self.model.eval()
        device = self.device

        ddim_timesteps = self.ddim_timesteps
        ddim_alphas = self.ddim_alphas
        ddim_alphas_prev = self.ddim_alphas_prev
        ddim_sigmas = self.ddim_sigmas

        if l_t is None:
            l_t = stochastic * \
                torch.randn_like(torch.zeros(
                    [batch_size, max_len, seq_dim])).to(device)

        # intermediates = {'y_inter': [l_t], 'pred_y0': [l_t]}
        time_range = np.flip(ddim_timesteps)
        total_steps = ddim_timesteps.shape[0]
        # print(f"Running DDIM Sampling with {total_steps} timesteps")

        for i, step in enumerate(time_range):
            index = total_steps - i - 1
            t = torch.full((batch_size,), step,
                           device=device, dtype=torch.long)

            y_t = l_t
            t = t.to(device)

            if (guidance_scale > 1.0):
                y_tt = torch.cat([l_t] * 2)
                tt = torch.cat([t] * 2)

                e_t = self.model(hidden_states=y_tt, encoder_hidden_states=encoder_hidden_states, timesteps=tt).to(
                    device).detach()

                # perform guidance
                noise_pred_uncond, noise_pred_cond = e_t.chunk(2)
                e_t = noise_pred_uncond + guidance_scale * \
                    (noise_pred_cond - noise_pred_uncond)

            else:
                e_t = self.model(hidden_states=y_t, encoder_hidden_states=encoder_hidden_states, timesteps=t).to(
                    device).detach()

            sqrt_one_minus_alphas = torch.sqrt(1. - ddim_alphas)
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full(e_t.shape, ddim_alphas[index], device=device)
            a_t_m_1 = torch.full(
                e_t.shape, ddim_alphas_prev[index], device=device)
            sigma_t = torch.full(e_t.shape, ddim_sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(
                e_t.shape, sqrt_one_minus_alphas[index], device=device)

            # direction pointing to x_t
            dir_b_t = (1. - a_t_m_1 - sigma_t ** 2).sqrt() * e_t
            noise = sigma_t * torch.randn_like(l_t).to(device)

            # reparameterize x_0
            b_0_reparam = (l_t - sqrt_one_minus_at * e_t) / a_t.sqrt()

            # compute b_t_m_1
            b_t_m_1 = a_t_m_1.sqrt() * b_0_reparam + 1 * dir_b_t + noise

            l_t = b_t_m_1
            pred_y0 = b_0_reparam

        svg_t_0 = l_t
        return svg_t_0

    def reverse_ddim_scheduler(self, batch_size=1, encoder_hidden_states=None, stochastic=True, max_len=25, seq_dim=32, guidance_scale=1.0, l_t=None):

        self.model.eval()
        device = self.device

        self.test_ddim_scheduler.set_timesteps(self.ddim_num_steps)

        if l_t is None:
            l_t = stochastic * \
                torch.randn_like(torch.zeros(
                    [batch_size, max_len, seq_dim])).to(device)

        for i, t in enumerate(self.test_ddim_scheduler.timesteps):

            t_cuda = torch.tensor([t]).to(device)

            if (guidance_scale > 1.0):
                y_tt = torch.cat([l_t] * 2)
                tt = torch.cat([t_cuda] * 2)

                e_t = self.model(hidden_states=y_tt, encoder_hidden_states=encoder_hidden_states, timesteps=tt).to(
                    device).detach()

                # perform guidance
                noise_pred_uncond, noise_pred_cond = e_t.chunk(2)
                e_t = noise_pred_uncond + guidance_scale * \
                    (noise_pred_cond - noise_pred_uncond)

            else:
                y_t = l_t
                e_t = self.model(hidden_states=y_t, encoder_hidden_states=encoder_hidden_states, timesteps=t_cuda).to(
                    device).detach()

            # compute the previous noisy sample x_t -> x_t-1
            # model_output: eps_theta, sample: l_t
            l_t = self.test_ddim_scheduler.step(e_t, t, l_t)['prev_sample']

        svg_t_0 = l_t
        return svg_t_0
    # -----------------------------------------------
