from ..models import ModelManager
from ..models.wan_video_dit import WanModel
from ..models.wan_video_text_encoder import WanTextEncoder
from ..models.wan_video_vae import WanVideoVAE
from ..models.wan_video_image_encoder import WanImageEncoder
from ..schedulers.flow_match import FlowMatchScheduler
from .base import BasePipeline
from ..prompters import WanPrompter
import torch, os
from einops import rearrange, repeat
import numpy as np
from PIL import Image
from tqdm import tqdm
from typing import Optional

from ..vram_management import enable_vram_management, AutoWrappedModule, AutoWrappedLinear
from ..models.wan_video_text_encoder import T5RelativeEmbedding, T5LayerNorm
from ..models.wan_video_dit import RMSNorm, sinusoidal_embedding_1d
from ..models.wan_video_vae import RMS_norm, CausalConv3d, Upsample



class WanVideoReCamMasterPipelinePacked(BasePipeline):

    def __init__(self, device="cuda", torch_dtype=torch.float16, tokenizer_path=None):
        super().__init__(device=device, torch_dtype=torch_dtype)
        self.scheduler = FlowMatchScheduler(shift=5, sigma_min=0.0, extra_one_step=True)
        self.prompter = WanPrompter(tokenizer_path=tokenizer_path)
        self.text_encoder: WanTextEncoder = None
        self.image_encoder: WanImageEncoder = None
        self.dit: WanModel = None
        self.vae: WanVideoVAE = None
        self.model_names = ['text_encoder', 'dit', 'vae']
        self.height_division_factor = 16
        self.width_division_factor = 16


    def enable_vram_management(self, num_persistent_param_in_dit=None):
        dtype = next(iter(self.text_encoder.parameters())).dtype
        enable_vram_management(
            self.text_encoder,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Embedding: AutoWrappedModule,
                T5RelativeEmbedding: AutoWrappedModule,
                T5LayerNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.dit.parameters())).dtype
        enable_vram_management(
            self.dit,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv3d: AutoWrappedModule,
                torch.nn.LayerNorm: AutoWrappedModule,
                RMSNorm: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
            max_num_param=num_persistent_param_in_dit,
            overflow_module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device="cpu",
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        dtype = next(iter(self.vae.parameters())).dtype
        enable_vram_management(
            self.vae,
            module_map = {
                torch.nn.Linear: AutoWrappedLinear,
                torch.nn.Conv2d: AutoWrappedModule,
                RMS_norm: AutoWrappedModule,
                CausalConv3d: AutoWrappedModule,
                Upsample: AutoWrappedModule,
                torch.nn.SiLU: AutoWrappedModule,
                torch.nn.Dropout: AutoWrappedModule,
            },
            module_config = dict(
                offload_dtype=dtype,
                offload_device="cpu",
                onload_dtype=dtype,
                onload_device=self.device,
                computation_dtype=self.torch_dtype,
                computation_device=self.device,
            ),
        )
        if self.image_encoder is not None:
            dtype = next(iter(self.image_encoder.parameters())).dtype
            enable_vram_management(
                self.image_encoder,
                module_map = {
                    torch.nn.Linear: AutoWrappedLinear,
                    torch.nn.Conv2d: AutoWrappedModule,
                    torch.nn.LayerNorm: AutoWrappedModule,
                },
                module_config = dict(
                    offload_dtype=dtype,
                    offload_device="cpu",
                    onload_dtype=dtype,
                    onload_device="cpu",
                    computation_dtype=dtype,
                    computation_device=self.device,
                ),
            )
        self.enable_cpu_offload()


    def fetch_models(self, model_manager: ModelManager):
        text_encoder_model_and_path = model_manager.fetch_model("wan_video_text_encoder", require_model_path=True)
        if text_encoder_model_and_path is not None:
            self.text_encoder, tokenizer_path = text_encoder_model_and_path
            self.prompter.fetch_models(self.text_encoder)
            self.prompter.fetch_tokenizer(os.path.join(os.path.dirname(tokenizer_path), "google/umt5-xxl"))
        self.dit = model_manager.fetch_model("wan_video_dit")
        self.vae = model_manager.fetch_model("wan_video_vae")
        self.image_encoder = model_manager.fetch_model("wan_video_image_encoder")


    @staticmethod
    def from_model_manager(model_manager: ModelManager, torch_dtype=None, device=None):
        if device is None: device = model_manager.device
        if torch_dtype is None: torch_dtype = model_manager.torch_dtype
        pipe = WanVideoReCamMasterPipelinePacked(device=device, torch_dtype=torch_dtype)
        pipe.fetch_models(model_manager)
        return pipe
    
    
    def denoising_model(self):
        return self.dit


    def encode_prompt(self, prompt, positive=True):
        prompt_emb = self.prompter.encode_prompt(prompt, positive=positive)
        return {"context": prompt_emb}
    
    
    def encode_image(self, image, num_frames, height, width):
        image = self.preprocess_image(image.resize((width, height))).to(self.device)
        clip_context = self.image_encoder.encode_image([image])
        msk = torch.ones(1, num_frames, height//8, width//8, device=self.device)
        msk[:, 1:] = 0
        msk = torch.concat([torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]], dim=1)
        msk = msk.view(1, msk.shape[1] // 4, 4, height//8, width//8)
        msk = msk.transpose(1, 2)[0]
        
        vae_input = torch.concat([image.transpose(0, 1), torch.zeros(3, num_frames-1, height, width).to(image.device)], dim=1)
        y = self.vae.encode([vae_input.to(dtype=self.torch_dtype, device=self.device)], device=self.device)[0]
        y = torch.concat([msk, y])
        y = y.unsqueeze(0)
        clip_context = clip_context.to(dtype=self.torch_dtype, device=self.device)
        y = y.to(dtype=self.torch_dtype, device=self.device)
        return {"clip_feature": clip_context, "y": y}


    def tensor2video(self, frames):
        frames = rearrange(frames, "C T H W -> T H W C")
        frames = ((frames.float() + 1) * 127.5).clip(0, 255).cpu().numpy().astype(np.uint8)
        frames = [Image.fromarray(frame) for frame in frames]
        return frames
    
    
    def prepare_extra_input(self, latents=None):
        return {}
    
    
    def encode_video(self, input_video, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        latents = self.vae.encode(input_video, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return latents
    
    
    def decode_video(self, latents, tiled=True, tile_size=(34, 34), tile_stride=(18, 16)):
        frames = self.vae.decode(latents, device=self.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
        return frames


    @torch.no_grad()
    def __call__(
        self,
        prompt,             # target_text
        negative_prompt="", # 
        source_video=None,  #
        target_camera=None, # 
        input_image=None,   
        input_video=None,
        denoising_strength=1.0,
        seed=None,          # 0
        rand_device="cpu",
        height=480,
        width=832,
        num_frames=81,
        cfg_scale=5.0,      # 5.0
        num_inference_steps=50, # 50
        sigma_shift=5.0,
        tiled=True,         # True
        tile_size=(30, 52),
        tile_stride=(15, 26),
        tea_cache_l1_thresh=None,
        tea_cache_model_id="",
        progress_bar_cmd=tqdm,
        progress_bar_st=None,

        latent_window_size=5,
    ):
        # Parameter check
        height, width = self.check_resize_height_width(height, width)
        if num_frames % 4 != 1:
            num_frames = (num_frames + 2) // 4 * 4 + 1
            print(f"Only `num_frames % 4 != 1` is acceptable. We round it up to {num_frames}.")
        
        # Tiler parameters
        tiler_kwargs = {"tiled": tiled, "tile_size": tile_size, "tile_stride": tile_stride}

        # Scheduler
        self.scheduler.set_timesteps(num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift)
        
        # Encode source video (recammaster)
        self.load_models_to_device(['vae'])
        source_video = source_video.to(dtype=self.torch_dtype, device=self.device)  # [1, 3, 81, 480, 832]
        source_latents = self.encode_video(source_video, **tiler_kwargs).to(dtype=self.torch_dtype, device=self.device)  # [1, 16, 21, 60, 104]
        start_latents = source_latents[:, :, :1, ...]  # [1, 16, 1, 60, 104]

        # Process target camera (recammaster)
        cam_emb = target_camera.to(dtype=self.torch_dtype, device=self.device)  # [1, 21, 12]
        start_cam_emb = cam_emb[:, :1, ...]  # [1, 21, 1, 12]

        # Encode prompts
        self.load_models_to_device(["text_encoder"])
        prompt_emb_posi = self.encode_prompt(prompt, positive=True)
        if cfg_scale != 1.0:
            prompt_emb_nega = self.encode_prompt(negative_prompt, positive=False)
            
        # Encode image
        if input_image is not None and self.image_encoder is not None:
            self.load_models_to_device(["image_encoder", "vae"])
            image_emb = self.encode_image(input_image, num_frames, height, width)
        else:
            image_emb = {} # empty
            
        # Extra input
        # extra_input = self.prepare_extra_input(latents) # empty
        extra_input = {}
        
        # TeaCache
        tea_cache_posi = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}
        tea_cache_nega = {"tea_cache": TeaCache(num_inference_steps, rel_l1_thresh=tea_cache_l1_thresh, model_id=tea_cache_model_id) if tea_cache_l1_thresh is not None else None}


        clean_latents_size, clean_latents_2x_size, clean_latents_4x_size = 1, 2, 8
        history_latents = torch.zeros(size=(1, 16, clean_latents_size + clean_latents_2x_size + clean_latents_4x_size, height//8, width//8), dtype=torch.bfloat16).cpu()
        history_pixels = None
        total_generated_latent_frames = 0


        total_latent_sections = (num_frames - 1) // (latent_window_size * 4)

        latent_paddings = reversed(range(total_latent_sections))

        if total_latent_sections > 4:
            latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]

        frames = []

        for latent_padding in latent_paddings:
            is_last_section = latent_padding == 0
            first_frame_size = int(is_last_section)
            # latent_padding_size = latent_padding * latent_window_size + 1 - first_frame_size
            latent_padding_size = latent_padding * latent_window_size 
            current_window_size = latent_window_size + first_frame_size

            # Initialize noise
            # latent_window_size = 5, 5 * 4 frames per section
            noise = self.generate_noise((1, 16, latent_window_size, height//8, width//8), seed=seed, device=rand_device, dtype=torch.float32)
            noise = noise.to(dtype=self.torch_dtype, device=self.device)
            latents = noise

            # indices = torch.arange(0, sum([latent_padding_size, current_window_size*2, clean_latents_size, clean_latents_2x_size, clean_latents_4x_size]))
            # blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indcies = \
            #     torch.split(indices, [latent_padding_size, current_window_size*2, clean_latents_size, clean_latents_2x_size, clean_latents_4x_size], dim=0)
            # clean_latent_indices = clean_latent_indices_post

            # clean_latents_pre = start_latents.to(history_latents)
            # clean_latents_post, clean_latents_2x, clean_latents_4x = \
            #     history_latents[:, :, :clean_latents_size+clean_latents_2x_size+clean_latents_4x_size, :, :].split([clean_latents_size, clean_latents_2x_size, clean_latents_4x_size], dim=2)
            # clean_latents = clean_latents_post

            indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size*2, clean_latents_size, clean_latents_2x_size, clean_latents_4x_size]))
            clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indcies = \
                torch.split(indices, [1, latent_padding_size, latent_window_size*2, clean_latents_size, clean_latents_2x_size, clean_latents_4x_size], dim=0)
            clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=0)

            clean_latents_pre = start_latents.to(history_latents)
            clean_latents_post, clean_latents_2x, clean_latents_4x = \
                history_latents[:, :, :clean_latents_size+clean_latents_2x_size+clean_latents_4x_size, :, :].split([clean_latents_size, clean_latents_2x_size, clean_latents_4x_size], dim=2)
            clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)

            current_cam_emb = cam_emb[:, 1 + latent_padding_size: 1 + latent_padding_size + latent_window_size, ...]
            cam_emb_post = cam_emb[:, 1 + latent_padding_size + latent_window_size: 1 + latent_padding_size + latent_window_size + 1, ...] if 1 + latent_padding_size + latent_window_size < 21 else torch.zeros_like(start_cam_emb).to(start_cam_emb)
            clean_cam_emb = torch.cat([start_cam_emb, cam_emb_post], dim=1)
            current_source_latents = source_latents[:, :, 1 + latent_padding_size: 1 + latent_padding_size + latent_window_size, ...]

            # Denoise
            self.load_models_to_device(["dit"])
            tgt_latent_len = latents.shape[2]
            for progress_id, timestep in enumerate(progress_bar_cmd(self.scheduler.timesteps)):
                timestep = timestep.unsqueeze(0).to(dtype=self.torch_dtype, device=self.device)

                # latents_input = torch.cat([latents, source_latents], dim=2)
                latents_input = torch.cat([latents, current_source_latents], dim=2)
                # latents_input = latents # [1, 16, 21, 60, 104] -> [1, 16, 5, 60, 104]

                # Inference
                noise_pred_posi = model_fn_wan_video(self.dit, 
                                                     latents_input, 
                                                     timestep=timestep, 
                                                     cam_emb=current_cam_emb, 
                                                     latent_indices=latent_indices,
                                                     clean_latents=clean_latents,
                                                     clean_cam_emb=clean_cam_emb,
                                                     clean_latents_indices=clean_latent_indices,
                                                     clean_latents_2x=clean_latents_2x,
                                                     clean_latents_2x_indices=clean_latent_2x_indices,
                                                     clean_latents_4x=clean_latents_4x,
                                                     clean_latents_4x_indices=clean_latent_4x_indcies,
                                                     **prompt_emb_posi, 
                                                     **image_emb, 
                                                     **extra_input, 
                                                     **tea_cache_posi)
                if cfg_scale != 1.0:
                    noise_pred_nega = model_fn_wan_video(self.dit, 
                                                         latents_input, 
                                                         timestep=timestep, 
                                                         cam_emb=current_cam_emb, 
                                                         latent_indices=latent_indices, 
                                                         clean_latents=clean_latents,
                                                         clean_cam_emb=clean_cam_emb,
                                                         clean_latents_indices=clean_latent_indices,
                                                         clean_latents_2x=clean_latents_2x,
                                                         clean_latents_2x_indices=clean_latent_2x_indices,
                                                         clean_latents_4x=clean_latents_4x,
                                                         clean_latents_4x_indices=clean_latent_4x_indcies,
                                                         **prompt_emb_nega, 
                                                         **image_emb, 
                                                         **extra_input, 
                                                         **tea_cache_nega)
                    noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
                else:
                    noise_pred = noise_pred_posi

                # Scheduler
                assert noise_pred.shape == latents_input.shape
                latents = self.scheduler.step(noise_pred[:,:,:tgt_latent_len,...], self.scheduler.timesteps[progress_id], latents_input[:,:,:tgt_latent_len,...])
                # latents = self.scheduler.step(noise_pred, self.scheduler.timesteps[progress_id], latents_input)

            if is_last_section:
                # add the first frame
                latents = torch.cat([start_latents, latents], dim=2)
            
            total_generated_latent_frames += latents.shape[2]
            history_latents = torch.cat([latents.to(history_latents), history_latents], dim=2)

            real_history_latents = history_latents[:, :, :total_generated_latent_frames, ...]

            self.load_models_to_device(['vae'])
            # Decode
            if history_pixels is None:
                history_pixels = self.decode_video(real_history_latents, **tiler_kwargs)
            else:
                section_latent_frames = current_window_size + latent_window_size
                overlapped_frames = latent_window_size * 4 - 3 # latent to frames

                current_pixels = self.decode_video(real_history_latents[:, :, :section_latent_frames], **tiler_kwargs)
                history_pixels = soft_append_bcthw(current_pixels, history_pixels, overlapped_frames)

            # print("final latents shape:", latents.shape)
            # frames = self.decode_video(latents, **tiler_kwargs)
            # print(f"frames shape: {frames.shape}")
            self.load_models_to_device([])
            
            # frames = self.tensor2video(frames[0])
            frames.append(self.tensor2video(history_pixels[0]))

            if is_last_section:
                break

        return frames



class TeaCache:
    def __init__(self, num_inference_steps, rel_l1_thresh, model_id):
        self.num_inference_steps = num_inference_steps
        self.step = 0
        self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = None
        self.rel_l1_thresh = rel_l1_thresh
        self.previous_residual = None
        self.previous_hidden_states = None
        
        self.coefficients_dict = {
            "Wan2.1-T2V-1.3B": [-5.21862437e+04, 9.23041404e+03, -5.28275948e+02, 1.36987616e+01, -4.99875664e-02],
            "Wan2.1-T2V-14B": [-3.03318725e+05, 4.90537029e+04, -2.65530556e+03, 5.87365115e+01, -3.15583525e-01],
            "Wan2.1-I2V-14B-480P": [2.57151496e+05, -3.54229917e+04,  1.40286849e+03, -1.35890334e+01, 1.32517977e-01],
            "Wan2.1-I2V-14B-720P": [ 8.10705460e+03,  2.13393892e+03, -3.72934672e+02,  1.66203073e+01, -4.17769401e-02],
        }
        if model_id not in self.coefficients_dict:
            supported_model_ids = ", ".join([i for i in self.coefficients_dict])
            raise ValueError(f"{model_id} is not a supported TeaCache model id. Please choose a valid model id in ({supported_model_ids}).")
        self.coefficients = self.coefficients_dict[model_id]

    def check(self, dit: WanModel, x, t_mod):
        modulated_inp = t_mod.clone()
        if self.step == 0 or self.step == self.num_inference_steps - 1:
            should_calc = True
            self.accumulated_rel_l1_distance = 0
        else:
            coefficients = self.coefficients
            rescale_func = np.poly1d(coefficients)
            self.accumulated_rel_l1_distance += rescale_func(((modulated_inp-self.previous_modulated_input).abs().mean() / self.previous_modulated_input.abs().mean()).cpu().item())
            if self.accumulated_rel_l1_distance < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                self.accumulated_rel_l1_distance = 0
        self.previous_modulated_input = modulated_inp
        self.step += 1
        if self.step == self.num_inference_steps:
            self.step = 0
        if should_calc:
            self.previous_hidden_states = x.clone()
        return not should_calc

    def store(self, hidden_states):
        self.previous_residual = hidden_states - self.previous_hidden_states
        self.previous_hidden_states = None

    def update(self, hidden_states):
        hidden_states = hidden_states + self.previous_residual
        return hidden_states



def model_fn_wan_video(
    dit: WanModel,
    x: torch.Tensor,
    timestep: torch.Tensor,
    cam_emb: torch.Tensor,
    context: torch.Tensor,
    clip_feature: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    tea_cache: TeaCache = None,
    latent_indices: Optional[torch.Tensor] = None,
    clean_latents: Optional[torch.Tensor] = None,
    clean_cam_emb: Optional[torch.Tensor] = None,
    clean_latents_indices: Optional[torch.Tensor] = None,
    clean_latents_2x: Optional[torch.Tensor] = None,
    clean_latents_2x_indices: Optional[torch.Tensor] = None,
    clean_latents_4x: Optional[torch.Tensor] = None,
    clean_latents_4x_indices: Optional[torch.Tensor] = None,
    **kwargs,
):
    # print("x shape:", x.shape) # [1, 16, 5, 60, 104]
    t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep))
    t_mod = dit.time_projection(t).unflatten(1, (6, dit.dim))
    context = dit.text_embedding(context)
    
    if dit.has_image_input:
        x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
        clip_embdding = dit.img_emb(clip_feature)
        context = torch.cat([clip_embdding, context], dim=1)

    # print("before patchify x shape:", x.shape) # [1, 16, 42, 60, 104]
    # x, (f, h, w) = dit.patchify(x) # patchify 被放到 framepack 里面了
    # print("after patchify x shape:", x.shape) # [1, 7800, 1536]

    x, freqs, (f, h, w) = frame_pack(dit, x, latent_indices, clean_latents, clean_latents_indices, clean_latents_2x, clean_latents_2x_indices, clean_latents_4x, clean_latents_4x_indices)
    
    for block in dit.blocks:
        x = block(x, context, cam_emb, t_mod, freqs, clean_cam_emb)

    x = dit.head(x, t)
    original_context_lenght = f * h * w
    x = x[:, -original_context_lenght:, :]  # [1, 21*30*52, 1536]
    x = dit.unpatchify(x, (f, h, w)) # [f, h, w] = [21, 30, 52]
    return x

class PatchEmbedForCleanLatents(torch.nn.Module):
    def __init__(self, in_dim, dim):
        super().__init__()
        self.proj = torch.nn.Conv3d(in_dim, dim, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.proj_2x = torch.nn.Conv3d(in_dim, dim, kernel_size=(2, 4, 4), stride=(2, 4, 4))
        self.proj_4x = torch.nn.Conv3d(in_dim, dim, kernel_size=(4, 8, 8), stride=(4, 8, 8))

    @torch.no_grad()
    def initialize_weight_from_another_conv3d(self, another_layer):
        weight = another_layer.weight.detach().clone()
        bias = another_layer.bias.detach().clone()

        sd = {
            'proj.weight': weight.clone(),
            'proj.bias': bias.clone(),
            'proj_2x.weight': repeat(weight, 'b c t h w -> b c (t tk) (h hk) (w wk)', tk=2, hk=2, wk=2) / 8.0,
            'proj_2x.bias': bias.clone(),
            'proj_4x.weight': repeat(weight, 'b c t h w -> b c (t tk) (h hk) (w wk)', tk=4, hk=4, wk=4) / 64.0,
            'proj_4x.bias': bias.clone(),
        }

        sd = {k: v.clone() for k, v in sd.items()}

        self = self.to(dtype=weight.dtype, device=weight.device)
        self.load_state_dict(sd)
        return


def get_rope_freqs(freqs, frame_indices, h, w, device):
    f = frame_indices.shape[0]
    return torch.cat([
        freqs[0][frame_indices].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
        freqs[3][frame_indices].view(f, 1, 1, -1).expand(f, h, w, -1),
        freqs[4][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        freqs[5][:w].view(1, 1, w, -1).expand(f, h, w, -1)
    ], dim=-1).unsqueeze(0).permute(0, 4, 1, 2, 3).to(device)


def frame_pack(dit,
               x, 
               latent_indices, 
               clean_latents, 
               clean_latents_indices, 
               clean_latents_2x, 
               clean_latents_2x_indices, 
               clean_latents_4x, 
               clean_latents_4x_indices):
        # patch_embedder_for_clean_latents = PatchEmbedForCleanLatents(dit.in_dim, dit.dim)
        # patch_embedder_for_clean_latents.initialize_weight_from_another_conv3d(dit.patch_embedding)
        # hidden_states = patch_embedder_for_clean_latents.proj(x) # [1, 1536, 5, 30, 52]
        hidden_states = dit.patch_embedding(x) # [1, 1536, 5, 30, 52]
        B, C, F, H, W = hidden_states.shape
        hidden_states = rearrange(hidden_states, 'b c t h w -> b (t h w) c').contiguous()
        # rope = RotaryPosEmbed((dit.head_dim - 2 * (dit.head_dim // 3), dit.head_dim // 3, dit.head_dim // 3)) # head_dim = 128
        
        # rope_freqs = rope(latent_indices, H, W, device=hidden_states.device)
        rope_freqs = get_rope_freqs(dit.freqs, latent_indices, H, W, device=hidden_states.device)
        rope_freqs = rope_freqs.flatten(2).transpose(1, 2)

        if clean_latents is not None and clean_latents_indices is not None:
            clean_latents = clean_latents.to(hidden_states)
            # clean_latents = patch_embedder_for_clean_latents.proj(clean_latents)
            clean_latents = dit.patch_embedding(clean_latents)
            clean_latents = rearrange(clean_latents, 'b c f h w -> b (f h w) c').contiguous()
            hidden_states = torch.cat([clean_latents, hidden_states], dim=1)

            # clean_latents_rope_freqs = get_rope_freqs(clean_latents_indices)
            clean_latents_rope_freqs = get_rope_freqs(dit.freqs, clean_latents_indices, H, W, device=hidden_states.device)
            clean_latents_rope_freqs = clean_latents_rope_freqs.flatten(2).transpose(1, 2)
            rope_freqs = torch.cat([clean_latents_rope_freqs, rope_freqs], dim=1)

        if clean_latents_2x is not None and clean_latents_2x_indices is not None:
            clean_latents_2x = clean_latents_2x.to(hidden_states)
            clean_latents_2x = pad_for_3d_conv(clean_latents_2x, (2, 4, 4))
            # clean_latents_2x = patch_embedder_for_clean_latents.proj_2x(clean_latents_2x)
            clean_latents_2x = dit.patch_embedding_2x(clean_latents_2x)
            clean_latents_2x = rearrange(clean_latents_2x, 'b c f h w -> b (f h w) c').contiguous()
            hidden_states = torch.cat([clean_latents_2x, hidden_states], dim=1)

            # clean_latents_2x_rope_freqs = get_rope_freqs(clean_latents_2x_indices)
            clean_latents_2x_rope_freqs = get_rope_freqs(dit.freqs, clean_latents_2x_indices, H, W, device=hidden_states.device)
            clean_latents_2x_rope_freqs = pad_for_3d_conv(clean_latents_2x_rope_freqs, (2, 2, 2))
            clean_latents_2x_rope_freqs = center_down_sample_3d(clean_latents_2x_rope_freqs, (2, 2, 2))
            clean_latents_2x_rope_freqs = clean_latents_2x_rope_freqs.flatten(2).transpose(1, 2)
            rope_freqs = torch.cat([clean_latents_2x_rope_freqs, rope_freqs], dim=1)

        if clean_latents_4x is not None and clean_latents_4x_indices is not None:
            clean_latents_4x = clean_latents_4x.to(hidden_states)
            clean_latents_4x = pad_for_3d_conv(clean_latents_4x, (4, 8, 8))
            # clean_latents_4x = patch_embedder_for_clean_latents.proj_4x(clean_latents_4x)
            clean_latents_4x = dit.patch_embedding_4x(clean_latents_4x)
            clean_latents_4x = rearrange(clean_latents_4x, 'b c f h w -> b (f h w) c').contiguous()
            hidden_states = torch.cat([clean_latents_4x, hidden_states], dim=1)

            # clean_latents_4x_rope_freqs = get_rope_freqs(clean_latents_4x_indices)
            clean_latents_4x_rope_freqs = get_rope_freqs(dit.freqs, clean_latents_4x_indices, H, W, device=hidden_states.device)
            clean_latents_4x_rope_freqs = pad_for_3d_conv(clean_latents_4x_rope_freqs, (4, 4, 4))
            clean_latents_4x_rope_freqs = center_down_sample_3d(clean_latents_4x_rope_freqs, (4, 4, 4))
            clean_latents_4x_rope_freqs = clean_latents_4x_rope_freqs.flatten(2).transpose(1, 2)
            rope_freqs = torch.cat([clean_latents_4x_rope_freqs, rope_freqs], dim=1)

        return hidden_states, rope_freqs, (F, H, W)

def pad_for_3d_conv(x, kernel_size):
    b, c, t, h, w = x.shape
    pt, ph, pw = kernel_size
    pad_t = (pt - (t % pt)) % pt
    pad_h = (ph - (h % ph)) % ph
    pad_w = (pw - (w % pw)) % pw
    return torch.nn.functional.pad(x, (0, pad_w, 0, pad_h, 0, pad_t), mode='replicate')

def center_down_sample_3d(x, kernel_size):
    return torch.nn.functional.avg_pool3d(x, kernel_size, stride=kernel_size)

def soft_append_bcthw(history, current, overlap=0):
    if overlap <= 0:
        return torch.cat([history, current], dim=2)

    assert history.shape[2] >= overlap, f"History length ({history.shape[2]}) must be >= overlap ({overlap})"
    assert current.shape[2] >= overlap, f"Current length ({current.shape[2]}) must be >= overlap ({overlap})"
    
    weights = torch.linspace(1, 0, overlap, dtype=history.dtype, device=history.device).view(1, 1, -1, 1, 1)
    blended = weights * history[:, :, -overlap:] + (1 - weights) * current[:, :, :overlap]
    output = torch.cat([history[:, :, :-overlap], blended, current[:, :, overlap:]], dim=2)

    return output.to(history)