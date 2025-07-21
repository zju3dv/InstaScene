from diffusers import StableDiffusionInpaintPipeline
from diffusers.utils.torch_utils import randn_tensor
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np


class MultiviewInpainter:
    def __init__(
            self,
            half_precision_weights: bool = True,
            pipeline_name: str = "runwayml/stable-diffusion-inpainting"):
        self.dtype = torch.float16 if half_precision_weights else torch.float32
        pipe_kwargs = {
            "safety_checker": None,
            "feature_extractor": None,
            "requires_safety_checker": False,
            "torch_dtype": torch.float16 if half_precision_weights else torch.float32,
        }

        self.inpaint_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            pipeline_name, **pipe_kwargs
        ).to("cuda")

        self.tokenizer = self.inpaint_pipeline.tokenizer
        self.text_encoder = self.inpaint_pipeline.text_encoder.eval()

        self.unet = self.inpaint_pipeline.unet.eval()
        self.vae = self.inpaint_pipeline.vae.eval()

        self.vae_scale_factor = 2 ** (len(self.inpaint_pipeline.vae.config.block_out_channels) - 1)
        self.vae_latent_channels = self.inpaint_pipeline.vae.config.latent_channels

        self.scheduler = self.inpaint_pipeline.scheduler

        self.num_train_timesteps = self.scheduler.num_train_timesteps

        self.device = "cuda:0"
        torch.cuda.empty_cache()

    def compute_text_embeddings(self, prompt, negative_prompt):
        with torch.no_grad():
            text_inputs = self.tokenizer(
                prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            prompt_embeds = self.text_encoder(text_inputs.input_ids.cuda(), attention_mask=None)[0]

            negative_text_inputs = self.tokenizer(
                negative_prompt,
                truncation=True,
                padding="max_length",
                max_length=self.tokenizer.model_max_length,
                return_tensors="pt",
            )
            negative_prompt_embeds = self.text_encoder(negative_text_inputs.input_ids.cuda(), attention_mask=None)[0]

        return prompt_embeds, negative_prompt_embeds

    def encode_images(self, imgs):
        imgs = imgs * 2.0 - 1.0
        sampled_posterior = self.vae.encode(imgs.to(self.device), return_dict=False)[0].sample().to(self.device)
        latents = sampled_posterior * 0.18215
        return latents

    def decode_latents(self, latents):
        scaled_latents = 1 / 0.18215 * latents
        image = self.vae.decode(scaled_latents.to(self.device), return_dict=False)[0].to(self.device)
        image = (image * 0.5 + 0.5).clamp(0, 1)
        return image

    def get_model_input(self,
                        images, masks,
                        generator=None,
                        starting_image=None,
                        starting_timestep=None):
        batch_size, _, height, width = images.shape

        noise = randn_tensor(
            shape=(
                batch_size,
                self.vae_latent_channels,
                height // self.vae_scale_factor,
                width // self.vae_scale_factor,
            ),
            generator=generator,
            device=torch.device(self.device),
            dtype=self.dtype,
        )
        # 渲染图像
        if starting_image is not None:  # 当前渲染的图
            with torch.no_grad():
                latents = self.encode_images(starting_image)  # 当前渲染图像
            latents = self.scheduler.add_noise(latents, noise, starting_timestep)  # [8,4,32,32]
        else:
            latents = noise
        # 将mask也降采样8倍
        latents_mask = torch.nn.functional.interpolate(
            masks,
            size=(height // self.vae_scale_factor, width // self.vae_scale_factor),
            mode="nearest",
        ).to(device=self.device, dtype=self.dtype)

        masked_image = torch.where(masks == 0, images, 0.5)
        with torch.no_grad():
            masked_image_latents = self.encode_images(masked_image)  #

        # uncondition_latent
        latents_mask_uncond = torch.ones_like(latents_mask)
        masked_image_uncond = torch.ones_like(masked_image) * 0.5
        with torch.no_grad():
            masked_image_latents_uncond = self.encode_images(masked_image_uncond)

        return {
            "latents": latents.to(device=self.device, dtype=self.dtype).detach().requires_grad_(False),
            "latents_mask": latents_mask.to(device=self.device, dtype=self.dtype),
            "masked_image_latents": masked_image_latents.to(device=self.device, dtype=self.dtype),
            # classifier guidance
            "latents_mask_uncond": latents_mask_uncond.to(device=self.device, dtype=self.dtype),
            "masked_image_latents_uncond": masked_image_latents_uncond.to(device=self.device, dtype=self.dtype),
            "noise": noise.to(device=self.device, dtype=self.dtype),
        }

    def forward_unet(self, sample, t, text_embeddings):

        def make_grid(tensors):
            """
            The batch size needs to be divisible by 4.
            Wraps with row major format.
            """
            batch_size, C, H, W = tensors.shape
            assert batch_size % 4 == 0
            num_grids = batch_size // 4
            t = tensors.view(num_grids, 4, C, H, W).transpose(0, 1)
            tensor = torch.cat(
                [
                    torch.cat([t[0], t[1]], dim=-1),
                    torch.cat([t[2], t[3]], dim=-1),
                ],
                dim=-2,
            )
            return tensor

        def undo_grid(tensors):
            batch_size, C, H, W = tensors.shape
            num_squares = batch_size * 4
            hh = H // 2
            hw = W // 2
            t = tensors.view(batch_size, C, 2, hh, 2, hw).permute(0, 2, 4, 1, 3, 5)
            t = t.reshape(num_squares, C, hh, hw)
            return t

        # process embeddings
        prompt_embeds, negative_prompt_embeds = text_embeddings

        batch_size = sample.shape[0] // 3

        prompt_embeds = torch.cat(
            [
                prompt_embeds.repeat(batch_size, 1, 1),
                negative_prompt_embeds.repeat(batch_size, 1, 1),
                negative_prompt_embeds.repeat(batch_size, 1, 1),
            ]
        )

        grid_sample = make_grid(sample)  # [将32,32拼成64,64]
        grid_prompt_embeds = prompt_embeds[:3].repeat(grid_sample.shape[0] // 3, 1, 1)
        noise_pred = self.unet(
            sample=grid_sample,  # 拼成512，512
            timestep=t,
            encoder_hidden_states=grid_prompt_embeds,
            return_dict=False,
        )[0]
        noise_pred = undo_grid(noise_pred)
        return noise_pred

    def get_noise_pred(self, t, model_input,
                       text_embeddings,
                       text_guidance_scale, image_guidance_scale,
                       denoise_in_grid=True,
                       multidiffusion_steps=8,
                       randomize_latents=True,  # 将图像打乱
                       generator=None):
        batch_size = model_input["latents"].shape[0]
        value = torch.zeros_like(model_input["latents"])
        count = torch.zeros_like(model_input["latents"])

        for i in tqdm(range(multidiffusion_steps)):  # 遍历8次
            if randomize_latents:  # False True
                indices = torch.randperm(batch_size)  # 将顺序打乱
            else:
                indices = torch.arange(batch_size)

            latents = model_input["latents"][indices]
            latents_mask = model_input["latents_mask"][indices]
            # uncond latent
            latents_mask_uncond = model_input["latents_mask_uncond"][indices]  # ones_like
            masked_image_latents = model_input["masked_image_latents"][indices]
            masked_image_latents_uncond = model_input["masked_image_latents_uncond"][indices]
            # 为什么这里是3倍
            latent_model_input = torch.cat([latents, latents, latents])  # uncond,textcond,imagecond
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            latents_mask_input = torch.cat([latents_mask, latents_mask, latents_mask_uncond])
            masked_image_latents_input = torch.cat(
                [
                    masked_image_latents,
                    masked_image_latents,
                    masked_image_latents_uncond,
                ]
            )
            latent_model_input_cat = torch.cat(
                [latent_model_input, latents_mask_input, masked_image_latents_input],
                dim=1,
            )  # [24,4,32,32],[24,1,32,32],[24,4,32,32]

            noise_pred_all = self.forward_unet(
                sample=latent_model_input_cat,  # [8,4,32,32]->[]
                t=t,
                text_embeddings=text_embeddings
            )
            # 多一个noise_image
            noise_pred_text, noise_pred_image, noise_pred_uncond = noise_pred_all.chunk(3)

            noise_pred = (
                    noise_pred_image
                    + text_guidance_scale * (noise_pred_text - noise_pred_image)
                    + image_guidance_scale * (noise_pred_image - noise_pred_uncond)
            )

            value[indices] += noise_pred
            count[indices] += 1

        final_noise_pred = torch.where(count > 0, value / count, value)

        scheduler_output = self.scheduler.step(final_noise_pred, t, model_input["latents"], generator=generator)
        pred_prev_sample = scheduler_output.prev_sample
        pred_original_sample = scheduler_output.pred_original_sample

        return pred_prev_sample, pred_original_sample, final_noise_pred

    def multiview_inpaint(self, origin_image, masks, render_image,
                          text_embeddings,
                          multidiffusion_steps=8,
                          num_inference_steps=20,
                          text_guidance_scale=0,
                          image_guidance_scale=1.5,
                          seed=9,
                          starting_lower_bound=0.4, starting_upper_bound=0.9):
        # get image
        # 对图像进行降采样512->256
        batch_size, _, height, width = origin_image.shape
        generator = torch.Generator("cuda").manual_seed(seed)

        # 当前的去噪stage
        min_step = int(self.num_train_timesteps * starting_lower_bound)
        max_step = int(self.num_train_timesteps * starting_upper_bound)  # 一样的。。
        # select t, set multi-step diffusion
        T = torch.randint(min_step, max_step + 1, [1], dtype=torch.long, device=self.device)
        self.scheduler.config.num_train_timesteps = T.item()

        self.scheduler.set_timesteps(num_inference_steps, device=self.device)

        # 将图像转成latent
        model_input = self.get_model_input(images=origin_image,
                                           masks=masks,
                                           generator=generator,
                                           starting_image=render_image,
                                           starting_timestep=self.scheduler.timesteps[0])
        # 开始去噪
        for i, t in tqdm(enumerate(self.scheduler.timesteps)):
            # 预测噪声
            with torch.no_grad():
                _, pred_original_sample, noise_pred = self.get_noise_pred(  # 预测噪声
                    t,  # 去噪
                    model_input,  # latents
                    text_embeddings,  # text_embedding
                    text_guidance_scale=text_guidance_scale,  # 0
                    image_guidance_scale=image_guidance_scale,  # 1.5
                    multidiffusion_steps=multidiffusion_steps,  # 1
                    generator=generator
                )

                model_input["latents"] = model_input["latents"].detach().requires_grad_(False)
                scheduler_output = self.scheduler.step(noise_pred, t, model_input["latents"], generator=generator)
                model_input["latents"] = scheduler_output.prev_sample  # 更新latents

            if i % 50 == 0:
                with torch.no_grad():
                    x0 = self.decode_latents(model_input["latents"].detach()).to(torch.float32)
                    Image.fromarray(np.uint8(
                        255.0 * torch.cat([image.permute(1, 2, 0) for image in x0], 1).cpu().numpy())).show()

        with torch.no_grad():
            x0 = self.decode_latents(model_input["latents"].detach()).to(torch.float32)
        return x0

    def pipeline_inpaint(self, prompt, negative_prompt, image, mask_image, num_inference_steps, generator, strength):
        return self.inpaint_pipeline(
            prompt,
            negative_prompt=negative_prompt,
            # "corgi head, detailed, pixar, animated, lego, batman mask",
            image=image, mask_image=mask_image,
            num_inference_steps=num_inference_steps,
            generator=generator,
            strength=strength,  # 越高，噪声添加越多，和原图越不像
        ).images[0].resize((image.size[0], image.size[1]))


# 计算prompt的text_embedding

#
if "__main__" == __name__:
    positive_prompt = "rubbish bin, simple geometry, neat, regulary"
    negative_prompt = "flake, thin"
    mvinpainter = MultiviewInpainter()
    # 计算prompt的text_embedding
    images = torch.load("/home/bytedance/Projects/3DGS/FunctionSplatting++/2d-gaussian-splatting/images.pt")
    masks = torch.load("/home/bytedance/Projects/3DGS/FunctionSplatting++/2d-gaussian-splatting/masks.pt")

    text_embeddings = mvinpainter.compute_text_embeddings(
        positive_prompt, negative_prompt)  # tokenize + encode

    inpainted_images = mvinpainter.multiview_inpaint(
        images.type(torch.float16), masks.type(torch.float16), images.type(torch.float16), text_embeddings,
        multidiffusion_steps=8,
        num_inference_steps=75,
        text_guidance_scale=7.5,
        image_guidance_scale=1.5,
        seed=9,
        starting_lower_bound=0.9,
        starting_upper_bound=0.9
    )
    Image.fromarray(np.uint8(
        255.0 * torch.cat([image.permute(1, 2, 0) for image in inpainted_images], 1).cpu().numpy())).show()
