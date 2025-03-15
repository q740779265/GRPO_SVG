from typing import List, Union
import warnings

import torch
import torch.distributed
import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import register, Dispatch
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import create_device_mesh, get_sharding_strategy
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager

from dataclasses import dataclass, field
from defusedxml import ElementTree as etree

import io
from statistics import mean
import cairosvg

import clip
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
)

def harmonic_mean(a: float, b: float, beta: float = 1.0) -> float:
    """
    Calculate the harmonic mean of two values, weighted using a beta parameter.

    Args:
        a: First value (e.g., precision)
        b: Second value (e.g., recall)
        beta: Weighting parameter

    Returns:
        Weighted harmonic mean
    """
    # Handle zero values to prevent division by zero
    if a <= 0 or b <= 0:
        return 0.0
    return (1 + beta**2) * (a * b) / (beta**2 * a + b)


def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Converts an SVG string to a PNG image using CairoSVG.

    If the SVG does not define a `viewBox`, it will add one using the provided size.

    Parameters
    ----------
    svg_code : str
        The SVG string to convert.
    size : tuple[int, int], default=(384, 384)
        The desired size of the output PNG image (width, height).

    Returns
    -------
    PIL.Image.Image
        The generated PNG image.
    """
    # Ensure SVG has proper size attributes
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)

class RewardModelWorker(Worker):
    """
    Note that we only implement the reward model that is subclass of AutoModelForTokenClassification.
    """

    def __init__(self, config, custom_reward_fn=None):
        super().__init__()
        import torch.distributed
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        self.config = config
        self.custom_reward_fn = custom_reward_fn

        # build device mesh for Ulysses Sequence Parallel
        world_size = torch.distributed.get_world_size()
        from torch.distributed.device_mesh import init_device_mesh

        fsdp_size = self.config.model.fsdp_config.get('fsdp_size', -1)
        self.device_mesh = create_device_mesh(world_size=world_size, fsdp_size=fsdp_size)

        self.ulysses_device_mesh = None
        self.ulysses_sequence_parallel_size = self.config.get('ulysses_sequence_parallel_size', 1)
        dp = world_size // self.ulysses_sequence_parallel_size
        if self.ulysses_sequence_parallel_size > 1:
            self.ulysses_device_mesh = init_device_mesh('cuda',
                                                        mesh_shape=(dp, self.ulysses_sequence_parallel_size),
                                                        mesh_dim_names=['dp', 'sp'])

        self.ulysses_sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        self.aesthetic_evaluator = AestheticEvaluator()


    def _build_model(self, config):
        # the following line is necessary
        from transformers import AutoModelForTokenClassification, AutoConfig
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, CPUOffload

        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.model.path)

        trust_remote_code = config.model.get('trust_remote_code', False)
        input_tokenizer_local_path = copy_local_path_from_hdfs(config.model.input_tokenizer)
        self.input_tokenizer = hf_tokenizer(input_tokenizer_local_path,
                                                trust_remote_code=trust_remote_code)

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # quantization_config = BitsAndBytesConfig(
            #     load_in_4bit=True,
            #     bnb_4bit_quant_type="nf4",
            #     bnb_4bit_use_double_quant=True,
            #     bnb_4bit_compute_dtype=torch.float16,
            # )
            self.processor = AutoProcessor.from_pretrained(local_path)
            reward_module = PaliGemmaForConditionalGeneration.from_pretrained(local_path,
                                                                            low_cpu_mem_usage=True,
                                                                            # quantization_config=quantization_config,
                                                                            torch_dtype=torch.bfloat16,
                                                                            # attn_implementation='flash_attention_2',
                                                                            trust_remote_code=trust_remote_code
                                                                            )
            reward_module.to(torch.bfloat16)
        auto_wrap_policy = get_fsdp_wrap_policy(module=reward_module, config=self.config.model.fsdp_config)

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        reward_module = FSDP(
            reward_module,
            param_init_fn=init_fn,
            use_orig_params=False,
            auto_wrap_policy=auto_wrap_policy,
            device_id=torch.cuda.current_device(),
            sharding_strategy=sharding_strategy,  # zero3
            sync_module_states=True,
            cpu_offload=CPUOffload(offload_params=True),
            forward_prefetch=False,
            device_mesh=self.device_mesh
        )

        return reward_module

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get('external_lib', None))
        self.reward_module = self._build_model(config=self.config)
        torch.cuda.empty_cache()

    

    def _expand_to_token_level(self, data: DataProto, scores: torch.Tensor):
        batch_size = data.batch.batch_size[0]
        # expand as token_level_reward
        attention_mask = data.batch['attention_mask']
        position_ids = data.batch['position_ids']
        response_length = data.batch['responses'].shape[-1]
        eos_mask_idx = torch.argmax(position_ids * attention_mask, dim=-1)  # (bsz,)
        token_level_scores = torch.zeros_like(attention_mask, dtype=scores.dtype)  # (bsz, seqlen)
        token_level_scores[torch.arange(batch_size), eos_mask_idx] = scores

        # select the response part
        token_level_scores = token_level_scores[:, -response_length:]

        return token_level_scores

    def _switch_chat_template(self, data: DataProto):
        images = []
        prompts = []
        aesthetic_scores = []
        for i in range(data.batch.batch_size[0]):
            # extract raw prompt

            # extract response
            response_ids = data.batch['responses'][i]
            response_length = response_ids.shape[-1]
            valid_response_length = data.batch['attention_mask'][i][-response_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            response = self.input_tokenizer.decode(valid_response_ids)
            # remove bos and eos
            response = response.replace(self.input_tokenizer.eos_token, '')

            from torch import tensor

            # print("label is: ",ground_truth)
            try:
                image = svg_to_png(response)
                aesthetic_scores.append(tensor([self.aesthetic_evaluator(image)]))

            except:
                import numpy as np
                random_array = np.random.randint(0, 256, (384, 384, 3), dtype=np.uint8)
                image = Image.fromarray(random_array, mode='RGB')
                aesthetic_scores.append(tensor([0.0]))

            images.append(image)
            prompts.append(data.non_tensor_batch['reward_model'][i]['ground_truth'])

            if self.rank == 0 and i == 0:
                # for debugging purpose
                print(f'Switch template. chat: {response}')

        inputs = self.processor(images=images, text=prompts, return_tensors='pt', padding=True, truncation=True, max_length=self.config.max_length)
        aesthetic_scores = torch.cat(aesthetic_scores, dim=0)
        # print('inputs_ids.shape=', inputs.input_ids.shape)
        # print('pixel_values.shape=', inputs.pixel_values.shape)
        rm_inputs = {'input_ids': inputs.input_ids, 'attention_mask': inputs.attention_mask, 'pixel_values': inputs.pixel_values, 'aesthetic_scores': aesthetic_scores}

        return rm_inputs


    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_rm_score(self, data: DataProto):
        import itertools
        from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
        data = data.to('cuda')
        rm_data = self._switch_chat_template(data)
        aesthetic_scores = rm_data.pop('aesthetic_scores')
        rm_data = DataProto.from_dict(rm_data)
        rm_data.batch = rm_data.batch.cuda()

        # perform forward computation
        with self.ulysses_sharding_manager:
            rm_data = self.ulysses_sharding_manager.preprocess_data(data=rm_data)
            data = self.ulysses_sharding_manager.preprocess_data(data=data)

            use_dynamic_bsz = self.config.use_dynamic_bsz
            if use_dynamic_bsz:
                max_token_len = self.config.forward_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                micro_batches, indices = rearrange_micro_batches(batch=rm_data.batch, max_token_len=max_token_len)
            else:
                micro_batches = rm_data.batch.split(self.config.micro_batch_size_per_gpu)
            output = []
            for micro_batch in micro_batches:
                rm_score = self._forward_micro_batch(micro_batch)
                output.append(torch.tensor([rm_score]))
                # print(rm_score)
            scores = torch.cat(output, dim=0)  # (batch_size)
            scores += aesthetic_scores
            if use_dynamic_bsz:
                indices = list(itertools.chain.from_iterable(indices))
                assert len(indices) == scores.size(0), f"{len(indices)} vs. {scores.size()}"
                revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
                scores = scores[revert_indices]

            token_level_scores = self._expand_to_token_level(data, scores)
            # Note that this is only the scores, may not be the final rewards used to train RL
            output = DataProto.from_dict(tensors={'rm_scores': token_level_scores})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        self.reward_module._handle.reshard(True)

        output = output.to('cpu')
        torch.cuda.empty_cache()
        return output
    
    def _forward_micro_batch(self, micro_batch):
        from flash_attn.bert_padding import pad_input, unpad_input, index_first_axis, rearrange
        from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad

        with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            attention_mask = micro_batch['attention_mask']
            pixel_values = micro_batch['pixel_values']
            batch_size, seqlen = input_ids.shape

            outputs = self.reward_module(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        pixel_values=pixel_values)
            logits = outputs.logits[:, -1, :]  # Logits for the last (predicted) token
            masked_logits = self.mask_yes_no(logits)
            probabilities = torch.softmax(masked_logits, dim=-1)

            yes_token_id = self.processor.tokenizer.convert_tokens_to_ids('yes')
            no_token_id = self.processor.tokenizer.convert_tokens_to_ids('no')
            yes_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' yes')
            no_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' no')

            prob_yes = probabilities[0, yes_token_id].item()
            prob_no = probabilities[0, no_token_id].item()
            prob_yes_space = probabilities[0, yes_with_space_token_id].item()
            prob_no_space = probabilities[0, no_with_space_token_id].item()

            total_yes_prob = prob_yes + prob_yes_space
            total_no_prob = prob_no + prob_no_space

            total_prob = total_yes_prob + total_no_prob
            rm_score = total_yes_prob / total_prob

            return rm_score
    
    def score(self, solution: str, submission: str) -> float:
        constraints = SVGConstraints()
        if not constraints.validate_svg(submission):
            return 0.0
        # Score
        vqa_evaluator = VQAEvaluator()
        aesthetic_evaluator = AestheticEvaluator()

        try:
            image = svg_to_png(submission)
            vqa_score = p_fidelity = self.get_yes_probability(image, solution)
            p_text = self.get_yes_probability(image, solution)
            return p_fidelity * (1 - p_text)
            aesthetic_score = aesthetic_evaluator.score(image)
            instance_score = harmonic_mean(vqa_score, aesthetic_score, beta=2.0)
            return float(instance_score)

        except:
            return 0.0
        
    def mask_yes_no(self, logits):
        """Masks logits for 'yes' or 'no'."""
        yes_token_id = self.processor.tokenizer.convert_tokens_to_ids('yes')
        no_token_id = self.processor.tokenizer.convert_tokens_to_ids('no')
        yes_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' yes')
        no_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' no')

        mask = torch.full_like(logits, float('-inf'))
        mask[:, yes_token_id] = logits[:, yes_token_id]
        mask[:, no_token_id] = logits[:, no_token_id]
        mask[:, yes_with_space_token_id] = logits[:, yes_with_space_token_id]
        mask[:, no_with_space_token_id] = logits[:, no_with_space_token_id]
        return mask

    def get_yes_probability(self, image, prompt) -> float:
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(
            'cuda:0'
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Logits for the last (predicted) token
            masked_logits = self.mask_yes_no(logits)
            probabilities = torch.softmax(masked_logits, dim=-1)

        yes_token_id = self.processor.tokenizer.convert_tokens_to_ids('yes')
        no_token_id = self.processor.tokenizer.convert_tokens_to_ids('no')
        yes_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' yes')
        no_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' no')

        prob_yes = probabilities[0, yes_token_id].item()
        prob_no = probabilities[0, no_token_id].item()
        prob_yes_space = probabilities[0, yes_with_space_token_id].item()
        prob_no_space = probabilities[0, no_with_space_token_id].item()

        total_yes_prob = prob_yes + prob_yes_space
        total_no_prob = prob_no + prob_no_space

        total_prob = total_yes_prob + total_no_prob
        renormalized_yes_prob = total_yes_prob / total_prob

        return renormalized_yes_prob

    
    




@dataclass(frozen=True)
class SVGConstraints:
    """Defines constraints for validating SVG documents.

    Attributes
    ----------
    max_svg_size : int, default=10000
        Maximum allowed size of an SVG file in bytes.
    allowed_elements : dict[str, set[str]]
        Mapping of the allowed elements to the allowed attributes of each element.
    """

    max_svg_size: int = 10000
    allowed_elements: dict[str, set[str]] = field(
        default_factory=lambda: {
            'common': {
                'id',
                'clip-path',
                'clip-rule',
                'color',
                'color-interpolation',
                'color-interpolation-filters',
                'color-rendering',
                'display',
                'fill',
                'fill-opacity',
                'fill-rule',
                'filter',
                'flood-color',
                'flood-opacity',
                'lighting-color',
                'marker-end',
                'marker-mid',
                'marker-start',
                'mask',
                'opacity',
                'paint-order',
                'stop-color',
                'stop-opacity',
                'stroke',
                'stroke-dasharray',
                'stroke-dashoffset',
                'stroke-linecap',
                'stroke-linejoin',
                'stroke-miterlimit',
                'stroke-opacity',
                'stroke-width',
                'transform',
            },
            'svg': {
                'width',
                'height',
                'viewBox',
                'preserveAspectRatio',
            },
            'g': {'viewBox'},
            'defs': set(),
            'symbol': {'viewBox', 'x', 'y', 'width', 'height'},
            'use': {'x', 'y', 'width', 'height', 'href'},
            'marker': {
                'viewBox',
                'preserveAspectRatio',
                'refX',
                'refY',
                'markerUnits',
                'markerWidth',
                'markerHeight',
                'orient',
            },
            'pattern': {
                'viewBox',
                'preserveAspectRatio',
                'x',
                'y',
                'width',
                'height',
                'patternUnits',
                'patternContentUnits',
                'patternTransform',
                'href',
            },
            'linearGradient': {
                'x1',
                'x2',
                'y1',
                'y2',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'radialGradient': {
                'cx',
                'cy',
                'r',
                'fx',
                'fy',
                'fr',
                'gradientUnits',
                'gradientTransform',
                'spreadMethod',
                'href',
            },
            'stop': {'offset'},
            'filter': {
                'x',
                'y',
                'width',
                'height',
                'filterUnits',
                'primitiveUnits',
            },
            'feBlend': {'result', 'in', 'in2', 'mode'},
            'feFlood': {'result'},
            'feOffset': {'result', 'in', 'dx', 'dy'},
            'path': {'d'},
            'rect': {'x', 'y', 'width', 'height', 'rx', 'ry'},
            'circle': {'cx', 'cy', 'r'},
            'ellipse': {'cx', 'cy', 'rx', 'ry'},
            'line': {'x1', 'y1', 'x2', 'y2'},
            'polyline': {'points'},
            'polygon': {'points'},
        }
    )

    def validate_svg(self, svg_code: str) -> None:
        """Validates an SVG string against a set of predefined constraints.

        Parameters
        ----------
        svg_code : str
            The SVG string to validate.

        Raises
        ------
        ValueError
            If the SVG violates any of the defined constraints.
        """
        # Check file size
        if len(svg_code.encode('utf-8')) > self.max_svg_size:
            return False

        # Parse XML
        tree = etree.fromstring(
            svg_code.encode('utf-8'),
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )

        elements = set(self.allowed_elements.keys())

        # Check elements and attributes
        for element in tree.iter():
            # Check for disallowed elements
            tag_name = element.tag.split('}')[-1]
            if tag_name not in elements:
                return False

            # Check attributes
            for attr, attr_value in element.attrib.items():
                # Check for disallowed attributes
                attr_name = attr.split('}')[-1]
                if (
                    attr_name not in self.allowed_elements[tag_name]
                    and attr_name not in self.allowed_elements['common']
                ):
                    return False

                # Check for embedded data
                if 'data:' in attr_value.lower():
                    return False
                if ';base64' in attr_value:
                    return False

                # Check that href attributes are internal references
                if attr_name == 'href':
                    if not attr_value.startswith('#'):
                        return False
        return True



class VQAEvaluator:
    """Evaluates images based on their similarity to a given text description."""

    def __init__(self):
        self.quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        self.model_path = 'google/paligemma-2/transformers/paligemma2-10b-mix-448'
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.model = PaliGemmaForConditionalGeneration.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            quantization_config=self.quantization_config,
        )
        self.questions = {
            'fidelity': 'Does <image> portray "{}" without any lettering? Answer yes or no.',
            'text': '<image> Text present: yes or no?',
        }

    def score(self, image: Image.Image, description: str) -> float:
        """Evaluates the fidelity of an image to a target description using VQA yes/no probabilities.

        Parameters
        ----------
        image : PIL.Image.Image
            The image to evaluate.
        description : str
            The text description that the image should represent.

        Returns
        -------
        float
            The score (a value between 0 and 1) representing the match between the image and its description.
        """
        p_fidelity = self.get_yes_probability(image, self.questions['fidelity'].format(description))
        p_text = self.get_yes_probability(image, self.questions['text'])
        return p_fidelity * (1 - p_text)

    def mask_yes_no(self, logits):
        """Masks logits for 'yes' or 'no'."""
        yes_token_id = self.processor.tokenizer.convert_tokens_to_ids('yes')
        no_token_id = self.processor.tokenizer.convert_tokens_to_ids('no')
        yes_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' yes')
        no_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' no')

        mask = torch.full_like(logits, float('-inf'))
        mask[:, yes_token_id] = logits[:, yes_token_id]
        mask[:, no_token_id] = logits[:, no_token_id]
        mask[:, yes_with_space_token_id] = logits[:, yes_with_space_token_id]
        mask[:, no_with_space_token_id] = logits[:, no_with_space_token_id]
        return mask

    def get_yes_probability(self, image, prompt) -> float:
        inputs = self.processor(images=image, text=prompt, return_tensors='pt').to(
            'cuda:0'
        )

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]  # Logits for the last (predicted) token
            masked_logits = self.mask_yes_no(logits)
            probabilities = torch.softmax(masked_logits, dim=-1)

        yes_token_id = self.processor.tokenizer.convert_tokens_to_ids('yes')
        no_token_id = self.processor.tokenizer.convert_tokens_to_ids('no')
        yes_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' yes')
        no_with_space_token_id = self.processor.tokenizer.convert_tokens_to_ids(' no')

        prob_yes = probabilities[0, yes_token_id].item()
        prob_no = probabilities[0, no_token_id].item()
        prob_yes_space = probabilities[0, yes_with_space_token_id].item()
        prob_no_space = probabilities[0, no_with_space_token_id].item()

        total_yes_prob = prob_yes + prob_yes_space
        total_no_prob = prob_no + prob_no_space

        total_prob = total_yes_prob + total_no_prob
        renormalized_yes_prob = total_yes_prob / total_prob

        return renormalized_yes_prob


class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticEvaluator:
    def __init__(self):
        self.model_path = '/home/wxf/lzq/data/model/sac+logos+ava1-l14-linearMSE.pth'
        self.clip_model_path = 'ViT-L/14'
        self.predictor, self.clip_model, self.preprocessor = self.load()

    def load(self):
        """Loads the aesthetic predictor model and CLIP model."""
        state_dict = torch.load(self.model_path, weights_only=True, map_location='cuda:0')

        # CLIP embedding dim is 768 for CLIP ViT L 14
        predictor = AestheticPredictor(768)
        predictor.load_state_dict(state_dict)
        predictor.to('cuda:0')
        predictor.eval()
        clip_model, preprocessor = clip.clip.load(self.clip_model_path, device='cuda:0')

        return predictor, clip_model, preprocessor


    def score(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Predicts the CLIP aesthetic score of one or multiple images.
        
        Args:
            images: A single PIL Image or a list of PIL Images
            
        Returns:
            torch.Tensor: A tensor of scores (scaled to [0, 1]) for all input images
        """
        # Convert single image to list if needed
        if isinstance(images, Image.Image):
            images = [images]
        
        # Process each image and collect features
        all_features = []
        for img in images:
            # Preprocess and move to device
            img_tensor = self.preprocessor(img).unsqueeze(0).to('cuda:0')
            
            with torch.no_grad():
                # Encode image
                image_features = self.clip_model.encode_image(img_tensor)
                # L2 normalize
                image_features /= image_features.norm(dim=-1, keepdim=True)
                all_features.append(image_features.cpu())
        
        # Stack all features
        if len(all_features) > 1:
            stacked_features = torch.cat(all_features, dim=0)
        else:
            stacked_features = all_features[0]
        
        # Move to device and predict scores
        scores = self.predictor(stacked_features.float())
        
        # Scale to [0, 1] and return as tensor
        return scores / 10.0


