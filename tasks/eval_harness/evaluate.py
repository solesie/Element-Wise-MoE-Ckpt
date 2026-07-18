# This code is originally from https://github.com/bigscience-workshop/Megatron-DeepSpeed
# under the license https://huggingface.co/spaces/bigscience/license

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),
                                             os.path.pardir,os.path.pardir)))

from lm_eval import evaluator, utils
from lm_eval.base import CacheHook, LM
from tqdm import tqdm
import torch.nn.functional as F

from pretrain_gpt import model_provider
import numpy as np
import time

import torch
from megatron import get_args
from megatron import get_tokenizer
from megatron.core.enums import ModelType
from megatron.core import mpu
from megatron.training import setup_model_and_optimizer, get_model
from megatron.core.tensor_parallel.mappings import gather_from_tensor_model_parallel_region

from megatron.utils import get_ltor_masks_and_position_ids, unwrap_model
from megatron.p2p_communication import recv_forward, send_forward
import json

from tasks.eval_harness.paper_tasks import (
    TASK_NAMES,
    get_paper_task_dict,
    protocol_metadata,
)

from torch.nn.parallel.distributed import DistributedDataParallel as torchDDP
from megatron.model.distributed import DistributedDataParallel as LocalDDP
from megatron.model.module import Float16Module
from deepspeed.accelerator import get_accelerator

class EvalHarnessAdaptor(LM):
    def __init__(self, model, tokenizer):
        args = get_args()
        self.args = args
        self.model = model
        self.tokenizer = tokenizer
        self.EOT_TOKEN_ID = tokenizer.eod

        self._max_length = args.seq_length

        # For ds we split into mini batches and then micro batches to keep pipelining api happy.
        # With Megatron we just go to micro_batches directly
        self._batch_size = args.micro_batch_size

        self.cache_hook = CacheHook(None)
        self.is_main = args.rank == 0
        self._device = get_accelerator().current_device_name()
        self.is_pipe_parallel = mpu.get_pipeline_model_parallel_world_size() > 1
        self.is_data_parallel = mpu.get_data_parallel_world_size() > 1
        self.adaptive_seq_len = args.adaptive_seq_len
        self.max_gen_toks = args.generation_max_tokens
        if self.is_data_parallel and args.moe_expert_parallel_size == 1: # For MoE model, allow a "fake data parallel" in order to partition model into multiple gpus
            raise NotImplementedError("Data parallelism is currently not supported for evaluation")

        self.is_last_stage = True if not self.is_pipe_parallel else mpu.is_pipeline_last_stage()  # only the last stage of the pipeline model will receive the logits

    @property
    def max_length(self):
        return self._max_length

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def device(self):
        return self._device


    def loglikelihood(self, requests):
        new_reqs = []
        for context, continuation in requests:
            if context == "":
                # end of text as context
                context_enc = [self.EOT_TOKEN_ID]
            else:
                context_enc = self.tokenizer_encode(context)

            continuation_enc = self.tokenizer_encode(continuation)

            new_reqs.append(((context, continuation), context_enc, continuation_enc))

        return self._loglikelihood_tokens(new_reqs)

    def loglikelihood_rolling(self, requests):
        # TODO: Implement caching once we've confirmed the perplexity implementation
        # TODO: automatic batch size detection for vectorization

        loglikelihoods = []
        with torch.no_grad():
            for string, in tqdm(requests):
                rolling_token_windows = list(map(utils.make_disjoint_window, utils.get_rolling_token_windows(
                    token_list=self.tokenizer_encode(string),
                    prefix_token=self.EOT_TOKEN_ID,
                    max_seq_len=self.max_length,
                    context_len=1,
                )))

                rolling_token_windows = [(None,) + x for x in rolling_token_windows]

                # TODO: extract out this call so it only gets called once and also somehow figure out partial caching for that
                string_nll = self._loglikelihood_tokens(rolling_token_windows, disable_tqdm=True)

                # discard is_greedy
                string_nll = [x[0] for x in string_nll]

                string_nll = sum(string_nll)
                loglikelihoods.append(string_nll)

        return loglikelihoods

    def _loglikelihood_tokens(self, requests, disable_tqdm=False):
        disable_tqdm = disable_tqdm if self.is_main else True
        res = []
        self.model.eval()
        with torch.no_grad():
            def _collate(x):
                toks = x[1] + x[2]
                return (-len(toks), tuple(toks))

            reord = utils.Reorderer(requests, _collate)
            for chunk in utils.chunks(tqdm(reord.get_reordered(), disable=disable_tqdm), self.batch_size):
                inps, contlens, inplens, padding_length = [], [], [], None
                for _, context_enc, continuation_enc in chunk:
                    # when too long to fit in context, truncate from the left
                    inp = torch.tensor(
                        (context_enc + continuation_enc)[-(self.max_length + 1):][:-1]
                        , dtype=torch.long).to(self.device)
                    inplen, = inp.shape

                    cont = continuation_enc

                    # since in _collate we make sure length is descending, the longest is always the first one.
                    padding_length = padding_length if padding_length is not None else inplen
                    if not self.adaptive_seq_len:
                        padding_length = self.max_length
                    # pad to length
                    inp = torch.cat([
                        inp,  # [seq]
                        torch.zeros(padding_length - inplen, dtype=torch.long).to(inp.device)  # [padding_length - seq]
                    ], dim=0)

                    inps.append(inp.unsqueeze(0))

                    contlens.append(cont)
                    inplens.append(inplen)

                logits = self._model_call(torch.cat(inps, dim=0))
                if logits is not None:
                    multi_logits = F.log_softmax(logits, dim=-1).cpu()  # [batch, seq, vocab]

                    for (cache_key, _, _), logits, inp, inplen, cont_toks in zip(chunk, multi_logits, inps, inplens, contlens):
                        contlen = len(cont_toks)
                        logits = logits[inplen - contlen:inplen].unsqueeze(0)  # [1, seq, vocab]
                        greedy_tokens = logits.argmax(dim=-1)
                        # cont_toks :: [1, seq]
                        cont_toks = torch.tensor(cont_toks, dtype=torch.long).unsqueeze(0)
                        max_equal = (greedy_tokens == cont_toks).all()
                        # last_token_slice = logits[:, -1, :].squeeze(0).tolist()

                        logits = torch.gather(logits, 2, cont_toks.unsqueeze(-1)).squeeze(-1)  # [1, seq]
                        answer = (float(logits.sum()), bool(max_equal))
                        # partial caching
                        if cache_key is not None:
                            self.cache_hook.add_partial("loglikelihood", cache_key, answer)
                        res.append(answer)

        if not mpu.is_pipeline_last_stage():
            # @HACK: To make the eval harness happy on threads that don't have access to the results.
            #        We just randomly generate some data.
            res = [(np.random.rand(), np.random.rand()>0.5) for _ in requests]

        return reord.get_original(res)

    def create_model_inputs(self, tokens):
        args = get_args()

        attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
            tokens,
            self.EOT_TOKEN_ID,
            args.reset_position_ids,
            args.reset_attention_mask,
            args.eod_mask_loss)

        return (tokens, position_ids, attention_mask), (tokens, loss_mask)

    def _model_call(self, inps):
        args = get_args()

        if args.deepspeed:
            if args.no_pipeline_parallel:
                # self.model.set_batch_fn(self.create_model_inputs)
                # round up to multiple of micro_batch_size
                new_size = ((len(inps) + args.micro_batch_size-1)  // args.micro_batch_size) * args.micro_batch_size
                padded = F.pad(inps, (0, 0, 0, new_size-len(inps)), value = 0)
                # dummy data iterator for pipelining.
                data_iterator = list((torch.stack(inp) for inp in utils.chunks(padded, args.micro_batch_size)))
                self.model.micro_batches = len(data_iterator)
                # output = self.model.eval_batch(iter(data_iterator), compute_loss = False, reduce_output = None)
                output = []
                for tokens in data_iterator:
                    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
                                                                tokens,
                                                                self.EOT_TOKEN_ID,
                                                                args.reset_position_ids,
                                                                args.reset_attention_mask,
                                                                args.eod_mask_loss)
                    a_output, *other_losses = self.model(tokens,
                        position_ids,
                        attention_mask,
                        tokentype_ids=None)
                    output.append(a_output)

                if output is not None:
                    output = torch.cat(output, 0)[:len(inps)]
                else:
                    output = None

                # hack #2 for adaptive_seq_len to work as total_loss gets appended to and shapes aren't the same
                if args.adaptive_seq_len:
                    self.model.total_loss = None
            else:
                self.model.set_batch_fn(self.create_model_inputs)
                # round up to multiple of micro_batch_size
                new_size = ((len(inps) + args.micro_batch_size-1)  // args.micro_batch_size) * args.micro_batch_size
                padded = F.pad(inps, (0, 0, 0, new_size-len(inps)), value = 0)
                # dummy data iterator for pipelining.
                data_iterator = list((torch.stack(inp) for inp in utils.chunks(padded, args.micro_batch_size)))
                self.model.micro_batches = len(data_iterator)
                if args.adaptive_seq_len:
                    # Each evaluation chunk may have a different sequence
                    # width, so discard DeepSpeed's cached receive buffers
                    # before the next activation metadata exchange.
                    self.model.reset_activation_shape()
                output = self.model.eval_batch(iter(data_iterator), compute_loss = False, reduce_output = None)


                if output is not None:
                    output = concat_pipeline_logits(output, len(inps))
                else:
                    output = None

                # hack #2 for adaptive_seq_len to work as total_loss gets appended to and shapes aren't the same
                if args.adaptive_seq_len:
                    self.model.total_loss = None
        else:
            # Since the shape of the micro-batch will change
            # We need set the correct shapes here
            # So that latter pipeline stages knows which shapes to expect.
            # Otherwise we will deadlock.

            args.micro_batch_size = len(inps)
            args.seq_length = len(inps[0])
            args.max_position_embeddings = args.seq_length

            input_tensor = recv_forward()

            # Forward pass through the model.
            unwrapped_model = unwrap_model(self.model, (torchDDP, LocalDDP, Float16Module))
            unwrapped_model.set_input_tensor(input_tensor)
            output = self.model(*self.create_model_inputs(inps)[0])
            send_forward(output)

        if mpu.is_pipeline_last_stage():
            return gather_from_tensor_model_parallel_region(output)[..., :self.tokenizer.vocab_size]
        else:
            return None

    def tokenizer_encode(self, text):
        """Tokenize text *without* adding special tokens."""
        # Splitting this into its own method in case we need to handle special cases for different tokenizers
        from megatron.tokenizer.gpt2_tokenization import GPT2Tokenizer
        if isinstance(self.tokenizer.tokenizer, GPT2Tokenizer):
            return self.tokenizer.tokenizer.encode(text)
        else:
            return self.tokenizer.tokenizer.encode(text, add_special_tokens=False)

    def tokenizer_decode(self, tokens):
        return self.tokenizer.tokenizer.decode(tokens)

    def greedy_until(self, requests):
        """Greedy generation for explicitly requested generation tasks.

        Every pipeline rank advances the same token sequence.  The last stage
        chooses the token and broadcasts it within its pipeline group before
        the next model call.
        """
        results = []
        self.model.eval()
        with torch.no_grad():
            for chunk in utils.chunks(requests, self.batch_size):
                contexts = []
                stops = []
                for context, until in chunk:
                    encoded = self.tokenizer_encode(context)
                    contexts.append(encoded[-self.max_length :])
                    stops.append([until] if isinstance(until, str) else list(until))

                generated = [[] for _ in chunk]
                active = [True] * len(chunk)
                for _ in range(self.max_gen_toks):
                    model_tokens = [
                        (context + continuation)[-self.max_length :]
                        for context, continuation in zip(contexts, generated)
                    ]
                    lengths = [len(tokens) for tokens in model_tokens]
                    width = max(lengths)
                    inputs = torch.full(
                        (len(chunk), width),
                        self.EOT_TOKEN_ID,
                        dtype=torch.long,
                        device=self.device,
                    )
                    for row, tokens in enumerate(model_tokens):
                        inputs[row, : len(tokens)] = torch.tensor(
                            tokens, dtype=torch.long, device=self.device
                        )

                    logits = self._model_call(inputs)
                    next_tokens = torch.zeros(
                        len(chunk), dtype=torch.long, device=self.device
                    )
                    if self.is_last_stage:
                        rows = torch.arange(len(chunk), device=self.device)
                        positions = torch.tensor(lengths, device=self.device) - 1
                        next_tokens.copy_(logits[rows, positions].argmax(dim=-1))
                    if self.is_pipe_parallel:
                        torch.distributed.broadcast(
                            next_tokens,
                            src=mpu.get_pipeline_model_parallel_last_rank(),
                            group=mpu.get_pipeline_model_parallel_group(),
                        )

                    token_list = next_tokens.tolist()
                    for row, token in enumerate(token_list):
                        if not active[row]:
                            continue
                        if token == self.EOT_TOKEN_ID:
                            active[row] = False
                            continue
                        generated[row].append(token)
                        decoded = self.tokenizer_decode(generated[row])
                        if any(stop and stop in decoded for stop in stops[row]):
                            active[row] = False
                    if not any(active):
                        break

                for continuation, until in zip(generated, stops):
                    decoded = self.tokenizer_decode(continuation)
                    stop_positions = [decoded.find(stop) for stop in until if stop]
                    stop_positions = [position for position in stop_positions if position >= 0]
                    if stop_positions:
                        decoded = decoded[: min(stop_positions)]
                    results.append(decoded)
        return results


from megatron.initialize import initialize_megatron
import megatron

from tools.convert_checkpoint.deepspeed_checkpoint import DeepSpeedCheckpoint
try:
    from .checkpoint_utils import (
        concat_pipeline_logits,
        deepspeed_model_only_eval_config,
        load_deepspeed_checkpoint_args,
        prepare_bf16_eval_module,
        resolve_deepspeed_checkpoint_dir,
    )
except ImportError:  # evaluate.py is normally launched as a script.
    from checkpoint_utils import (
        concat_pipeline_logits,
        deepspeed_model_only_eval_config,
        load_deepspeed_checkpoint_args,
        prepare_bf16_eval_module,
        resolve_deepspeed_checkpoint_dir,
    )

def override_args(args, override_args, skip_keys, skip_if_specified_keys):
    for k, v in vars(override_args).items():
        if k in skip_keys:
            continue
        if k in skip_if_specified_keys and getattr(args, k) is not None:
            continue
        setattr(args, k, v)


# Note(Hesslow):
# The model loading is a bit convoluted.
# We want to parse out the model arguments from the checkpoint and use those to initialize megatron-ds.
#
# However megatron-ds expects its arguments on the command line.
# And at that point we don't know them.
#
# Instead we use Jasons way: we load the arguments form the checkpoint and then override _parse_args to return whatever args we want.
#
# If the checkpoint is old, some new arguments may have been introduced and the code will expect these arguments to exist.
# In order to support this we _first_ parse the arguments normally, and then override them with the arguments from the checkpoint.
# Keeping the default-value of newer arguments.
#
# We then use the megatron deepspeed converter to load the deepspeed checkpoints as if they we're megatron checkpoints.
def load_ds_checkpoint_and_setup_megatron(extra_args_provider):
    # parse the megatorn args. But wait with initalizing megatron.
    # avoid printing the arguments, since they will later be overridden.
    _print_args = megatron.arguments._print_args
    megatron.arguments._print_args = lambda *_args, **kwarg: None
    args = parse_args(extra_args_provider=extra_args_provider)

    if args.deepspeed:
        world = int(args.world_size)
        tp = int(args.tensor_model_parallel_size)
        pp = int(args.pipeline_model_parallel_size)
        ep = int(args.moe_expert_parallel_size)
        parallel_product = tp * pp * ep
        if parallel_product <= 0 or world % parallel_product:
            raise RuntimeError(
                f"invalid evaluator topology world={world}, tp={tp}, pp={pp}, ep={ep}"
            )
        cp_path = resolve_deepspeed_checkpoint_dir(
            args.load,
            require_model_only=True,
            expected_topology={
                "world_size": world,
                "tensor_parallel_size": tp,
                "pipeline_parallel_size": pp,
                "expert_parallel_size": ep,
                "data_parallel_size": world // parallel_product,
            },
        )
        cp_args = load_deepspeed_checkpoint_args(cp_path)
        bundled_vocab = os.path.join(cp_path, 'tokenizer', 'gpt2-vocab.json')
        bundled_merges = os.path.join(cp_path, 'tokenizer', 'gpt2-merges.txt')
        bundled_ds_config = os.path.join(cp_path, 'deepspeed_config.json')
        if os.path.isfile(bundled_vocab):
            cp_args.vocab_file = bundled_vocab
        if os.path.isfile(bundled_merges):
            cp_args.merge_file = bundled_merges
        if not args.masked_softmax_fusion:
            cp_args.masked_softmax_fusion = False
        if not isinstance(getattr(cp_args, 'deepspeed_config_dict', None), dict):
            if not os.path.isfile(bundled_ds_config):
                raise RuntimeError('checkpoint has no embedded DeepSpeed config')
            with open(bundled_ds_config, 'r', encoding='utf-8') as config_file:
                cp_args.deepspeed_config_dict = json.load(config_file)
        ds_checkpoint = None
    else:
        cp_path = args.load
        ds_checkpoint = DeepSpeedCheckpoint(args.load,
                                            tp_degree=args.tensor_model_parallel_size,
                                            pp_degree=args.pipeline_model_parallel_size,
                                            no_pp=args.no_pipeline_parallel)
        cp_args = ds_checkpoint.get_args()
    # Merge the current args with the checkpoint args.
    skip_keys = ['world_size', 'rank', 'local_rank','device_count', 'micro_batch_size','global_batch_size', 'batch_size', 'tensorboard_dir', 'deepspeed', 'deepspeed_config',
                     'data_parallel_size', 'pipeline_model_parallel_size', 'tensor_model_parallel_size', 'moe_expert_parallel_size', 'moe_token_dropping', 'load', 'rampup_batch_size', 'iteration', 'inference', 'random_ltd']

    skip_if_specified = ['merge_file', 'vocab_file']

    if args.eval_fp32:
        cp_args.fp16 = False
        cp_args.bf16 = False
        cp_args.params_dtype = torch.float32

    cp_args.tokenizer_type = 'GPT2BPETokenizer'

    override_args(args, cp_args, skip_keys, skip_if_specified)
    # setup_model_and_optimizer passes the embedded dict to DeepSpeed. Recent
    # DeepSpeed versions reject receiving both this dict and a config path.
    if args.deepspeed:
        args.deepspeed_config = None

    # stop megatron from reparsing the arguments.
    megatron.arguments.parse_args = lambda *_args, **kwarg: args
    megatron.global_vars._ensure_var_is_not_initialized = lambda *_args, **kwarg: None
    megatron.global_vars._GLOBAL_ARGS = args

    initialize_megatron(extra_args_provider=extra_args_provider)
    megatron.global_vars._GLOBAL_ARGS = args
    torch.distributed.barrier()

    # Initializing megatron will update eg. tokenizer size. Override again.
    override_args(args, cp_args, skip_keys, skip_if_specified)
    if args.deepspeed:
        # Keep Megatron's BF16 parameter construction from the checkpoint,
        # but do not instantiate the training ZeRO/BF16 optimizer wrappers in
        # a model-only evaluator.  Also collapse the training GAS to one eval
        # batch so PipelineEngine does not replay the 10K training schedule.
        args.deepspeed_config_dict = deepspeed_model_only_eval_config(
            args.deepspeed_config_dict,
            micro_batch_size=args.micro_batch_size,
            data_parallel_size=mpu.get_data_parallel_world_size(),
        )
        args.zero_stage = 0

    # print final arguments.
    _print_args("eval_harness arguments", args)
    if args.deepspeed:

        # Hack #3:
        # Loading pipelined models in deepspeed with different TP than it was originally trained on fails
        # due to a sanity check, that makes sure that all state_dicts that we merge contains attention layers.
        # This, however, is not true for pipelining when we will merge the state_dict for the embeddings which
        # which does not contain these attention-specific keys.
        #
        # Deepspeed does however manage to load the model if we just turn off this sanity check.
        import deepspeed
        deepspeed.runtime.state_dict_factory.MegatronSDLoader.sanity_check = lambda self, ckpt_file_name: None


        args.load = None
        model, _, _ = setup_model_and_optimizer(model_provider, ModelType.encoder_or_decoder)
        model = model[0]
        if not args.eval_fp32:
            # Disabling DeepSpeed's BF16 optimizer wrapper also disables its
            # automatic module cast.  Apply that cast explicitly so plain
            # torch modules such as positional embeddings cannot remain FP32.
            prepare_bf16_eval_module(model.module)
        zero_enabled = model._config.zero_enabled
        model._config.zero_enabled = False
        try:
            load_path, _ = model.load_checkpoint(
                cp_path,
                tag='.',
                load_optimizer_states=False,
                load_lr_scheduler_states=False,
                load_module_only=True,
            )
        finally:
            model._config.zero_enabled = zero_enabled
        if load_path is None:
            raise RuntimeError(f'DeepSpeed failed to load model-only checkpoint: {cp_path}')
    else:
        # This converter has a script-style import and is not needed by the
        # normal DeepSpeed path.  Keep it lazy so DeepSpeed evaluation does not
        # depend on tools/convert_checkpoint being on sys.path.
        from tools.convert_checkpoint.deepspeed_to_megatron import _create_rank_checkpoint

        model = get_model(model_provider)[0]
        # Initialize megatron model using the parsed state dict.
        sd = _create_rank_checkpoint(ds_checkpoint, None, mpu.get_tensor_model_parallel_rank(), mpu.get_pipeline_model_parallel_rank(), True)

        model.load_state_dict(sd['model'], strict=True)

    if args.eval_fp32:
        model = model.float()

    torch.distributed.barrier()
    return model

def tasks_args(parser):
    """Provide extra arguments required for tasks."""
    group = parser.add_argument_group(title='Evaluation options')
    group.add_argument('--task_list', type=str, default=','.join(TASK_NAMES),
                       help='Comma-separated subset of the fixed paper tasks.')
    group.add_argument('--results_path', type=str, default = "./results.json", help='Path to where the results will be stored.')
    group.add_argument('--paper-data-dir', type=str,
                       default='/home/paul/data/lm_eval_paper_6tasks',
                       help='Prepared paper-task data root containing manifest.json.')
    group.add_argument('--adaptive_seq_len',  default = False, action='store_true',
                       help='Should the sequence length be adapted to the batch during evaluation, if in fp16 the results will be slightly different due to numerical errors but greatly speed up evaluation.')
    group.add_argument('--num_fewshot', type=int, default = 0, help='Number of few-shot prompts.')
    group.add_argument('--eval_fp32',  default = False, action='store_true', help='Should the evaluation run in fp32')
    group.add_argument('--limit', type=int, default=None,
                       help='Evaluation-example limit for plumbing smoke tests only.')
    group.add_argument('--generation-max-tokens', type=int, default=64,
                       help='Maximum greedy continuation length for generation tasks.')
    return parser

from megatron.arguments import parse_args

def main():
    start = time.time()
    model = load_ds_checkpoint_and_setup_megatron(extra_args_provider=tasks_args)

    args = get_args()
    if args.num_fewshot != 0:
        raise ValueError('the paper protocol is fixed to zero-shot (--num_fewshot 0)')
    os.environ['LM_EVAL_PAPER_DATA'] = args.paper_data_dir
    if args.deepspeed and args.adaptive_seq_len:
        # adaptive_seq_len hack #1:
        # CL automatically enables reset_activation_shape() which allows us to change input shapes
        # and it also reshapes the attenion scores in attention_mask_func
        args.curriculum_learning_legacy = 1

    task_list = list(TASK_NAMES) if args.task_list == 'all' else args.task_list.split(',')
    task_dict = get_paper_task_dict(task_list)

    model.module.activation_checkpoint_interval = 0
    model._compute_loss = False
    # Standalone scoring consumes logits only. DeepSpeed 0.19 still polls
    # auxiliary losses with compute_loss=False, before GPTModelPipe has values.
    model.module.get_additional_losses = lambda: None
    model.fwd_outputs = []

    tokenizer = get_tokenizer()
    adaptor = EvalHarnessAdaptor(model, tokenizer)
    results = evaluator.evaluate(
        adaptor,
        task_dict,
        False,
        args.num_fewshot,
        args.limit,
    )
    results['paper_protocol'] = protocol_metadata()

    candidate = args.rank if (
        mpu.is_pipeline_last_stage() and mpu.get_tensor_model_parallel_rank() == 0
    ) else args.world_size
    writer = torch.tensor(candidate, dtype=torch.int64, device=adaptor.device)
    torch.distributed.all_reduce(writer, op=torch.distributed.ReduceOp.MIN)
    writer_rank = int(writer.item())
    if args.rank == writer_rank:
        print(json.dumps(results, indent=2))
        os.makedirs(os.path.dirname(os.path.abspath(args.results_path)), exist_ok=True)
        with open(args.results_path, 'w') as outfile:
            json.dump(results, outfile, indent = 4)
    end = time.time()
    if args.rank == writer_rank:
        print("evaluation of {} ends in {:.2f} sec, or {:.2f} min, or {:.2f} hr".format(args.task_list, end-start, (end-start)/60.0, (end-start)/3600.0))

if __name__ == '__main__':
    main()
