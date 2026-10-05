# Lab brief: `notebooks/07-finetuning-lora.ipynb`

From the Academic Director to the Neural Lab Engineer. Lecture: `lectures/07-finetuning-lora.qmd` (same symbols and equation names). Lab standards: `PLAN.md` section 5. This file is not rendered by Quarto.

**Objectives exercised** (from `_variables.yml`, `m07`): turn a pretrained causal LM into an instruction follower; apply LoRA and explain why it works; choose decoding settings and evaluate generated text.

**Every time, size and threshold below is an estimate or a target, not a measurement,** unless marked *checked*. Replace each with the value you measure on a free T4, and tell me if a target cannot be met. *Checked* means I ran it on 2026-10-04 on macOS (CPU, `transformers` 5.18.0, `peft` 0.21.2, `torch` 2.14.1), not on Colab.

## Model (proposed pin; goes in `_variables.yml`, not in the notebook)

**`HuggingFaceTB/SmolLM2-135M`**, the *base* model, revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`.

- *Checked:* Apache-2.0 (model card and Hub API), not gated, 134,515,008 parameters, `LlamaForCausalLM`, 30 blocks, hidden size 576, `q_proj` 576→576, `v_proj` 576→192, no biases.
- *Checked:* the base tokenizer has **no chat template** and no pad token, but `<|im_start|>` (id 1) and `<|im_end|>` (id 2) are already in the vocabulary. So the lab sets a ChatML template itself and never resizes the embeddings.
- *Checked:* before tuning, greedy output for "Name the capital of France." repeats the request and runs to the length limit. That is the "before" the lab needs.
- Why not the `-Instruct` variant: it already follows instructions, so there is no before and after. Why not GPT-2 (MIT): its attention uses one fused `Conv1D` module (`c_attn`), which muddles "target the query and value projections", and it has no ChatML tokens. If quality at 135M is too poor to read, move to `SmolLM2-360M` (Apache-2.0, 361,821,120 parameters, *checked*) and re-measure the budget.
- **Load with `dtype=torch.float32`.** *Checked:* the default load is bfloat16, in which the merge check is off by 1.8 in the logits and a CPU step is about 20 times slower. On the T4, use float32 weights with autocast if you want speed.

## Data (provided)

Dolly 15k through the `data/README.md` loading contract (pinned URL and SHA-256). You fix the subset: suggested filter is prompt at most 128 tokens and response at most 64 tokens after formatting; then seed 0, 2,000 train, 200 held-out, and 5 fixed display prompts drawn from the held-out set. User content is `instruction`, plus a blank line and `context` when it is not empty. Commit the subset under `data/` if it is under 1 MB, with the CC BY-SA 3.0 attribution, and record it in `_variables.yml` and `data/README.md`.

## Core path (50 minutes)

| # | Participant writes | Checkpoint | Metric tested | Min |
|---|---|---|---|---|
| 0 | Nothing: generate from the base model on the 5 prompts | none | none | 3 |
| 1 | `LoRALinear.__init__` and `forward` on one `nn.Linear` (eqs. `lora`, `lora-forward`) | Output **equals** the base layer at init (B = 0); trainable count `== r * (d_in + d_out)`; base weight has `requires_grad == False`. Then a provided cell prints the gradient norms at init: predict which is zero (eq. `lora-grad`) | Exact equality; exact count | 10 |
| 2 | `merged_weight` (eq. `lora-merge`), after a provided loop has trained the layer on a toy regression so B ≠ 0 | A plain `nn.Linear` with the merged weight reproduces the adapter output, `torch.allclose(atol=1e-5)` in float32 | Max absolute difference | 5 |
| 3 | `format_chat(messages, add_generation_prompt)` | String **equals** `tokenizer.apply_chat_template(..., tokenize=False)` for both values of the flag | Exact string match | 6 |
| 4 | `build_labels(input_ids, response_mask)` (eq. `masked-loss`); the mask is 1 on response tokens and the closing `<|im_end|>` | Number of scored positions on a known example; model `.loss` equals a provided hand computation of the masked mean | Loss agreement to 1e-4 | 8 |
| 5 | Complete `LoraConfig`; predict the trainable count before running | `print_trainable_parameters` equals the prediction (460,800 for r = 8 on `q_proj`, `v_proj`; *checked*). Training loop provided and started here. After it: **before vs after** (below) | Held-out response loss and perplexity; stop rate | 10 |
| 6 | `rouge_n(candidate, reference, n)` (eq. `rouge`); then change decoding settings on one prompt, predicting the effect first | Asserts on hand-made pairs, including a repeated-word candidate; printed mean ROUGE-1 before and after | ROUGE-1 on held-out replies | 8 |

Format per exercise: Predict, Run, Explain, Check; `# TODO N` stub, folded solution, short "why this works" note.

## Before vs after: the checkpoint is a number

All on the same 200 held-out examples, same template, greedy decoding, `max_new_tokens` fixed, base model measured **before** `get_peft_model` (it wraps in place).

1. **Response loss and perplexity** (eqs. `masked-loss`, `ppl-resp`), averaged over all response tokens. Assert `loss_after < loss_before - margin`, with the margin set from your measurement across at least three seeds.
2. **Stop rate:** share of replies that emit `<|im_end|>` before the length limit. Expected near 0 before; set the "after" threshold from measurement.
3. **Mean ROUGE-1** against the reference responses (on a subset if generation is slow). Print it; do not assert a direction unless it is reliable across seeds.
4. **Side by side:** the 5 fixed prompts, before and after, in one table. This is for reading and is not asserted.

## Stretch (one section, last, optional): rank sweep

`r` in {1, 4, 16, 64} with `lora_alpha = 2r`, same steps and seed, starting from the base model each time. Plot held-out response loss against trainable parameters (log x-axis). Assert only the parameter counts. Not required by any later lab.

## Compute budget (free Colab T4; whole notebook under 10 minutes; all estimates)

| Step | Budget |
|---|---|
| Setup, model and data download | under 1.5 min |
| "Before" evaluation (loss on 200, generation on about 50) | under 1 min |
| LoRA training, about 250 steps at batch size 16 | 2 to 3 min |
| "After" evaluation | under 1 min |
| Stretch: four short runs of about 100 steps | under 3 min |

**CPU path:** a `FAST` flag (default on when no GPU is found) with about 300 training examples, 40 steps, 50 held-out examples and 10 generations. This is the path CI runs. Measure it; the "after" thresholds will differ and must be set separately.

## What the lab saves (Module 8's stretch loads it)

`model.save_pretrained("lab07-adapter")` and `tokenizer.save_pretrained("lab07-adapter")`. *Checked:* this writes `adapter_config.json`, `adapter_model.safetensors` (1.9 MB at r = 8), `chat_template.jinja` and the tokenizer files, and reloads with `PeftModel.from_pretrained(base, "lab07-adapter")` to identical logits. Also provide `generate_reply(model, tokenizer, messages, **gen_kwargs) -> str`. A Colab runtime does not carry files into another notebook, so add an optional cell that zips the folder for download or copies it to Drive. We do not host the tuned weights (`data/README.md`, ShareAlike).

## Flags for the Lab Engineer

1. **ChatML tokens may be barely trained in the base model.** *Checked:* the base response loss on one example is about 11 nats, far above ordinary text. With frozen embeddings and LoRA on `q_proj` and `v_proj` only, learning to emit `<|im_end|>` may be slow. Measure the stop rate. If it is poor, try in this order: `target_modules="all-linear"`, more steps, then a plain-text template (`### Instruction:` / `### Response:` ending in `<|endoftext|>`). Tell me before changing the template; the lecture names ChatML.
2. Decide whether the newline after the final `<|im_end|>` is scored. The lecture says the mask ends at `<|im_end|>`.
3. Set `tokenizer.pad_token = tokenizer.eos_token`, pad on the right for training, give padding the label `-100`, and pass `pad_token_id` to `generate`.
4. Use the lecture's names: `A`, `B`, `r`, `alpha`, `scale`, `W_0` as `base.weight`, `response_mask`. Write batch size as `batch_size`, never `B`.
5. Pin `transformers` and `peft` in the setup cell after checking what Colab preinstalls. `apply_chat_template(tokenize=True)` returns a `BatchEncoding` in 5.18 (*checked*); pass `return_dict=True` explicitly.
6. Report measured loss, perplexity, stop rate, ROUGE-1, trainable counts and run times (T4 and CPU), and send me the five before-and-after outputs.

## As built (Director review, 2026-10-05)

The notebook was reviewed from its source. huggingface.co was blocked in the build environment, so the SmolLM2-135M and Dolly path is **written, not run**: the model cells ran only on the test-only random stand-ins (`NLP_LLMS_LAB07_OFFLINE=tiny` or `smollm2-shape`), whose numbers are not results. The departures below are accepted; the lecture now matches the notebook.

- **Exercise 5** also has participants write `lora_param_count(model, r, target_modules)`, eq. `lora-count` summed over the targeted `nn.Linear` layers; the checkpoint compares it with `peft`'s count and with 460,800.
- **Exercise 2:** `merged_weight(layer)` is a function attached to the class (`LoRALinear.merged_weight = merged_weight`), not a method in the class body. The toy training loop is 300 Adam steps at rank 4 on a rank-2 target change.
- **Exercise 6:** decoding is an editable `SAMPLING` cell (three seeds next to two greedy runs), not a `# TODO`. `rouge_n` is the only function written.
- **Flag 2 resolved:** a training sequence stops at the closing `<|im_end|>`; the newline the template writes after it is dropped and never scored (`encode_example`, asserted against `format_chat`).
- **Stop tokens:** `generate_reply` and `generate_batch` stop at `<|im_end|>` or `<|endoftext|>` (the tokenizer's `eos_token`, the base model's end-of-document habit). The "stop rate" is reported as three shares: ended with `<|im_end|>`, with `<|endoftext|>`, or at the length limit. `max_new_tokens` is 64 throughout.
- **Template:** ChatML, unchanged (flag 1: no switch to a plain-text template).
- **Training:** AdamW, learning rate 5e-4, 10 warmup steps then linear decay, gradient clipping at 1.0, `lora_dropout=0.0`, batch size 16. Full run 250 steps (two passes over 2,000 examples); `FAST` run 300 train, 50 held-out, 10 generated, 40 steps, 15 stretch steps.
- **Stretch:** one section, last, with two cells: the rank sweep, and one `all-linear` adapter at r = 8 trained for `RUN["steps"]` (2,442,240 trainable), compared on held-out loss and how replies end. On the full run the second cell trains another 250 steps, which may exceed the 3-minute stretch budget; measure it.
- **Not committed:** the Dolly subset under `data/` (the notebook writes `lab07_dolly_subset.jsonl` and prints its hash on the first online run).
- Exercise minutes: 3 + 10 + 5 + 6 + 8 + 10 + 8 = 50.

**Measured without the model** (reported by the Lab Engineer, CPU, `transformers` 5.18.0, `peft` 0.21.2, `torch` 2.14.1): `LoRALinear` on `nn.Linear(576, 192)` with r = 8 equals the base output at initialization; 6,144 trainable = r(d_in + d_out); dL/dA = 0 and dL/dB ≠ 0 at initialization; the merged layer matches the adapter to 1.91e-6; `format_chat` equals `apply_chat_template` exactly. At SmolLM2-135M's shapes (random weights): r = 8 on `q_proj`, `v_proj` gives 460,800 trainable (0.3414% of all parameters including the adapter); r = 1, 4, 16, 64 give 57,600, 230,400, 921,600 and 3,686,400; `all-linear` at r = 8 gives 2,442,240.

**Stand-in observation, not a SmolLM2 result:** on the random stand-ins, `q_proj`/`v_proj` adapters rarely ended replies with `<|im_end|>` (0%), against 80–96% with `all-linear`. Whether this holds on the real model is open; the stretch measures it on the first T4 run. The main path stays on `q_proj`, `v_proj`. The lecture mentions it only as an open question, without these numbers.

## Not verified by the Director

- Every number that needs the real model: base and tuned response loss and perplexity, stop rates, ROUGE-1, the five before-and-after outputs, the size of `lab07-adapter`, and run times on a T4 and on a CPU runtime.
- The loss margins in the after-training checkpoint (0.5 nats full, 0.2 nats `FAST`) are targets, not measurements. Set them from at least three seeds.
- That SmolLM2's `tokenizer.eos_token` is `<|endoftext|>` (id 0) at the pinned revision. The stand-in copies this; the lab's stop logic relies on it.
- The lecture's section 1 sample output is illustrative; the brief's earlier *checked* note (the base model repeats the request and runs to the length limit) was not re-run.
- The `generate` defaults quoted in lecture section 7 (`max_length` 20, `top_k` 50 when unset, `top_k=0` disables top-k) and `merge_and_unload` not being in place were checked against the installed `transformers` 5.18.0 and `peft` 0.21.2 source, not against the live documentation.
- `quarto render` was not run (Quarto is not installed in the build environment).
