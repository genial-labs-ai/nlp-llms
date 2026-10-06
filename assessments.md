---
title: "Assessments"
subtitle: "Entry check and exit check"
---

<!--
Instructor page, rendered by Quarto. Module titles come from _variables.yml through var
shortcodes. Objective numbers refer to the objectives listed at the top of each lecture
page, which are generated from _variables.yml; they are not repeated here. Numbers in
the answers are the build runs recorded in data/baselines.json and quoted in the lectures.
-->

Two short checks. The **entry check** tells a participant, before Day 1, whether the prerequisites are in place. The **exit check** asks one or two questions per module objective, so a participant (or an instructor) can see which objectives landed. Neither is graded. Every question can be answered from the lectures and labs; none is a trick question.

## How to use them

- **Entry check.** Participants take it on the [Before Day 1](prepare.qmd) page, the first step before Setup: about 15 minutes, on paper or in a notebook, with no web search. One rule applies: a participant who misses two or more of the three questions in an area does that area's remediation, listed on the same page, before Day 1; one miss in an area needs no action. Send the page a week ahead, so there is time for the remediation.
- **Exit check.** On Days 2 to 5, the 15 minutes of retrieval practice that open the day use five of these questions: about three from the previous day and two from earlier days. The picks and the routine are on each day page ([Day 2](day-2.qmd#retrieval), [Day 3](day-3.qmd#retrieval), [Day 4](day-4.qmd#retrieval), [Day 5](day-5.qmd#retrieval)). Participants can also use the questions of the day's modules as a self-check that evening, or all of them after Day 5. Answers are folded under each question. Each answer names the lab and exercise that produced the evidence, so a participant who misses a question knows which cell to rerun.
- **Numbers.** Where an answer quotes a measured number, it is the build run recorded in `data/baselines.json` and quoted in the lecture. If your room ran on another runtime and got a different number, accept the room's number with the same reasoning.
- **Labs whose real path has not run.** The [readiness page](readiness.qmd) lists which labs have run on their real models, and where. A question on a lab that has not tests the reasoning of the lecture and the checkpoint, not a model result.

## Entry check {#entry-check}

Participants take this check on the [Before Day 1](prepare.qmd#entry-check) page, which also says how to read the score and what to study in each area.

{{< include /prepare/entry-check.md >}}

## Exit check

Questions are numbered *module.objective*, with a letter when an objective has two. Objective numbers refer to the list at the top of each lecture page. Each question has its own link, with the dot replaced by a hyphen: `#q10-2` for 10.2, `#q12-3a` for 12.3a. The day pages use these links. The outcome numbers refer to the workshop's learning outcomes:

| Outcome | By the end of the workshop a participant can | Modules |
|---|---|---|
| 1 | Explain the line of ideas from count-based language models to transformers, and say what problem each step solved | 1–5 |
| 2 | Implement the core of each model in PyTorch: skip-gram with negative sampling, an LSTM language model, seq2seq with attention, and a small GPT | 2–5 |
| 3 | Use the Hugging Face stack to tokenize, load, fine-tune and evaluate pretrained models, including LoRA | 6, 7 |
| 4 | Use the OpenAI and Claude APIs for prompting, structured output and tool use, and compare them on the same task | 8 |
| 5 | Explain RLHF end to end, train a toy version, and name its failure modes | 9, 10 |
| 6 | Measure calibration, explain how RLCD's objective differs from RLHF's, and use a calibrated decision model (Jev) through its API | 11, 12 |
| 7 | Build and evaluate a RAG pipeline with LlamaIndex and LangChain | 13 |
| 8 | Build a LangGraph agent with tools, state, human-in-the-loop checkpoints and confidence-gated control | 14 |
| 9 | Combine these into one system, evaluate it, and report accuracy, abstention and cost | 15 |

### Module 1 · {{< var modules.m01.title >}}

Objectives: [lecture 1](lectures/01-text-as-data.qmd).

[**1.1**]{#q1-1} (Objective 1 · Outcome 1) At `min_count = 2`, part of the validation split maps to `<unk>`. Name one change that lowers that share, and one cost of making it.

::: {.callout-tip collapse="true" title="Answer"}
Lower `min_count` to 1, enlarge the vocabulary another way, or tokenize at the character level. The cost: more vocabulary entries seen once or twice, whose counts are unreliable (Zipf's long tail), or, for characters, longer sequences and perplexities that are no longer comparable with word-level ones. **Evidence:** Lab 1, Exercise 1 (coverage printed at `min_count` 1 and 2).
:::

[**1.2a**]{#q1-2a} (Objective 2 · Outcome 1) What is the perplexity of a uniform model over a vocabulary of 65 characters, and why?

::: {.callout-tip collapse="true" title="Answer"}
65. Every token gets probability $1/65$, so the mean negative log-probability is $\ln 65$ and its exponential is 65. **Evidence:** Lab 1, Checkpoint 3 (empty counts give perplexity equal to the vocabulary size).
:::

[**1.2b**]{#q1-2b} (Objective 2 · Outcome 1) At the word level, the trigram model's test perplexity (1,437) is higher than the bigram's (304). Why does more context make it worse?

::: {.callout-tip collapse="true" title="Answer"}
Sparsity. At $n = 3$, 77% of the test trigrams never occur in training, so most predictions fall back on small smoothed probabilities. More context helps only while the counts can support it. **Evidence:** Lab 1, Exercise 3 and the baseline card (word-level perplexities, labeled not comparable with later labs).
:::

[**1.3**]{#q1-3} (Objective 3 · Outcome 1) Naive Bayes and logistic regression are both linear classifiers over word counts. How does each set its weights, and which one's probabilities do you expect to be overconfident, and why?

::: {.callout-tip collapse="true" title="Answer"}
Naive Bayes sets its weights by counting words per class, under the assumption that words are independent given the class. Logistic regression sets them by minimizing cross-entropy. Naive Bayes is the overconfident one: correlated words are counted as independent evidence, so the evidence is double-counted. The false assumption harms the probabilities more than the ranking. **Evidence:** Lab 1, Exercises 5 and 6 (both classifiers, per-class precision and recall, the most confident errors).
:::

### Module 2 · {{< var modules.m02.title >}}

Objectives: [lecture 2](lectures/02-word-vectors.qmd).

[**2.1**]{#q2-1} (Objective 1 · Outcome 1) Why do *hot* and *cold* end up close together in a skip-gram embedding space?

::: {.callout-tip collapse="true" title="Answer"}
The distributional hypothesis: words that occur in similar contexts get similar vectors. Antonyms share most of their contexts ("the water is ___"), so cosine similarity reflects shared contexts, not synonymy. **Evidence:** Lab 2, Exercise 2 (nearest neighbors of the probe words).
:::

[**2.2a**]{#q2-2a} (Objective 2 · Outcome 2) Write the skip-gram negative-sampling loss for one center word $w$, its context word $o$ and $K$ noise words $n_1, \dots, n_K$. What is its value when all vectors are zero?

::: {.callout-tip collapse="true" title="Answer"}
$\mathcal{L} = -\log \sigma(u_o^\top e_w) - \sum_{k=1}^{K} \log \sigma(-u_{n_k}^\top e_w)$. With zero vectors every $\sigma(0) = 1/2$, so $\mathcal{L} = (K + 1)\log 2$: the first training loss in the lab. **Evidence:** Lab 2, Exercise 1 (Checkpoint 1 tests the all-zero case; the training cell asserts the first loss).
:::

[**2.2b**]{#q2-2b} (Objective 2 · Outcome 1) What does negative sampling buy over the full softmax, and what does it give up?

::: {.callout-tip collapse="true" title="Answer"}
Cost: each pair costs $O(K)$ instead of $O(|V|)$. It gives up normalized probabilities: it is a different objective, so it cannot be used to compute a perplexity. **Evidence:** lecture 2, section 4; Lab 2, Exercise 1.
:::

[**2.3**]{#q2-3} (Objective 3 · Outcome 2) In Lab 2, averaged skip-gram embeddings with a feed-forward network scored 0.867 test accuracy, against 0.884 for TF-IDF with logistic regression on the same split. Give two reasons from the lecture why the dense model need not win.

::: {.callout-tip collapse="true" title="Answer"}
Averaging discards word order and blurs the few distinctive words that decide a topic. The embeddings were trained on about one million tokens. On topic classification with plenty of labels, word identity carries most of the signal, which TF-IDF keeps. Whether dense features win is an empirical question. **Evidence:** Lab 2, Exercises 3 and 4 and the results table.
:::

### Module 3 · {{< var modules.m03.title >}}

Objectives: [lecture 3](lectures/03-sequence-models.qmd).

[**3.1**]{#q3-1} (Objective 1 · Outcome 2) Write the RNN step, and the two LSTM lines that update the cell state and the hidden state.

::: {.callout-tip collapse="true" title="Answer"}
RNN: $h_t = \tanh(W h_{t-1} + U x_t + b)$. LSTM: $c_t = f_t \odot c_{t-1} + i_t \odot g_t$ and $h_t = o_t \odot \tanh(c_t)$, with forget gate $f_t$, input gate $i_t$, candidate $g_t$ and output gate $o_t$. **Evidence:** Lab 3, Exercises 1 and 4 (checked against `nn.RNNCell` and `nn.LSTMCell`).
:::

[**3.2a**]{#q3-2a} (Objective 2 · Outcome 1) The RNN trained in Lab 3 has $\lVert W \rVert \approx 5.2$, yet its gradient still vanishes with distance. How?

::: {.callout-tip collapse="true" title="Answer"}
The gradient across $T - s$ steps is a product of one-step Jacobians, each with a factor $\mathrm{diag}(1 - h_j^2)$. For saturated $\tanh$ units that factor is near 0, so the product shrinks whatever $\lVert W \rVert$ is. $\lVert W \rVert > 1$ only removes the guarantee of vanishing. **Evidence:** Lab 3, Exercise 3, part A (gradient norm against distance; the printed norm of $W$).
:::

[**3.2b**]{#q3-2b} (Objective 2 · Outcome 1) Does gradient clipping fix vanishing gradients? What does the LSTM change?

::: {.callout-tip collapse="true" title="Answer"}
No. Clipping caps the size of an update, which handles explosion. The LSTM adds a cell state with an additive, gated update; along that path the Jacobian is a product of forget gates, which the network can hold near 1. In the lab, 50 characters back, the LSTM's gradient was about $5 \times 10^4$ times the RNN's. **Evidence:** Lab 3, Exercise 3, part B (clipping demonstration) and the LSTM gradient plot after Exercise 4.
:::

[**3.3**]{#q3-3} (Objective 3 · Outcome 1) Why may Lab 3 compare the LSTM's test perplexity (5.01) with Lab 1's character 5-gram (6.25), but not with Lab 1's word-level bigram (304)?

::: {.callout-tip collapse="true" title="Answer"}
Perplexities are comparable only at the same token unit, on the same test text, vocabulary and scored positions. The LSTM and the 5-gram are both scored per character on the same 60,394 test characters with the same 65-character vocabulary. A word-level perplexity is on a different scale. **Evidence:** Lab 3, the results table (the checkpoint asserts the LSTM beats the trigram and the 5-gram).
:::

### Module 4 · {{< var modules.m04.title >}}

Objectives: [lecture 4](lectures/04-seq2seq-attention.qmd).

[**4.1**]{#q4-1} (Objective 1 · Outcome 2) In the plain encoder-decoder, what is the decoder's first state, and how is the training loss computed?

::: {.callout-tip collapse="true" title="Answer"}
The last encoder state, $s_0 = h_S$. The decoder is trained with teacher forcing: its inputs are the gold target tokens shifted right by one, and the loss is the cross-entropy of each next target token, averaged over non-padding tokens. **Evidence:** Lab 4, Exercise 1 (`forward_plain`, `seq2seq_loss`).
:::

[**4.2**]{#q4-2} (Objective 2 · Outcome 1) Without attention, Lab 4's model gets about 0.9 exact match on inputs with one date and 0 on inputs with two or more, under the same training budget as the attention model. Explain why, and say why this is not the vanishing-gradient problem.

::: {.callout-tip collapse="true" title="Answer"}
The whole source must pass through one fixed-size vector, while the information the decoder needs grows with the number of dates. That is a capacity limit: the fixed-vector bottleneck. Vanishing gradients are an optimization problem on long paths; the bottleneck would remain even with perfect training. **Evidence:** Lab 4, Exercises 2 and 3 (exact match by $K$, without and with attention).
:::

[**4.3a**]{#q4-3a} (Objective 3 · Outcome 2) Write dot-product attention for a decoder state $s_t$ over encoder states $h_1, \dots, h_S$ with a padding mask. How does the additive score differ?

::: {.callout-tip collapse="true" title="Answer"}
Scores $e_{t,i} = s_t^\top h_i$; set masked positions to $-\infty$; weights $\alpha_{t,i} = \mathrm{softmax}_i(e_{t,i})$ over source positions; context $\bar{h}_t = \sum_i \alpha_{t,i} h_i$. The additive score replaces the dot product with a small network, $u^\top \tanh(W_q s_t + W_k h_i)$. **Evidence:** Lab 4, Exercises 3 and 4 (weights sum to 1; padded positions get exactly 0).
:::

[**4.3b**]{#q4-3b} (Objective 3 · Outcome 1) A row of the attention heat-map puts 0.99 on one source character. What does that show, and what does it not show?

::: {.callout-tip collapse="true" title="Answer"}
It shows which encoder state the decoder read at that step: a soft alignment. It does not show how the information was used, so it is evidence, not an explanation. The softmax is over source positions, never exactly 0 or 1. **Evidence:** Lab 4, Exercise 5 (heat-maps and the alignment hit rate).
:::

### Module 5 · {{< var modules.m05.title >}}

Objectives: [lecture 5](lectures/05-transformer-from-scratch.qmd).

[**5.1**]{#q5-1} (Objective 1 · Outcome 2) Why are attention scores divided by $\sqrt{d_k}$? Use Lab 5's measurement.

::: {.callout-tip collapse="true" title="Answer"}
With unit-variance entries, $q^\top k$ has variance $d_k$, so the score's standard deviation grows as $\sqrt{d_k}$: the lab measured about 4, 8 and 16 for $d_k$ = 16, 64 and 256, and about 1 after scaling. Large scores saturate the softmax, which becomes nearly one-hot with tiny gradients. **Evidence:** Lab 5, Exercise 1 and its provided measurement.
:::

[**5.2a**]{#q5-2a} (Objective 2 · Outcome 2) With $d_k = d / n_h$, does going from one head to four change the number of parameters?

::: {.callout-tip collapse="true" title="Answer"}
No. Four heads are four narrower lookups with the same total projection size. **Evidence:** lecture 5, section 4; the stretch of Lab 5 asserts the count does not change with $n_h$.
:::

[**5.2b**]{#q5-2b} (Objective 2 · Outcome 2) List the parts of a pre-norm decoder-only transformer from token IDs to logits, and say what the causal mask guarantees.

::: {.callout-tip collapse="true" title="Answer"}
Token embeddings plus position embeddings; $n_\ell$ blocks, each $x + \mathrm{Attn}(\mathrm{LN}(x))$ followed by $x + \mathrm{FFN}(\mathrm{LN}(x))$; a final layer norm; the output layer. The mask sets scores of later positions to $-\infty$, so position $t$ depends only on tokens $1, \dots, t$, and one forward pass gives $T$ valid predictions. **Evidence:** Lab 5, Exercises 0, 2 and 3 (the leak tests on one step and on the whole model).
:::

[**5.3**]{#q5-3} (Objective 3 · Outcome 1) At full budget the mini-GPT scored 1.548 nats per character and the LSTM retrained with the same loop 1.573, from one seed; the seed-to-seed spread was 0.02 to 0.03 nats. What may you conclude?

::: {.callout-tip collapse="true" title="Answer"}
A tie at this budget, not a ranking. The difference is within seed noise. The comparison is meaningful only with each model's parameter count, steps and time stated; with a shorter budget the LSTM won clearly. **Evidence:** Lab 5, Exercise 4 (the results table).
:::

### Module 6 · {{< var modules.m06.title >}}

Objectives: [lecture 6](lectures/06-pretraining-huggingface.qmd).

[**6.1a**]{#q6-1a} (Objective 1 · Outcome 3) Byte-pair encoding on the corpus *low* ×5, *lower* ×2, *newest* ×6, *widest* ×3, with an end-of-word symbol: what is the count of the pair (e, s), and what is the first merge?

::: {.callout-tip collapse="true" title="Answer"}
9 (6 from *newest*, 3 from *widest*). It ties with (s, t), also 9; ties go to the pair that sorts first, so the first merge is `e + s`. The lab's first five merges are `e+s`, `es+t`, `est+_`, `l+o`, `lo+w`. **Evidence:** Lab 6, Exercise 1.
:::

[**6.1b**]{#q6-1b} (Objective 1 · Outcome 3) In masked language modeling, what happens to the selected positions, and why does BERT have no perplexity?

::: {.callout-tip collapse="true" title="Answer"}
15% of positions are selected; of those, 80% become `[MASK]`, 10% a random token and 10% stay unchanged, and all selected positions are predicted. The model uses context on both sides, so it defines no left-to-right probability of a sentence: no perplexity in Module 1's sense, and no generation. **Evidence:** Lab 6, Exercise 4 (the three shares).
:::

[**6.2**]{#q6-2} (Objective 2 · Outcome 3) You load a pretrained body into `AutoModelForSequenceClassification` and see a warning about newly initialized weights. Is something broken? And what must always match the model?

::: {.callout-tip collapse="true" title="Answer"}
Nothing is broken: the classification head is new and must be trained. The tokenizer must be the model's own, because token IDs index rows of the model's embedding matrix; the wrong tokenizer gives wrong rows and no error. **Evidence:** Lab 6, Exercises 2, 3 and 5 (`AutoTokenizer` and the `AutoModelFor...` classes).
:::

[**6.3**]{#q6-3} (Objective 3 · Outcome 3) Before fine-tuning a four-class classifier, what validation loss do you expect, and which parameters does fine-tuning update?

::: {.callout-tip collapse="true" title="Answer"}
About $\ln 4 \approx 1.386$: a fresh head gives near-uniform predictions. Standard fine-tuning updates every parameter, body and head, with a small learning rate for a few epochs; training only the head is a different, cheaper method. **Evidence:** Lab 6, Exercise 5 (the starting loss is checked against $\log 4$).
:::

### Module 7 · {{< var modules.m07.title >}}

Objectives: [lecture 7](lectures/07-finetuning-lora.qmd).

[**7.1**]{#q7-1} (Objective 1 · Outcome 3) Instruction tuning changes which of these: the architecture, the loss, or the data? Which token positions are scored?

::: {.callout-tip collapse="true" title="Answer"}
The data. The model and the next-token cross-entropy are unchanged; the examples are formatted with a chat template, and the loss is averaged over the response tokens only, including the closing `<|im_end|>`. **Evidence:** Lab 7, Exercises 3 and 4 (`format_chat`, `build_labels`).
:::

[**7.2a**]{#q7-2a} (Objective 2 · Outcome 3) LoRA at rank $r = 8$ on `q_proj` ($576 \to 576$) and `v_proj` ($576 \to 192$) in each of 30 blocks: how many trainable parameters?

::: {.callout-tip collapse="true" title="Answer"}
$r(d_{\text{in}} + d_{\text{out}})$ per layer: $8 \times 1{,}152 + 8 \times 768 = 15{,}360$ per block, and $460{,}800$ in all, about 0.34% of the model. **Evidence:** Lab 7, Exercise 5 (`lora_param_count` against `peft`).
:::

[**7.2b**]{#q7-2b} (Objective 2 · Outcome 3) Why does LoRA training start exactly at the pretrained model, and why can it still move?

::: {.callout-tip collapse="true" title="Answer"}
$B = 0$ at initialization, so $BA = 0$ and $W = W_0$. The gradient of $B$ depends on $Ax$, which is not zero, so $B$ moves at the first step; it is $A$'s gradient that is zero at first. After training, $BA$ can be merged into $W_0$ at no inference cost. **Evidence:** Lab 7, Exercises 1 and 2 and the provided gradient cell.
:::

[**7.3**]{#q7-3} (Objective 3 · Outcome 3) After fine-tuning, response perplexity falls and ROUGE-1 rises. Name one thing neither number shows, and two decoding settings you set when measuring.

::: {.callout-tip collapse="true" title="Answer"}
Perplexity measures the probability of the reference responses, not the quality of generated text; ROUGE counts shared words with one reference, so a wrong answer can score well. Neither shows correctness: read the outputs. When measuring, decode greedily, and always set a stop token and a length limit. **Evidence:** Lab 7, Exercise 6 and the before-and-after checkpoint.
:::

### Module 8 · {{< var modules.m08.title >}}

Objectives: [lecture 8](lectures/08-llm-apis.qmd).

[**8.1a**]{#q8-1a} (Objective 1 · Outcome 4) In a tool-calling loop, who runs the tool, and what goes back to the model?

::: {.callout-tip collapse="true" title="Answer"}
Your code runs it; the model only writes a request. You append the model's reply and one result per call, matched by the call's ID. An unknown tool or bad arguments become an error result, not an exception. The loop stops at $K_{\max}$ calls. **Evidence:** Lab 8, Exercise 4 (`execute`, `run_tools`).
:::

[**8.1b**]{#q8-1b} (Objective 1 · Outcome 4) A structured-output call returns JSON that validates against your schema. What does that guarantee?

::: {.callout-tip collapse="true" title="Answer"}
Form, not truth. It guarantees a parseable record; whether the fields are right is a separate number. Validity and exact match are reported separately. **Evidence:** Lab 8, Exercises 2 and 3 (`extract` with validate-and-retry; `score`).
:::

[**8.2**]{#q8-2} (Objective 2 · Outcome 4) Why does Lab 8 count an invalid output as wrong, and print a standard error beside every exact-match rate?

::: {.callout-tip collapse="true" title="Answer"}
If invalid outputs were dropped, a provider that fails more often could look more accurate: the denominator is always $N$. With $N$ = 40 (or 12 on the CPU fallback), the standard error is large, so small differences between providers cannot be read. **Evidence:** Lab 8, Exercise 3.
:::

[**8.3**]{#q8-3} (Objective 3 · Outcome 4) At \$1 per million input tokens and \$5 per million output tokens, what does a request with 400 input and 60 output tokens cost? Why does a tool loop's input cost grow faster than its number of calls?

::: {.callout-tip collapse="true" title="Answer"}
$(400 \times 1 + 60 \times 5)/10^6 = \$0.0007$. The API is stateless, so every call resends the whole message list, which grows by about $\delta$ tokens per round: input tokens total $K n_0 + \delta K(K-1)/2$, quadratic in $K$. **Evidence:** Lab 8, Exercise 5 (`cost_usd`; measured input tokens against the formula).
:::

### Module 9 · {{< var modules.m09.title >}}

Objectives: [lecture 9](lectures/09-preference-learning.qmd).

[**9.1**]{#q9-1} (Objective 1 · Outcome 5) In the reinforcement-learning view of text generation, what are the state, the action and the policy?

::: {.callout-tip collapse="true" title="Answer"}
The state is the prompt plus the response so far; the action is the next token; the policy is the language model's next-token distribution. The log-probability of a response is the sum of its tokens' log-probabilities. **Evidence:** Lab 9, Part A (the toy sequence task) and lecture 9, section 5.
:::

[**9.2a**]{#q9-2a} (Objective 2 · Outcome 5) Write the REINFORCE loss with a baseline, and say why the baseline does not bias the gradient.

::: {.callout-tip collapse="true" title="Answer"}
$-\big((G - b).\mathrm{detach}() \cdot \log \pi_\theta\big)$, summed over positions and averaged over the batch. The score function $\nabla \log \pi_\theta$ has mean zero, so subtracting a baseline that does not depend on the sampled response leaves the expected gradient unchanged and can cut its variance. **Evidence:** Lab 9, Exercises 1 and 2 (mean estimates against the exact gradient, with and without a baseline).
:::

[**9.2b**]{#q9-2b} (Objective 2 · Outcome 5) Why must the leave-one-out baseline exclude the sample's own reward, and how does the batch-mean baseline relate to it?

::: {.callout-tip collapse="true" title="Answer"}
A baseline that includes the sample's own reward depends on the response it is subtracted from, which shrinks the gradient. Using the batch mean as the baseline gives exactly $(1 - 1/N)$ times the gradient that the leave-one-out baseline gives. **Evidence:** Lab 9, Exercise 2 (the exact identity is a checkpoint).
:::

[**9.3**]{#q9-3} (Objective 3 · Outcome 5) Write the Bradley–Terry probability that response $a$ is preferred to $b$, and the reward-model loss. Why can the reward model not reach 100% held-out accuracy?

::: {.callout-tip collapse="true" title="Answer"}
$P(a \succ b) = \sigma\big((g_a - g_b)/\tau_{\text{label}}\big)$; loss $-\log \sigma\big(r_\phi(y_w) - r_\phi(y_l)\big)$. Raters are noisy and many pairs are close, so even the true rule would mislabel some pairs: the ceiling is $\mathrm{Acc}^\star$, often well below 1. **Evidence:** Lab 9, Exercises 3 and 4 (`bt_prob`, `bt_loss`, held-out accuracy beside $\mathrm{Acc}^\star$).
:::

### Module 10 · {{< var modules.m10.title >}}

Objectives: [lecture 10](lectures/10-rlhf.qmd).

[**10.1**]{#q10-1} (Objective 1 · Outcome 5) Name the three stages of the InstructGPT pipeline and what each produces.

::: {.callout-tip collapse="true" title="Answer"}
Supervised fine-tuning on demonstrations (the SFT model, which becomes the reference); a reward model trained on pairwise comparisons; policy optimization against the reward model with a KL penalty toward the reference. **Evidence:** Lab 9 (the reward model) and Lab 10 (the policy step).
:::

[**10.2**]{#q10-2} (Objective 2 · Outcome 5) Write the per-token reward Lab 10 uses. What does $\beta$ control, and what does the KL penalty not protect against?

::: {.callout-tip collapse="true" title="Answer"}
$r_t = -\beta\big(\log \pi_\theta(y_t \mid s_t) - \log \pi_{\text{ref}}(y_t \mid s_t)\big) + \mathbf{1}[t = |y|]\, r_\phi(x, y)$. $\beta$ sets the trade-off between reward and drift from the reference; changing it changes the optimum, not only the speed. The penalty punishes leaving the reference's support, not concentrating on part of it, so it does not prevent mode collapse. **Evidence:** Lab 10, Exercises 2 and 3 (`token_rewards`, `exact_kl`).
:::

[**10.3**]{#q10-3} (Objective 3 · Outcome 5) What is the DPO loss when the policy equals the reference, and what does DPO remove from RLHF?

::: {.callout-tip collapse="true" title="Answer"}
$\log 2$: both implicit rewards are 0 and $-\log \sigma(0) = \log 2$. DPO removes the reward model and the sampling loop; it keeps the KL-regularized objective and the reference model. Both log-probabilities can fall during training; DPO raises the gap. **Evidence:** Lab 10, Exercise 5 (`dpo_loss`).
:::

[**10.4**]{#q10-4} (Objective 4 · Outcome 5) With $\beta = 0$, the reward model's score of the policy rises. What does the gold rule show, and why can the reward model not see it?

::: {.callout-tip collapse="true" title="Answer"}
Reward hacking: the policy drifts far from the reference, stuffs positive words, and loses variety, and the gold reward falls because the rule caps positive words and penalizes crowding. The reward model was trained on the reference's own samples, where the cap and the crowding term almost never fire, so it never learned them. Other failure modes named in the lecture: mode collapse, sycophancy, and optimizing for what raters can judge rather than what is true. **Evidence:** Lab 10, Exercise 4 (the asserted signature; not yet verified across seeds on a T4).
:::

### Module 11 · {{< var modules.m11.title >}}

Objectives: [lecture 11](lectures/11-calibration.qmd).

[**11.1**]{#q11-1} (Objective 1 · Outcome 6) A classifier gives confidence 0.8 on 1,000 cases. What must be true for it to be calibrated there? Can a calibrated model be useless?

::: {.callout-tip collapse="true" title="Answer"}
About 800 of those cases must be right. Yes: a model that ignores its input and predicts the class frequencies (0.25 each on a balanced four-class set) is calibrated, with ECE 0, and is 25% accurate. **Evidence:** Lab 11, Exercise 1 (the input-blind check).
:::

[**11.2**]{#q11-2} (Objective 2 · Outcome 6) Lab 1's classifier at `C = 1` has test accuracy 0.888 and mean confidence 0.723. Is it overconfident or underconfident? Why would you not read much into an ECE of 0.05 measured on 100 examples?

::: {.callout-tip collapse="true" title="Answer"}
Underconfident: it is right more often than it claims (ECE 0.164). ECE has a positive noise floor: a perfectly calibrated forecaster measured on 100 examples with 15 bins scores about 0.07 on average. **Evidence:** Lab 11, Exercises 1 and 2 (the `C` sweep and the printed noise floor).
:::

[**11.3**]{#q11-3} (Objective 3 · Outcome 6) Why is the Brier score a proper scoring rule while accuracy is not?

::: {.callout-tip collapse="true" title="Answer"}
For a binary event with true probability $p$, the expected Brier score of a report $q$ is $(q - p)^2 + p(1 - p)$, minimized only at $q = p$: honesty pays. Expected accuracy is the same for every $q > 0.5$, so it cannot reward honest probabilities; the linear score even rewards overconfidence. **Evidence:** Lab 11, the stretch (expected scores on a grid), and Exercise 2.
:::

[**11.4**]{#q11-4} (Objective 4 · Outcome 6) A wrong action costs 10 and deferring costs 1. Above what confidence should the system act? Does temperature scaling change the risk–coverage curve?

::: {.callout-tip collapse="true" title="Answer"}
Act when $\hat{p} > \lambda^* = 1 - 1/10 = 0.9$. The curve depends only on the order of the confidences, which temperature scaling keeps for two classes (for four classes the top-class order changed on 0.92% of test pairs). What changes is where the cost-based threshold sits. **Evidence:** Lab 11, Exercises 3 and 5.
:::

### Module 12 · {{< var modules.m12.title >}}

Objectives: [lecture 12](lectures/12-rlcd-jev.qmd). These questions follow the honesty rule: they test the difference between what TypeSafe has stated and what is our illustration. The key for 12.3a follows lecture 12, section 5, as corrected from TypeSafe's documentation on 2026-10-06; re-check it if the workshop lead's sign-off changes that section.

[**12.1**]{#q12-1} (Objective 1 · Outcome 6) In the lecture's framing (ours, not TypeSafe's method), how does an outcome reward with a proper score differ from RLHF's preference reward? Name one thing Lab 12's toy experiment does not show.

::: {.callout-tip collapse="true" title="Answer"}
A preference reward is relative, comes from a model of raters, and never sees a probability. An outcome reward is computed from a verified label and can score the whole reported distribution; with a proper score such as the Brier score, only the true probabilities earn the most reward, whereas an accuracy reward is maximized by putting all probability on one answer. The toy does not show how Jev was trained (TypeSafe has not published RLCD), nor that training for calibration beats repairing it afterwards: on held-out wordings both toy models were overconfident, and temperature scaling closed most of the gap. **Evidence:** Lab 12, Exercise 1 (the table on `train`, `dev` and `test`).
:::

[**12.2**]{#q12-2} (Objective 2 · Outcome 6) With costs wrong = 20, ask = 0.5, miss = 4 and escalate = 3, compute $\tau_{\text{act}}$ and $\tau_{\text{esc}}$. What does the system do at $\hat{p} = 0.6$?

::: {.callout-tip collapse="true" title="Answer"}
$\tau_{\text{act}} = 1 - 0.5/(20 - 4) = 0.96875$ and $\tau_{\text{esc}} = 1 - (3 - 0.5)/4 = 0.375$. At 0.6 it asks: the expected costs are 8 (act), 2.1 (ask) and 3 (escalate). **Evidence:** Lab 12, Exercise 4 (the worked example is a checkpoint).
:::

[**12.3a**]{#q12-3a} (Objective 3 · Outcome 6) Sort each statement into one of three groups: stated by TypeSafe in its own sources; reported by third parties and not found in a TypeSafe source; our illustration.

1. "System One models are trained for calibrated decisions; validate their performance in the target domain."
2. RLCD optimizes calibrated probabilities instead of preference, as RLHF does.
3. A Brier-score reward keeps a small decision model's probabilities calibrated on its training items, while an accuracy reward drives them toward 1.
4. Jev answers in 70 to 500 ms.
5. A choice answer carries a probability for every option and a separate `confidence` that summarizes how concentrated the distribution is.

::: {.callout-tip collapse="true" title="Answer"}
Stated by TypeSafe: 1 (`typesafe-ai/skills`, `SKILL.md`; the documentation's System One page states the same first clause), 2 in substance, and 5 (`typesafe-sdk`, `SKILL.md` and the documentation's Confidence page). For 2, TypeSafe's AI primer (read 2026-10-06) says that RLCD "trains TypeSafe to return decisions and calibrated probabilities instead of generated text", while RLHF "trains models to produce responses people prefer"; the words "instead of preference" are the third parties' summary, not TypeSafe's. Reported by third parties: 4. It is not in TypeSafe's documentation, which says instead that "Most queries complete in about 100 ms". Our illustration: 3, from Lab 12's toy model; it is evidence about our toy, not about how Jev was trained. TypeSafe names RLCD and states its aim, but has not published its reward, data or algorithm. **Evidence:** lecture 12, section 5; Lab 12, Exercise 1 and the closing cell.
:::

[**12.3b**]{#q12-3b} (Objective 3 · Outcome 6) A yes/no answer returns `noul = 0.2`. What is the chosen answer and its probability? A choice answer returns probabilities $(0.6, 0.1, 0.1, 0.1, 0.1)$ and `confidence = 0.5`. Which number goes on the reliability diagram?

::: {.callout-tip collapse="true" title="Answer"}
"No", with probability $1 - 0.2 = 0.8$ ("yes" when `noul >= 0.5`). For the choice answer, 0.6: the probability of the chosen answer. `confidence` measures how concentrated the distribution is; on calibrated synthetic data its ECE was above 0.10 while the probabilities' was below 0.01. **Evidence:** Lab 12, Exercises 2 and 3 (`chosen_answer`, `calibration_arrays`).
:::

### Module 13 · {{< var modules.m13.title >}}

Objectives: [lecture 13](lectures/13-rag.qmd).

[**13.1a**]{#q13-1a} (Objective 1 · Outcome 7) Name the seven stages of a RAG pipeline, in order.

::: {.callout-tip collapse="true" title="Answer"}
Load, chunk, embed, index, retrieve, rerank, generate. **Evidence:** Lab 13, Exercises 2 to 5.
:::

[**13.1b**]{#q13-1b} (Objective 1 · Outcome 7) What can a reranker not fix, and which number should a Jev reranker sort by?

::: {.callout-tip collapse="true" title="Answer"}
It only reorders the $k_0$ candidates it is given, so it cannot recover evidence the retriever missed. Sort by the expected relevance level computed from the probabilities (`score`), not by `confidence`. **Evidence:** Lab 13, Exercise 4.
:::

[**13.2a**]{#q13-2a} (Objective 2 · Outcome 7) Chunks of 512 tokens reach a higher recall@5 than chunks of 128. Why is that not enough to choose 512, and on which split do you choose?

::: {.callout-tip collapse="true" title="Answer"}
Larger chunks contain more text, so they cover more evidence at the same $k$. Compare settings at a fixed context budget, $n_{\text{ctx}} = kL$. Choose on `dev`; report the choice on `test` once, with gained and lost questions. **Evidence:** Lab 13, Exercise 2 (the sweep at a fixed budget).
:::

[**13.2b**]{#q13-2b} (Objective 2 · Outcome 7) Three questions have their first relevant chunk at ranks 3, 1 and nowhere in the list. What is the MRR?

::: {.callout-tip collapse="true" title="Answer"}
$(1/3 + 1 + 0)/3 \approx 0.444$. **Evidence:** Lab 13, Exercise 1 (the lecture's worked example is the checkpoint).
:::

[**13.3**]{#q13-3} (Objective 3 · Outcome 7) An answer is fully faithful to its retrieved passages, and wrong. Where is it in the two-by-two table of evidence retrieved against correct, and what would you change first?

::: {.callout-tip collapse="true" title="Answer"}
Evidence not retrieved, answer wrong: a retrieval failure. It is faithful to the wrong passages, so change retrieval first (chunking, $k$, hybrid retrieval, reranking). A faithfulness score is a judge model's opinion until it has been compared with human labels. **Evidence:** Lab 13, Exercise 5 and the two-by-two table.
:::

### Module 14 · {{< var modules.m14.title >}}

Objectives: [lecture 14](lectures/14-agents.qmd).

[**14.1**]{#q14-1} (Objective 1 · Outcome 8) What does an explicit LangGraph graph make visible that Module 8's tool loop did not, and what stops a runaway loop?

::: {.callout-tip collapse="true" title="Answer"}
The state, the nodes that update it, the edges and conditional routing that read it, and a checkpoint after every step. A runaway loop is stopped by your own count of model calls ($K_{\max}$, kept in the state) and a `recursion_limit` set on every run; the default in `langgraph` 1.2.12 is 10,007 steps. **Evidence:** Lab 14, Exercise 2 (a script that never finishes ends at $K_{\max}$).
:::

[**14.2**]{#q14-2} (Objective 2 · Outcome 8) A node calls the decision model and then calls `interrupt()`. What happens when the run resumes, and how do you fix it? What does replay re-execute?

::: {.callout-tip collapse="true" title="Answer"}
LangGraph reruns the interrupted node from its first line, so the decision model is called twice. Move the decision call into an earlier node; the lab checks it is called once. Replay re-executes every node after the snapshot, including tools that act on the world, so side effects must be idempotent. Memory comes from the thread: a second call with the same `thread_id` sees the earlier messages. **Evidence:** Lab 14, Exercises 3 and 4.
:::

[**14.3a**]{#q14-3a} (Objective 3 · Outcome 8) The guard's thresholds are $\tau_{\text{esc}} = 0.375$ and $\tau_{\text{act}} = 0.969$. What happens at $p_{\text{allow}} = 0.6$? Why does the guard compare `noul` itself rather than the probability of its chosen answer?

::: {.callout-tip collapse="true" title="Answer"}
It asks: the run pauses for a person. The guard's question is "is this email allowed?", and the threshold is about the probability of "allowed", so it compares `noul` directly; $\max(\texttt{noul}, 1 - \texttt{noul})$ would treat a confident "not allowed" as a reason to act. **Evidence:** Lab 14, Exercise 2 (`region`, `after_guard`).
:::

[**14.3b**]{#q14-3b} (Objective 3 · Outcome 8) A guard scores an unsafe-action rate of 0 out of 45 disallowed items. What may you claim, and what else must you report?

::: {.callout-tip collapse="true" title="Answer"}
Not that the guard is safe: 0 out of 45 is consistent with a true rate of several percent, and claiming below 1% would take hundreds of disallowed items with no failure. A guard that escalates everything also scores 0, so report the held-back rate and the number of interrupts beside it, with $N$. **Evidence:** Lab 14, Exercise 5 and the closing cell.
:::

### Module 15 · {{< var modules.m15.title >}}

Objectives: [lecture 15](lectures/15-capstone.qmd). These questions were written from lecture 15 before `15-capstone.ipynb` was built; check them against the notebook's starter system before you use them. The capstone's evaluation set does not exist yet, so no question here asks for a measured result.

[**15.1**]{#q15-1} (Objective 1 · Outcome 9) In the capstone starter, which nodes ask the decision model a question, and what does each decide?

::: {.callout-tip collapse="true" title="Answer"}
`plan` asks whether the question needs evidence from two places; at $p_{\text{multi}} \ge \tau_{\text{plan}} = 0.5$ the generator splits it into two queries. `verify` asks whether every claim is supported; at $p_{\text{sup}} \ge \tau_{\text{verify}} = 0.8$ the draft is delivered, otherwise the system retrieves again with $2k$ chunks if a retry is left, and abstains if not. **Evidence:** Lab 15, the exercise (`after_verify`) and the self-test.
:::

[**15.2a**]{#q15-2a} (Objective 2 · Outcome 9) With $\ell_{\text{wrong}} = 5$ and $\ell_{\text{abs}} = 1$, and 40 answerable and 25 unanswerable questions, what is the cost per question of abstaining on everything? When does answering pay?

::: {.callout-tip collapse="true" title="Answer"}
$40/65 \approx 0.62$: each answerable question costs 1 when abstained on, each unanswerable one 0. Answering pays only when the answer is right more than $1 - 1/5 = 80\%$ of the time (Chow's rule). **Evidence:** Lab 15, the scoring cell (lecture 15, section 3).
:::

[**15.2b**]{#q15-2b} (Objective 2 · Outcome 9) A change gains 7 questions and loses 2 against the baseline. Is the improvement shown? What else do you compare $g + l$ with?

::: {.callout-tip collapse="true" title="Answer"}
Not at the 0.05 level: the sign test gives $p \approx 0.18$ (8 and 1 would give 0.039). Report it as "not shown", not "no effect". Compare $g + l$ with the run-to-run flip count between two baseline runs; a change that moves fewer questions than the flips has shown nothing. **Evidence:** Lab 15, `compare` and `flips` on the share card.
:::

[**15.3**]{#q15-3} (Objective 3 · Outcome 9) A two-minute share answers four questions. What are they, in order?

::: {.callout-tip collapse="true" title="Answer"}
What you changed, and why the traces suggested it; what you predicted; what happened to the cost, accuracy and abstention, with $g$, $l$ and $p$; what it cost in dollars or calls and in seconds. End with the one thing that surprised you. **Evidence:** Lab 15, the hypothesis card and the share card.
:::
