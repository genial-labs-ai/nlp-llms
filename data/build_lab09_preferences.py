"""Build Lab 9's preference data: data/lab09_prompts.json and data/lab09_preferences.jsonl.gz.

The pairs are two continuations of the same prompt sampled from Lab 10's reference
policy, the small pretrained GPT-2 of Lab 6 (`models.causal_lm` in _variables.yml),
labeled by a synthetic rater: Bradley-Terry with a label temperature on a gold rule we
wrote. The gold rule is known only because the data are synthetic. Notebooks never run
this script; they load the committed files (see data/README.md). The third file Lab 10
needs, data/lab09_reward_model.pt, is written by Lab 9's own solution code (see the end
of this docstring).

Needs huggingface.co (the model and tokenizer, about 350 MB). A GPU is optional: a T4
takes an estimated minute of sampling, a CPU an estimated 15 to 25 minutes (neither
measured yet). Run from the repository root:

    # 1. Statistics only: 1,000 train pairs (2,000 samples), prints the acceptance table,
    #    writes nothing. Send the table to the Academic Director and the Lab 10 author.
    python data/build_lab09_preferences.py --stats-only

    # 2. The full build into data/. Refuses to write if an acceptance criterion fails.
    python data/build_lab09_preferences.py

    # 3. The reward model, from the notebook's own solution code at seed 0 (CPU is enough).
    #    test_notebooks.py runs the notebook in a temporary directory, so the path is absolute.
    NLP_LLMS_REWARD_MODEL_OUT="$PWD/data/lab09_reward_model.pt" \\
        python scripts/test_notebooks.py 09-preference-learning
    #    Seeds 1 and 2, kept outside the repository for Lab 10's robustness check:
    NLP_LLMS_RM_SEED=1 NLP_LLMS_REWARD_MODEL_OUT=/tmp/lab09_reward_model_seed1.pt \\
        python scripts/test_notebooks.py 09-preference-learning

Offline test of everything except GPT-2 (word-level stand-in tokenizer and sampler; the
files it writes are NOT the dataset and are refused inside data/):

    python data/build_lab09_preferences.py --offline-tiny --out /tmp/lab09-stand-in

Recipe, all of it fixed:

- Prompts: every subject (2 tokens) + frame (6 tokens), 256 prompts of exactly
  PROMPT_LEN = 8 tokens, split by frame into train (160), held-out (32) and Lab 10's
  evaluation prompts (64). No end-of-text token in front.
- Pairs: 8,000 train pairs (50 per train prompt) and 1,000 held-out pairs. Pair i of a
  split uses prompt i mod (number of prompts) of that split, in an order shuffled with
  random.Random(seed).
- Responses: two independent samples per pair from the reference policy in float32 and
  eval() mode, temperature 1, no top-k or nucleus truncation, from logits[..., :EOS_ID],
  so end-of-text is never sampled and every response has exactly RESPONSE_LEN = 24
  tokens. torch.Generator seeded with `seed` on the sampling device.
- Labels: response a is chosen with probability bt_prob(g_a, g_b, GOLD["tau_label"]),
  drawn from random.Random(seed + 1).
- Response text is tokenizer.decode(ids, clean_up_tokenization_spaces=False).

Sampling on different hardware or library versions is not bit-for-bit reproducible, so
the committed files and their SHA-256 hashes are the dataset, not this script.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
import platform
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

DATA = Path(__file__).resolve().parent
MODEL = "distilbert/distilgpt2"  # models.causal_lm in _variables.yml; tests check they agree
FORMAT_PROMPTS = "nlp-llms/lab09-prompts"
SEED = 0
TRAIN_PAIRS, HELDOUT_PAIRS = 8000, 1000
STATS_ONLY_PAIRS = 1000

# ---- Restated verbatim from the Lab 9 -> Lab 10 interface (briefs/09-preference-learning.md).
PROMPT_LEN = 8      # tokens in every prompt; the build script asserts it
RESPONSE_LEN = 24   # tokens in every response; end-of-text is never sampled
EOS_ID = 50256      # == tokenizer.eos_token_id == len(tokenizer) - 1; asserted on load

WORD = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)*")
GOLD = {
    "positive": ["good", "great", "happy", "love", "loved", "wonderful", "beautiful", "nice",
                 "best", "amazing", "perfect", "excellent", "glad", "fun", "enjoyed",
                 "delicious", "proud", "lovely", "fantastic", "excited"],
    "negative": ["bad", "sad", "terrible", "awful", "hate", "angry", "worst", "horrible",
                 "poor", "wrong", "sick", "afraid", "worried", "upset", "disappointed",
                 "pain", "ugly", "cried", "scared", "dead"],
    "cap": 3, "crowd_threshold": 0.25, "crowd_weight": 10.0, "tau_label": 0.5,
}

def complete_words(prompt, response):
    """Lowercased words that end inside the response. A first word that continues the
    prompt's last word is joined to it; a word that touches the end may have been cut
    by the length limit, so it is dropped."""
    text = prompt + response
    return [m.group().lower() for m in WORD.finditer(text) if len(prompt) < m.end() < len(text)]

def gold_reward(prompt, response, gold=GOLD):
    words = complete_words(prompt, response)
    if not words:
        return 0.0
    pos, neg = set(gold["positive"]), set(gold["negative"])
    n_pos = len({w for w in words if w in pos})             # distinct positive words
    n_neg = sum(w in neg for w in words)                    # every negative occurrence
    f_list = sum(w in pos or w in neg for w in words) / len(words)
    return float(min(n_pos, gold["cap"]) - n_neg
                 - gold["crowd_weight"] * max(0.0, f_list - gold["crowd_threshold"]))
# ---- End of the restated interface.


def bt_prob(g_a, g_b, tau_label):
    """P(a preferred to b) for a Bradley-Terry rater with label temperature tau_label."""
    return 1.0 / (1.0 + math.exp(-(g_a - g_b) / tau_label))


SUBJECTS = ["My mother", "My father", "My sister", "My brother", "My friend", "My boss",
            "Our neighbor", "The teacher", "The doctor", "The manager", "The waiter",
            "The student", "The driver", "The coach", "The nurse", "The chef"]
FRAMES = {
    "train": ["looked at the results and said", "opened the old letter and felt",
              "walked into the room and saw", "tasted the soup and told us",
              "read the review twice and thought", "came home late and found the",
              "listened to the new song and", "looked out the window and said",
              "heard the news this morning and", "finished the long day and felt"],
    "heldout": ["watched the game last night and", "visited the old house and said"],
    "eval": ["tried the new restaurant and said", "read the long email and felt",
             "saw the final bill and said", "spent the whole weekend at the"],
}
SUBJECT_LEN, FRAME_LEN = 2, 6

# Hand-written responses that the reference policy rarely writes, scored after
# prompts["heldout"][0] in Lab 9's preview cell. "fill" items repeat their text until
# exactly RESPONSE_LEN tokens; the others must fit in RESPONSE_LEN tokens.
PREVIEW = [
    ("one positive word repeated", " great", True),
    ("distinct positive words, comma-separated",
     " good, great, happy, wonderful, beautiful, nice, amazing, perfect, excellent.", False),
    ("'I love it!' repeated", " I love it!", True),
    ("ordinary reply, one positive word",
     " it was a good day, and we all went home early to rest before dinner.", False),
    ("ordinary reply, two negative words",
     " it was a bad day, and the long wait made everyone feel sick and tired.", False),
]

# Acceptance criteria of the brief ("Measure on 2,000 reference samples").
MAX_TIE_SHARE = 0.6
MAX_CAP_SHARE = 0.01
MAX_CROWD_SHARE = 0.01
MIN_ORACLE_AT_TAU = 0.6


# ----------------------------------------------------------------------------- policies


class Gpt2Policy:
    """The reference policy: GPT-2 from the Hub, float32, eval() mode."""

    is_stand_in = False

    def __init__(self, model_id, revision, device, model=None, tokenizer=None):
        if model is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
            model = AutoModelForCausalLM.from_pretrained(
                model_id, revision=revision, dtype=torch.float32)
            revision = resolve_revision(model_id, revision, model)
        self.model = model.to(device).float().eval()
        self.tokenizer = tokenizer
        self.model_id, self.revision, self.device = model_id, revision, device

    def check_vocabulary(self):
        tok = self.tokenizer
        assert tok.eos_token_id == EOS_ID == len(tok) - 1, (tok.eos_token_id, len(tok))
        assert self.model.config.vocab_size == EOS_ID + 1, self.model.config.vocab_size

    def encode(self, text):
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids):
        return self.tokenizer.decode(ids, clean_up_tokenization_spaces=False)

    def decode_prompt(self, ids):
        return self.tokenizer.decode(ids)

    @torch.no_grad()
    def next_logits(self, ids, past=None):
        """Logits of the next token without end-of-text, (B, EOS_ID), and the new cache."""
        out = self.model(input_ids=ids, past_key_values=past, use_cache=True)
        return out.logits[:, -1, :EOS_ID].float(), out.past_key_values

    @torch.no_grad()
    def sample(self, prompt_ids, generator):
        """(B, PROMPT_LEN) prompt ids -> (B, RESPONSE_LEN) response ids, temperature 1."""
        logits, past = self.next_logits(prompt_ids.to(self.device))
        steps = []
        for t in range(RESPONSE_LEN):
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
            steps.append(nxt)
            if t + 1 < RESPONSE_LEN:
                logits, past = self.next_logits(nxt, past)
        return torch.cat(steps, dim=1).cpu()

    @torch.no_grad()
    def check_cache(self, prompt_ids, generator):
        """The cached logits of sample() must equal a full forward pass over the sequence."""
        prompt_ids = prompt_ids.to(self.device)
        logits, past = self.next_logits(prompt_ids)
        seq, cached = prompt_ids, [logits]
        for _ in range(RESPONSE_LEN - 1):
            nxt = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator)
            seq = torch.cat([seq, nxt], dim=1)
            logits, past = self.next_logits(nxt, past)
            cached.append(logits)
        full = self.model(input_ids=seq).logits[:, PROMPT_LEN - 1:, :EOS_ID].float()
        diff = (torch.stack(cached, dim=1) - full).abs().max().item()
        assert diff < 1e-3, f"cached and uncached logits differ by {diff}"
        return diff

    def describe(self):
        import transformers

        device = (torch.cuda.get_device_name(0) if str(self.device).startswith("cuda")
                  else f"CPU ({platform.processor() or platform.machine()})")
        return {"device": device, "torch": torch.__version__,
                "transformers": transformers.__version__, "python": platform.python_version()}


def resolve_revision(model_id, revision, model):
    """The commit hash of the loaded checkpoint, so that it can be pinned."""
    try:
        from huggingface_hub import model_info

        return model_info(model_id, revision=revision).sha
    except Exception:  # fall back to what transformers recorded
        commit = getattr(model.config, "_commit_hash", None)
        if commit:
            return commit
        raise


class StandInPolicy:
    """TEST ONLY. A word-level stand-in for GPT-2 so the pipeline runs without the Hub.

    Every word (with its leading space) and every punctuation mark is one token, with an
    id from a locally built vocabulary inside GPT-2's id range. Responses are drawn
    token by token from a fixed distribution. Nothing about it resembles GPT-2's text
    statistics; files built with it are not the dataset.
    """

    is_stand_in = True
    FILLER = ("the a and to of it was I we they he she that in on for with at but so not "
              "then all just very really one day time home back again here there out up "
              "about what when this my our his her their you me us them some more much "
              "had have has did said told thought felt saw found went came made got took "
              "still never always only even also well now later today morning night room "
              "door table window letter soup song news results review house game bill "
              "weekend dinner food work school car street city friend family people kids "
              "little old new long last first big small next other same few many quiet "
              "slowly quickly finally almost maybe perhaps because while after before "
              "into over through around away down off like than as if or no yes").split()

    def __init__(self, seed=0):
        words = sorted(set(self.FILLER) | set(GOLD["positive"]) | set(GOLD["negative"])
                       | {w for s in SUBJECTS for w in s.split()}
                       | {w for fs in FRAMES.values() for f in fs for w in f.split()}
                       | {w.strip(",.!") for _, r, _ in PREVIEW for w in r.split()})
        tokens = [" " + w for w in words] + [w for w in words] + [",", ".", "!"]
        self.vocab = {tok: 1000 + 3 * i for i, tok in enumerate(tokens)}  # ids < EOS_ID
        self.inverse = {i: tok for tok, i in self.vocab.items()}
        assert max(self.vocab.values()) < EOS_ID
        # The response distribution: filler with Zipf weights, list words rare, punctuation.
        rng = random.Random(seed)
        filler = [" " + w for w in self.FILLER]
        rng.shuffle(filler)
        weights = {tok: 1.0 / (rank + 5) for rank, tok in enumerate(filler)}
        total = sum(weights.values())
        weights = {tok: 0.78 * w / total for tok, w in weights.items()}
        for i, w in enumerate(GOLD["positive"] + GOLD["negative"]):
            weights[" " + w] = 0.02 / (1 + i % 5) / 9.12  # each list about 0.9% of tokens
        for tok, w in ((",", 0.07), (".", 0.06), ("!", 0.01)):
            weights[tok] = w
        self.response_tokens = list(weights)
        p = torch.tensor([weights[t] for t in self.response_tokens], dtype=torch.float64)
        self.response_probs = p / p.sum()
        self.response_ids = torch.tensor([self.vocab[t] for t in self.response_tokens])
        self.model_id, self.revision, self.device = "STAND-IN (not GPT-2)", "none", "cpu"

    def check_vocabulary(self):
        pass

    def encode(self, text):
        return [self.vocab[tok] for tok in re.findall(r" ?[A-Za-z']+|[,.!]", text)]

    def decode(self, ids):
        return "".join(self.inverse[i] for i in ids)

    decode_prompt = decode

    def sample(self, prompt_ids, generator):
        n = prompt_ids.shape[0] * RESPONSE_LEN
        draws = torch.multinomial(self.response_probs, n, replacement=True, generator=generator)
        return self.response_ids[draws].reshape(prompt_ids.shape[0], RESPONSE_LEN)

    def check_cache(self, prompt_ids, generator):
        return 0.0

    def describe(self):
        return {"device": "cpu", "torch": torch.__version__, "transformers": None,
                "python": platform.python_version(), "stand_in": True}


# ----------------------------------------------------------------------------- building


def words_in_lists(text):
    lists = set(GOLD["positive"]) | set(GOLD["negative"])
    return sorted({w.lower() for w in WORD.findall(text)} & lists)


def build_prompts(policy):
    """{split: [{"text", "ids"}]}; asserts every length, decoding and list-word rule."""
    prompts = {}
    for subject in SUBJECTS:
        assert len(policy.encode(subject)) == SUBJECT_LEN, (subject, policy.encode(subject))
    for split, frames in FRAMES.items():
        prompts[split] = []
        for frame in frames:
            assert len(policy.encode(" " + frame)) == FRAME_LEN, (frame, policy.encode(" " + frame))
            for subject in SUBJECTS:
                text = f"{subject} {frame}"
                ids = policy.encode(text)
                assert len(ids) == PROMPT_LEN, (text, ids)
                assert EOS_ID not in ids, text
                assert policy.decode_prompt(ids) == text == policy.decode(ids), text
                assert not words_in_lists(text), (text, words_in_lists(text))
                prompts[split].append({"text": text, "ids": ids})
    assert [len(prompts[s]) for s in ("train", "heldout", "eval")] == [160, 32, 64]
    return prompts


def build_preview(policy):
    items = []
    for name, text, fill in PREVIEW:
        ids = policy.encode(text)
        if fill:
            ids = (ids * RESPONSE_LEN)[:RESPONSE_LEN]
        assert 0 < len(ids) <= RESPONSE_LEN, (name, len(ids))
        items.append({"name": name, "response": policy.decode(ids), "response_ids": ids})
    return items


def sample_pairs(policy, prompts, n_pairs, split, seed, generator, batch_pairs):
    """n_pairs dicts with prompt, two responses (ids and text) and both gold scores."""
    order = list(range(len(prompts)))
    random.Random(f"{seed}-{split}").shuffle(order)
    plan = [prompts[order[i % len(order)]] for i in range(n_pairs)]
    pairs = []
    for start in range(0, n_pairs, batch_pairs):
        chunk = plan[start:start + batch_pairs]
        rows = torch.tensor([p["ids"] for p in chunk for _ in range(2)])  # (2B, 8), a and b
        responses = policy.sample(rows, generator)  # (2B, 24)
        assert responses.shape == (len(rows), RESPONSE_LEN), responses.shape
        assert not (responses == EOS_ID).any(), "end-of-text was sampled"
        assert int(responses.min()) >= 0 and int(responses.max()) < EOS_ID
        for k, prompt in enumerate(chunk):
            a, b = responses[2 * k].tolist(), responses[2 * k + 1].tolist()
            text_a, text_b = policy.decode(a), policy.decode(b)
            pairs.append({"prompt": prompt["text"], "prompt_ids": prompt["ids"],
                          "a": text_a, "a_ids": a, "b": text_b, "b_ids": b,
                          "gold_a": gold_reward(prompt["text"], text_a),
                          "gold_b": gold_reward(prompt["text"], text_b), "split": split})
        print(f"  {split}: {len(pairs):,} / {n_pairs:,} pairs", flush=True)
    return pairs


def label_pairs(pairs, seed, tau_label):
    """Bradley-Terry rater: a is chosen with probability bt_prob(g_a, g_b, tau_label)."""
    rng = random.Random(seed + 1)
    out = []
    for p in pairs:
        a_wins = rng.random() < bt_prob(p["gold_a"], p["gold_b"], tau_label)
        win, lose = ("a", "b") if a_wins else ("b", "a")
        out.append({"prompt": p["prompt"], "prompt_ids": p["prompt_ids"],
                    "chosen": p[win], "rejected": p[lose],
                    "chosen_ids": p[win + "_ids"], "rejected_ids": p[lose + "_ids"],
                    "gold_chosen": p["gold_" + win], "gold_rejected": p["gold_" + lose],
                    "split": p["split"]})
    return out


# ----------------------------------------------------------------------------- statistics


def components(prompt, response):
    words = complete_words(prompt, response)
    pos, neg = set(GOLD["positive"]), set(GOLD["negative"])
    counts = Counter(w for w in words if w in pos)
    n_list = sum(w in pos or w in neg for w in words)
    return {"n_words": len(words), "n_pos": len(counts),
            "repeated_pos": any(c >= 2 for c in counts.values()),
            "f_list": n_list / len(words) if words else 0.0,
            "list_words": [w for w in words if w in pos or w in neg]}


def oracle_accuracy(g_a, g_b, tau_label):
    """Mean of max(p, 1 - p) over pairs: the best possible pairwise accuracy (eq. oracle)."""
    probs = [bt_prob(a, b, tau_label) for a, b in zip(g_a, g_b, strict=True)]
    return sum(max(p, 1 - p) for p in probs) / len(probs)


def reference_statistics(pairs):
    """The table of the brief, on every sampled response of `pairs` (unlabeled pairs)."""
    samples = [(p["prompt"], p[k]) for p in pairs for k in ("a", "b")]
    gold = [p[g] for p in pairs for g in ("gold_a", "gold_b")]
    comps = [components(prompt, resp) for prompt, resp in samples]
    n = len(samples)
    word_counts = Counter(w for c in comps for w in c["list_words"])
    g_a, g_b = [p["gold_a"] for p in pairs], [p["gold_b"] for p in pairs]
    return {
        "samples": n,
        "pairs": len(pairs),
        "share_gold_nonzero": sum(g != 0 for g in gold) / n,
        "tie_share": sum(a == b for a, b in zip(g_a, g_b, strict=True)) / len(pairs),
        "cap_active_share": sum(c["n_pos"] > GOLD["cap"] for c in comps) / n,
        "crowding_active_share": sum(c["f_list"] > GOLD["crowd_threshold"] for c in comps) / n,
        "repeated_positive_share": sum(c["repeated_pos"] for c in comps) / n,
        "mean_complete_words": sum(c["n_words"] for c in comps) / n,
        "mean_gold": sum(gold) / n,
        "gold_values": dict(sorted(Counter(round(g, 4) for g in gold).most_common(12))),
        "list_word_counts": {w: word_counts.get(w, 0)
                             for w in GOLD["positive"] + GOLD["negative"]},
        "oracle_accuracy": {str(t): oracle_accuracy(g_a, g_b, t) for t in (0.25, 0.5, 1.0)},
    }


def acceptance(stats):
    """[(criterion, value, passed)] for the brief's acceptance criteria."""
    tau = str(float(GOLD["tau_label"]))
    oracle = stats["oracle_accuracy"].get(tau)
    if oracle is None:
        oracle = stats["oracle_accuracy"]["0.5"]
    return [
        (f"tie share <= {MAX_TIE_SHARE}", stats["tie_share"], stats["tie_share"] <= MAX_TIE_SHARE),
        (f"cap active (n_pos >= {GOLD['cap'] + 1}) < {MAX_CAP_SHARE:.0%}",
         stats["cap_active_share"], stats["cap_active_share"] < MAX_CAP_SHARE),
        (f"crowding active (f_list > {GOLD['crowd_threshold']}) < {MAX_CROWD_SHARE:.0%}",
         stats["crowding_active_share"], stats["crowding_active_share"] < MAX_CROWD_SHARE),
        (f"Acc* at tau_label {GOLD['tau_label']} >= {MIN_ORACLE_AT_TAU}", oracle,
         oracle >= MIN_ORACLE_AT_TAU),
    ]


def print_statistics(stats, checks):
    print(f"\nReference-sample statistics ({stats['samples']:,} samples, {stats['pairs']:,} "
          "same-prompt pairs)")
    rows = [("share with g != 0", stats["share_gold_nonzero"]),
            ("tie share of same-prompt pairs", stats["tie_share"]),
            ("cap active (n_pos >= 4)", stats["cap_active_share"]),
            ("crowding active (f_list > threshold)", stats["crowding_active_share"]),
            ("some positive word occurs twice or more", stats["repeated_positive_share"]),
            ("mean complete words per response", stats["mean_complete_words"]),
            ("mean gold score", stats["mean_gold"])]
    rows += [(f"Acc* at tau_label {t}", v) for t, v in stats["oracle_accuracy"].items()]
    for name, value in rows:
        print(f"  {name:<42} {value:8.4f}")
    print("  most frequent gold values:", stats["gold_values"])
    print("  list-word counts:")
    counts = stats["list_word_counts"]
    for kind in ("positive", "negative"):
        print(f"    {kind}: " + ", ".join(f"{w} {counts[w]}" for w in GOLD[kind]))
    print("\nAcceptance criteria (briefs/09-preference-learning.md)")
    for name, value, passed in checks:
        print(f"  {'PASS' if passed else 'FAIL'}  {name}: {value:.4f}")


# ----------------------------------------------------------------------------- files


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_preferences(path, labeled):
    lines = [json.dumps(p, separators=(",", ":")) for p in labeled]
    raw = ("\n".join(lines) + "\n").encode("utf-8")
    buffer = io.BytesIO()
    with gzip.GzipFile(filename="", mode="wb", fileobj=buffer, mtime=0, compresslevel=9) as f:
        f.write(raw)  # mtime=0 and no file name: the same content gives the same bytes
    Path(path).write_bytes(buffer.getvalue())


def verify_files(prompts_path, prefs_path, policy):
    """Re-read both files and re-check every invariant of the interface."""
    doc = json.loads(Path(prompts_path).read_text(encoding="utf-8"))
    assert doc["format"] == FORMAT_PROMPTS and doc["version"] == 1
    assert doc["gold"] == GOLD
    split_prompts = {s: {p["text"]: p["ids"] for p in doc["prompts"][s]} for s in FRAMES}
    for items in doc["prompts"].values():
        for p in items:
            assert len(p["ids"]) == PROMPT_LEN and policy.decode_prompt(p["ids"]) == p["text"]
    pairs = [json.loads(line) for line in
             gzip.decompress(Path(prefs_path).read_bytes()).decode("utf-8").splitlines()]
    fields = {"prompt", "prompt_ids", "chosen", "rejected", "chosen_ids", "rejected_ids",
              "gold_chosen", "gold_rejected", "split"}
    counts = Counter()
    for p in pairs:
        assert set(p) == fields, set(p) ^ fields
        assert split_prompts[p["split"]][p["prompt"]] == p["prompt_ids"]
        for k in ("chosen", "rejected"):
            ids = p[k + "_ids"]
            assert len(ids) == RESPONSE_LEN and EOS_ID not in ids
            assert policy.decode(ids) == p[k]
            assert gold_reward(p["prompt"], p[k]) == p["gold_" + k]
        counts[p["split"]] += 1
    assert doc["build"]["preferences_sha256"] == sha256(prefs_path)
    return counts


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", type=Path, default=DATA, help="output directory (default data/)")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--revision", default=None,
                        help="model revision (default: main, resolved to a commit and recorded)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--train-pairs", type=int, default=TRAIN_PAIRS)
    parser.add_argument("--heldout-pairs", type=int, default=HELDOUT_PAIRS)
    parser.add_argument("--batch-pairs", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--stats-only", action="store_true",
                        help=f"sample {STATS_ONLY_PAIRS:,} train pairs, print statistics, "
                             "write nothing")
    parser.add_argument("--allow-failing-criteria", action="store_true",
                        help="write the files even if an acceptance criterion fails")
    parser.add_argument("--offline-tiny", action="store_true",
                        help="TEST ONLY: word-level stand-in instead of GPT-2; refuses data/")
    args = parser.parse_args(argv)
    started = time.time()
    torch.manual_seed(args.seed)

    if args.offline_tiny:
        if args.out.resolve() == DATA.resolve():
            sys.exit("--offline-tiny writes stand-in files; give an --out outside data/")
        policy = StandInPolicy(args.seed)
        args.device = "cpu"
    else:
        policy = Gpt2Policy(args.model, args.revision, args.device)
    policy.check_vocabulary()
    print(f"Policy: {policy.model_id} @ {policy.revision} on {args.device}")

    prompts = build_prompts(policy)
    preview = build_preview(policy)
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    diff = policy.check_cache(torch.tensor([prompts["train"][0]["ids"]] * 2), generator)
    print(f"Prompts: 160 / 32 / 64 of {PROMPT_LEN} tokens; cached-sampling check {diff:.2e}")

    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    if args.stats_only:
        pairs = sample_pairs(policy, prompts["train"], STATS_ONLY_PAIRS, "train", args.seed,
                             generator, args.batch_pairs)
        stats = reference_statistics(pairs)
        checks = acceptance(stats)
        print_statistics(stats, checks)
        print(f"\nStatistics only; nothing written ({time.time() - started:.0f} s).")
        return 0 if all(passed for *_, passed in checks) else 1

    pairs = []
    for split, n in (("train", args.train_pairs), ("heldout", args.heldout_pairs)):
        pairs += sample_pairs(policy, prompts[split], n, split, args.seed, generator,
                              args.batch_pairs)
    sampling_seconds = time.time() - started
    stats = reference_statistics(pairs)
    checks = acceptance(stats)
    print_statistics(stats, checks)
    if not all(passed for *_, passed in checks) and not args.allow_failing_criteria:
        print("\nAn acceptance criterion failed: nothing written. Change only the lists, "
              "crowd_threshold, the pair count or tau_label (see the brief), in this script "
              "and in both notebooks, then rebuild.")
        return 1

    labeled = label_pairs(pairs, args.seed, GOLD["tau_label"])
    args.out.mkdir(parents=True, exist_ok=True)
    prefs_path = args.out / "lab09_preferences.jsonl.gz"
    prompts_path = args.out / "lab09_prompts.json"
    write_preferences(prefs_path, labeled)
    doc = {
        "format": FORMAT_PROMPTS, "version": 1,
        "model": policy.model_id, "revision": policy.revision,
        "subjects": SUBJECTS, "frames": FRAMES, "prompts": prompts, "preview": preview,
        "gold": GOLD,
        "build": {
            "script": "data/build_lab09_preferences.py", "seed": args.seed,
            "label_seed": args.seed + 1, "pairs": {"train": args.train_pairs,
                                                    "heldout": args.heldout_pairs},
            "sampling": "float32, eval(), temperature 1, no truncation, logits[..., :EOS_ID]",
            "prompt_len": PROMPT_LEN, "response_len": RESPONSE_LEN, "eos_id": EOS_ID,
            **policy.describe(),
            "sampling_seconds": round(sampling_seconds, 1),
            "preferences_sha256": sha256(prefs_path),
            "stand_in": policy.is_stand_in,
        },
        "statistics": stats,
    }
    prompts_path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    counts = verify_files(prompts_path, prefs_path, policy)
    print(f"\nVerified: {counts['train']:,} train and {counts['heldout']:,} held-out pairs.")
    for path in (prompts_path, prefs_path):
        print(f"  {path.name}: {path.stat().st_size:,} bytes, sha256 {sha256(path)}")
    if not policy.is_stand_in:
        print(f"\nPin models.causal_lm_revision: \"{policy.revision}\"")
        print("Paste both hashes into LAB09_FILES in notebooks/09-preference-learning.ipynb "
              "and into the datasets entries of _variables.yml and data/README.md.")
    print(f"Done in {time.time() - started:.0f} s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
