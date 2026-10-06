<!--
Hand-written, not generated: edit the entry check here. scripts/gen_tables.py owns
everything under _includes/; this file is kept apart from those on purpose.
Included by prepare.qmd (the Before Day 1 page) and assessments.md, which both sit at
the site root, so relative links resolve the same way from either page. The scoring
rule and the remediation for each area are on prepare.qmd, section "Read your score".
-->

Fifteen questions in five areas, three per area, matching the prerequisites on the [home page](index.qmd#prerequisites). Allow about 15 minutes. Work on paper or in a notebook, without searching the web, and write each answer down before you open its folded key.

### Python

**E1.** What does this expression evaluate to?

```python
[x * x for x in range(5) if x % 2 == 0]
```

::: {.callout-tip collapse="true" title="Answer"}
`[0, 4, 16]`. The comprehension keeps 0, 2 and 4 and squares them.
:::

**E2.** What does this print?

```python
class Counter:
    def __init__(self):
        self.n = 0

    def add(self, k=1):
        self.n += k
        return self

c = Counter().add().add(3)
print(c.n)
```

::: {.callout-tip collapse="true" title="Answer"}
`4`. Each call to `add` changes the same object and returns it, so the calls chain: 0 + 1 + 3.
:::

**E3.** A loop over rows read from a CSV file stops with this traceback. What is wrong, and what is the fix?

```text
  File "count.py", line 7, in <module>
    total = total + row["count"]
TypeError: unsupported operand type(s) for +: 'int' and 'str'
```

::: {.callout-tip collapse="true" title="Answer"}
`row["count"]` is a string (the CSV reader returns text), and Python will not add a string to an integer. Convert it: `total = total + int(row["count"])`. The last line of a traceback names the error; the line above it shows where it happened.
:::

### NumPy

**E4.** `A` has shape `(3, 4)` and `b` has shape `(4,)`. What are the shapes of `A + b` and `A @ b`?

::: {.callout-tip collapse="true" title="Answer"}
`A + b` is `(3, 4)`: `b` is broadcast across the three rows. `A @ b` is `(3,)`: a matrix-vector product.
:::

**E5.** With `x = np.arange(12).reshape(3, 4)`, what are `x[1, :]` and `x[:, -1]`?

::: {.callout-tip collapse="true" title="Answer"}
`x[1, :]` is `[4, 5, 6, 7]` (the second row). `x[:, -1]` is `[3, 7, 11]` (the last column).
:::

**E6.** `X` has shape `(N, d)`. Write one line that divides each row by its Euclidean length, and say why `keepdims=True` matters.

::: {.callout-tip collapse="true" title="Answer"}
`X / np.linalg.norm(X, axis=1, keepdims=True)`. With `keepdims=True` the norms have shape `(N, 1)`, which broadcasts across the `d` columns. Without it they have shape `(N,)`, which NumPy tries to align with the columns and fails (or, if `N == d`, silently divides the wrong way).
:::

### Machine-learning basics

**E7.** Why are hyperparameters chosen on a validation split rather than on the test split?

::: {.callout-tip collapse="true" title="Answer"}
Choosing on the test split makes the test score an optimistic estimate: the setting was picked because it did well on those examples. The test split is used once, after every choice is fixed. The workshop follows this rule in every lab.
:::

**E8.** Logistic regression predicts $p = \sigma(w^\top x + b)$ for a label $y \in \{0, 1\}$. Write the loss for one example, and its derivative with respect to the logit $z = w^\top x + b$.

::: {.callout-tip collapse="true" title="Answer"}
The cross-entropy (log loss) $\mathcal{L} = -\big[y \log p + (1 - y)\log(1 - p)\big]$. Its derivative with respect to $z$ is $p - y$: prediction minus target. The same form returns in Modules 1, 2 and 6.
:::

**E9.** Gradient descent updates $\theta \leftarrow \theta - \eta \nabla_\theta \mathcal{L}$. What do you expect to see if the learning rate $\eta$ is far too large?

::: {.callout-tip collapse="true" title="Answer"}
The loss oscillates or grows instead of falling, because each step overshoots the minimum. It can also become `nan`.
:::

### PyTorch

**E10.** Put these five lines of one training step in order.

```python
optimizer.step()
loss = F.cross_entropy(logits, y)
optimizer.zero_grad()
loss.backward()
logits = model(x)
```

::: {.callout-tip collapse="true" title="Answer"}
`optimizer.zero_grad()`, `logits = model(x)`, `loss = F.cross_entropy(logits, y)`, `loss.backward()`, `optimizer.step()`. Putting `zero_grad()` right after `step()` instead is also correct; what matters is that gradients are cleared before the next `backward()`, because PyTorch adds to them.
:::

**E11.** What does `model.eval()` change, and what does `with torch.no_grad():` change?

::: {.callout-tip collapse="true" title="Answer"}
`model.eval()` switches layers whose behavior differs between training and evaluation, such as dropout (turned off) and batch normalization (uses stored statistics). It does not stop gradients. `torch.no_grad()` stops autograd from recording operations, which saves memory and time when no gradient is needed. Several labs rely on the difference: Lab 5's leak test runs in `eval()` mode.
:::

**E12.** `layer = nn.Linear(4, 3)` is applied to a tensor of shape `(8, 4)`. What is the output shape, and how many parameters does the layer have?

::: {.callout-tip collapse="true" title="Answer"}
Output `(8, 3)`. Parameters: a $3 \times 4$ weight matrix and 3 biases, $12 + 3 = 15$.
:::

### Mathematics

**E13.** Let $y = \sigma(z)$ with $z = wx$. What is $\dfrac{dy}{dw}$?

::: {.callout-tip collapse="true" title="Answer"}
By the chain rule, $\dfrac{dy}{dw} = \sigma(z)\big(1 - \sigma(z)\big)\, x$, using $\sigma'(z) = \sigma(z)(1 - \sigma(z))$.
:::

**E14.** $P(A) = 0.3$ and $P(B \mid A) = 0.5$. What is $P(A \text{ and } B)$? And what is the expected value of one roll of a fair six-sided die?

::: {.callout-tip collapse="true" title="Answer"}
$P(A \text{ and } B) = P(B \mid A)\,P(A) = 0.15$. The expected value is $(1 + 2 + \dots + 6)/6 = 3.5$.
:::

**E15.** A model predicts the distribution $(0.7, 0.2, 0.1)$ over three classes and the true class is the first. What is the cross-entropy, in nats? What is it for a uniform prediction over four classes? And what is the cosine similarity of the vectors $(1, 0)$ and $(1, 1)$?

::: {.callout-tip collapse="true" title="Answer"}
$-\ln 0.7 \approx 0.357$ nats. For a uniform prediction over four classes, $\ln 4 \approx 1.386$ nats: the starting loss of Lab 6's four-class classifier. The cosine is $1 / \sqrt{2} \approx 0.707$.
:::
