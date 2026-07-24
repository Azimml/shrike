---
name: Feature request
about: Suggest an engine feature or improvement
title: "[feature] "
labels: enhancement
---

## Problem

What are you trying to do that shrike does not support today?

## Proposed solution

What you'd like to see, and roughly how it might fit the existing engine
(scheduler / block manager / model runner / server).

## Scope check

shrike is deliberately single-GPU, single-model, bf16. Tensor/pipeline
parallelism, quantization, and LoRA are out of scope (see the README's
"Limitations" and "Future work"). Does this request fit that scope? If it
extends it, why is it worth the added complexity?

## Alternatives considered

Other approaches you weighed, and why this one is preferable.
