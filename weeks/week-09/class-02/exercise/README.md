# W9C2 Activity: Spend the Compute Budget

Teams of four, whiteboards, no laptops. Nothing to install and nothing to
submit. Write your numbers on the board with units.

## What you have

A training budget of **1.2 x 10^22 FLOPs**, and two relationships:

| | |
|---|---|
| cost of a training run | `C = 6ND` |
| compute-optimal split | `D = 20N` |

`N` is parameters, `D` is training tokens, `C` is FLOPs.

## Round 1 (5 min), everybody

Spend the whole budget with no other constraints. What are `N` and `D`?

## Round 2 (15 min), your team's constraint

Your team has one of these. The budget does not change.

**Team A, the data runs out.** You hold 80B tokens of licensed text and may not
repeat any of it.

**Team B, it has to be served.** Inference must run on one 16GB GPU in bf16
(2 bytes per parameter), with 4GB kept for activations and the KV cache.

**Team C, inference dwarfs training.** The model will serve 5 x 10^12 tokens
over its life, at roughly `2N` FLOPs per token. Training cost and inference cost
come out of the same account.

Report three things: your `N`, your `D`, and the tokens-per-parameter ratio
`D/N`. Then say in one sentence what your constraint did to the Round 1 answer.

## Round 3 (10 min), all together

One number from each team on the board, then we argue about which of them a
real lab would actually ship.
