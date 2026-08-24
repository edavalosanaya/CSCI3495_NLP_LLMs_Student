---
marp: true
title: "W4C1: Neural Networks for NLP"
paginate: true
---

# Week 4, Class 1: Neural Networks for NLP

## Feedforward nets, training & backprop

<!-- layout: title -->
<!-- img: visuals/assets/cajal-cortex-1899.png -->
<!-- caption: Cajal drew the cortex by hand in 1899: layers of tiny units, all wired to each other. Ask the class what survives when we shrink one of those cells down to w·x + b. -->
<!-- source: Santiago Ramon y Cajal (1899), detail, Wikimedia Commons, public domain -->

---

# Today's roadmap

<!-- img: visuals/agenda.png -->
<!-- caption: From linear models to MLPs, how nets learn, then build an MLP text classifier in PyTorch. -->
<!-- source: tiles: SLP3 (Jan 2026 draft) Figs. 6.5, 6.8, 6.17; whiteboard photo Unsplash License; original figure + seeded example run -->

---

# Foundations

<!-- layout: section -->

---

# Last week's classifier is one straight cut

<!-- img: visuals/linear-vs-nonlinear.png -->
<!-- caption: Logistic regression's boundary w·x + b = 0 is a single straight line, however you train it. Search every orientation and the best line still gets 8 of these 24 points wrong; one closed curve gets none. The next slide shows the smallest version of the same problem, XOR. -->
<!-- source: original figure -->

---

# Why go neural?

<!-- img: visuals/assets/slp3-fig-6-5.png -->
<!-- caption: The cleanest counterexample: XOR. AND and OR fall to one line; for XOR no straight cut exists (this sank the perceptron, Minsky & Papert 1969). Text is full of XORs: "not bad" flips the label of "bad". -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 6.5 -->

---

# The fix: transform the space first

<!-- img: visuals/assets/slp3-fig-6-7.png -->
<!-- caption: A tiny hidden layer maps x-space into a new h-space where (0,1) and (1,0) land on the same point, and XOR becomes one straight cut again. Nonlinear transform, then a linear cut: that is the whole trick. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 6.7 (after Goodfellow et al. 2016) -->

---

# The neuron

<!-- img: visuals/neuron.png -->
<!-- caption: A unit computes z = w·x + b, then a nonlinear activation a = g(z). Logistic regression is one neuron with g = sigmoid. -->
<!-- source: original figure -->

---

# One artificial neuron is almost nothing. Why do millions of them learn so much?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-network.jpg -->
<!-- caption: Hold that thought as we stack neurons into layers. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1507413245164-6160d8298b31 -->

---

# Stacking neurons into an MLP

<!-- img: visuals/mlp-anatomy.png -->
<!-- caption: A 2-layer feedforward net and its math, side by side: W and U are the weight matrices, the gray +1 node carries the biases b, and each layer is one line of algebra. The worked softmax shows score 3.2 grabbing 74% of the probability. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 6.8, Eqs. 6.8-6.11 -->

---

# Activations: why nonlinearity matters

<!-- img: visuals/activations.png -->
<!-- caption: Sigmoid, tanh, and ReLU. Without a nonlinearity, stacked linear layers collapse to one linear layer. -->
<!-- source: original figure -->

---

# The output layer

<!-- img: visuals/softmax.png -->
<!-- caption: Softmax turns K logits into a probability distribution; the largest probability is the prediction. -->
<!-- source: original figure -->

---

# The whole network, in three equations

<!-- img: visuals/forward-eqs.png -->
<!-- caption: Bookmark this slide: every feedforward classifier this term is these three lines. The book writes sigma for ANY activation (sigmoid, tanh, ReLU). -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Eq. 6.12 -->

---

# How nets learn

<!-- layout: section -->

---

# The loss: how wrong are we?

<!-- img: visuals/cross-entropy.png -->
<!-- caption: Cross-entropy is small when the true class's predicted probability is high, large when it is low. -->
<!-- source: original figure -->

---

# Gradient descent

<!-- img: visuals/gradient-descent.gif -->
<!-- caption: The gradient points uphill; step the opposite way. The learning rate η sets the step size, and the steps shrink on their own as the slope flattens. -->
<!-- source: original animation -->

---

# One minute of calculus: the chain rule

<!-- img: visuals/chain-rule.png -->
<!-- caption: Nested functions change at multiplied rates. Backprop needs nothing deeper than this. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Eqs. 6.31-6.32 -->

---

# Computation graphs: the forward pass

<!-- img: visuals/assets/slp3-fig-6-15.png -->
<!-- caption: Any formula becomes a graph of tiny steps. L(a,b,c) = c(a + 2b) breaks into d = 2b, then e = a + d, then L = ce; with a = 3, b = 1, c = -2 the values flow left to right and L = -10. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 6.15 -->

---

# One node's job: the chain rule

<!-- img: visuals/assets/slp3-fig-6-16.png -->
<!-- caption: Each node multiplies the gradient arriving from upstream by its own local gradient and passes the product downstream: dL/dd = (dL/de)(de/dd). That is ALL backpropagation ever does, node by node. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 6.16 -->

---

# Working out all three gradients

<!-- img: visuals/backprop-worked.png -->
<!-- caption: Five local gradients, then multiply along each path back from L. Keep these numbers in mind: the next slide is the same computation drawn on the graph. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Eqs. 6.33-6.34 -->

---

# Backpropagation: the backward pass

<!-- img: visuals/assets/slp3-fig-6-17.png -->
<!-- caption: The same graph, run right to left. Follow any red value: it is the upstream gradient times a local one, exactly the rule from the previous slide, until every input knows dL/d(itself). -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 6.17 -->

---

# Now scale that to a real network

<!-- img: visuals/assets/slp3-fig-6-18.png -->
<!-- caption: The graph for a 2-input, 2-hidden-unit net, the same tiny MLP you will trace by hand today. Already this involved, and GPT-class models have billions of nodes. Nobody does this by hand: loss.backward() builds and walks the graph for you. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 6.18 -->

---

# The math behind that graph (for reference)

<!-- img: visuals/backprop-net-eqs.png -->
<!-- caption: Don't memorize the derivation; keep the takeaway: at the output, the gradient is simply prediction minus truth. The book walks the rest back node by node (Section 6.6). -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Eqs. 6.35-6.43 -->

---

# The training loop

<!-- img: visuals/training-loop.png -->
<!-- caption: zero_grad → forward → loss → backward → step. This shape recurs all term, even for Transformers. -->
<!-- source: weeks/week-04/class-01/exercise/mlp_classifier.py -->

---

# Neural NLP

<!-- layout: section -->

---

# From text to a vector: pool the embeddings

<!-- img: visuals/assets/slp3-fig-6-13.png -->
<!-- caption: Each word's one-hot picks its row of the shared embedding matrix E; pooling (mean or sum) squeezes the N embeddings into one input vector. Fast, cheap, and a surprisingly strong baseline (Iyyer et al. 2015). -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 6.13 -->

---

# What pooling throws away

<!-- img: visuals/order-lost.png -->
<!-- caption: "dog bites man" and "man bites dog" give the same averaged vector, word order is lost. -->
<!-- source: original figure -->

---

# Keep the order: concatenate instead

<!-- img: visuals/assets/slp3-fig-6-14.png -->
<!-- caption: A feedforward language model concatenates the context embeddings side by side, so each position keeps its identity, and predicts the next word. The catch: the window is stuck at N words, which is next class's problem. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 6.14 -->

---

# Embeddings: pretrained or learned?

<!-- img: visuals/embeddings-source.png -->
<!-- caption: Start pretrained when labeled data is scarce; learning from scratch pays off with plenty of data. -->
<!-- source: original figure -->

---

# Overfitting & regularization

<!-- img: visuals/overfitting.png -->
<!-- caption: A flexible net can memorize the training set; dropout, weight decay, and early stopping fight back. -->
<!-- source: original figure -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the hand-traced MLP on the whiteboard. -->
<!-- source: original figure -->

---

# Exercise

<!-- layout: section -->

---

# Before you code: draw the network by hand

<!-- layout: statement -->
<!-- img: visuals/assets/photo-whiteboard.jpg -->
<!-- caption: Ten minutes at the whiteboard with a partner, then the code writes itself. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1542626991-cbc4e32524cc -->

---

# Activity: hand-trace the XOR network

<!-- img: visuals/activity-whiteboard.png -->
<!-- caption: This exact net solves XOR, and every weight and bias is on the sheet (green box). Fill the table row by row, check the XOR column, then do the one concrete gradient step. Answers get checked live on the board. -->
<!-- source: original figure; weights after Goodfellow et al. 2016, via SLP3 Fig. 6.6 -->

---

# Exercise: an MLP sentiment classifier

<!-- img: visuals/exercise-code.png -->
<!-- caption: Fill in the four TODOs, average embeddings, the MLP forward pass, and the five-line training loop. -->
<!-- source: weeks/week-04/class-01/exercise/mlp_classifier.py -->

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- caption: Loss falls to ~0 and the tiny separable set is fit perfectly, your first neural NLP classifier. -->
<!-- source: example run, seeded (torch.manual_seed(0)) -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- caption: Neuron + nonlinearity → MLP; learn by gradient descent on a loss; the training loop is universal. -->
<!-- source: original figure -->

---

# Reading & homework

<!-- img: visuals/reading-hw.png -->
<!-- caption: Required: seq2seq (Sutskever, Vinyals & Le, 2014), the only quizzed reading. J&M Ch. 6 & 13 are optional. HW2 is out today, due W5C1. -->
<!-- source: resources/landmark-papers.md -->

---

# Next class

<!-- img: visuals/next-up.png -->
<!-- caption: To model order and long-range structure we need RNNs and LSTMs, next class, plus Quiz 4. -->
<!-- source: original figure -->
