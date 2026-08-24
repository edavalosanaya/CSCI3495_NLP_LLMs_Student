---
marp: true
title: "W2C2: Text Classification"
paginate: true
---

# Text Classification

<!-- layout: title -->

## Week 2, Class 2: Naive Bayes, Logistic Regression & Evaluation
Lecture (~30 min), Quiz 2 (~10 min), break, exercise (~35 min)

<!-- img: visuals/assets/photo-spam-cans.jpg -->
<!-- caption: Why is junk email called "spam"? By the end of class you can build the classifier that catches it. -->
<!-- source: Arnold Gatilao, CC BY 2.0, Wikimedia Commons (SPAM Shrine.jpg) -->

---

# Today

<!-- img: visuals/today-agenda.png -->
<!-- caption: Task, features, two classifiers, evaluation, then Quiz 2 and the Sentiment Showdown. -->
<!-- source: deck figures; SLP3 Fig. 4.3 (Jurafsky & Martin); photo: Unsplash License -->

---

# Every inbox is a classifier at work

<!-- img: visuals/assets/photo-paperwork.jpg -->
<!-- caption: Spam filters, support routing, and review tagging all map a document to a label. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1450101499163-c8848c66ca85 -->

---

# The task

<!-- img: visuals/the-task.png -->
<!-- caption: Map a document to one label from a fixed set: sentiment, spam, topic, intent. -->
<!-- source: original figure -->

- Supervised: learn from labeled examples, predict on new text

---

# Features: bag-of-words

<!-- img: visuals/bow-pipeline.png -->
<!-- caption: A document becomes counts of its words; order is thrown away. -->
<!-- source: original figure -->

---

# Is this review positive or negative?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-reviews.jpg -->
<!-- caption: "It was not bad at all." Decide for yourself, then watch a machine try. -->
<!-- source: Pexels License, https://images.pexels.com/photos/3850250/pexels-photo-3850250.jpeg -->

---

# Bayes' rule, from counting

<!-- img: visuals/bayes-rule.png -->
<!-- caption: Take Monday's "two ways to count the overlap", divide by P(d), and Bayes' rule appears: it flips an impossible count into countable pieces. -->
<!-- source: original figure -->

---

# The naive assumption, laid out

<!-- img: visuals/naive-assumption.png -->
<!-- caption: Exact sentences never repeat, so treat words as independent given the class and multiply per-word probabilities. -->
<!-- source: original figure -->

---

# Naive Bayes: the idea

<!-- img: visuals/nb-idea.png -->
<!-- caption: A generative classifier picks the class that best explains the words (Bayes' rule). -->
<!-- source: original figure -->

---

# Training Naive Bayes

<!-- img: visuals/training-nb.png -->
<!-- caption: Just count, with add-one smoothing, the same trick as our language model. -->
<!-- source: original figure -->

- No gradient descent needed; training is a single counting pass

---

# The independence assumption

<!-- img: visuals/independence.png -->
<!-- caption: Words aren't really independent ("San" then "Francisco"), yet NB still works well. -->
<!-- source: original figure -->

---

# Logistic regression: the idea

<!-- img: visuals/logreg-idea.png -->
<!-- caption: A discriminative classifier models P(c | document) directly with the sigmoid. -->
<!-- source: original figure -->

---

# What the model sees: a feature vector x

<!-- img: visuals/assets/slp3-fig-4-2.png -->
<!-- caption: Six hand-designed sentiment features turn this whole review into x = [3, 2, 1, 3, 0, 4.19]. Whatever the document, the model only ever sees x. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 4.2 -->

---

# The entire model: multiply, add, squash

<!-- img: visuals/lr-score-anatomy.png -->
<!-- caption: The book's trained weights score the Fig. 4.2 review. Each weight says how strongly its feature votes, and for which class; the bias b is the starting score before any evidence. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Ch. 4.3.1 worked example -->

---

# Which w and b are best? Measure the mistake

<!-- img: visuals/nll-loss.png -->
<!-- caption: Training = picking w and b. The loss is the negative log likelihood of the correct labels: near zero when the model gives the truth high probability, huge when it is confidently wrong. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Ch. 4.5, cross-entropy loss -->

---

# Learning = rolling downhill on the loss

<!-- img: visuals/assets/slp3-fig-4-3.png -->
<!-- caption: Gradient descent: the slope of the loss says which way is uphill, so move every weight a small step the opposite way. Repeat until the loss stops falling. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 4.3 -->

- The gradient is just the slope, generalized to all weights at once

---

# The learning rate sets the step size

<!-- img: visuals/learning-rate.png -->
<!-- caption: The learning rate is a hyperparameter YOU choose, not a weight the model learns. A common trick: start larger, then decay it as training goes on. -->
<!-- source: original figure; Jurafsky & Martin, SLP3 (Jan 2026 draft), Ch. 4.6 -->

---

# NB vs. logistic regression

<!-- img: visuals/nb-vs-lr.png -->
<!-- caption: Both are linear over bag-of-words; they differ in what they model and how they train. -->
<!-- source: original figure -->

---

# Evaluating a classifier

<!-- img: visuals/accuracy-trap.png -->
<!-- caption: Accuracy alone misleads on imbalanced data; look at which kind of error happens. -->
<!-- source: original figure -->

---

# The confusion matrix

<!-- img: visuals/confusion-matrix.png -->
<!-- caption: Four outcomes for a target class: TP, FN, FP, TN. -->
<!-- source: original figure -->

---

# Precision, recall, F1

<!-- img: visuals/precision-recall.png -->
<!-- caption: Precision avoids false alarms; recall avoids misses; F1 is their harmonic mean. -->
<!-- source: original figure -->


---

# Which metric matters?

<!-- img: visuals/which-metric.png -->
<!-- caption: Match the metric to the cost of errors, and always use a held-out test set. -->
<!-- source: original figure -->

---

# Recap

<!-- img: visuals/recap2.png -->
<!-- caption: Bag-of-words → Naive Bayes & logistic regression → precision/recall/F1. -->
<!-- source: original figure -->

---

# Quiz 2 (paper)

<!-- img: visuals/quiz.png -->
<!-- caption: Covers the Week 2 lecture on n-gram LMs plus the reading (J&M Ch. 3 to 5). -->
<!-- source: original figure -->

- Clear your desk; finish early and your break starts early, need more time and it comes out of the break

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the Sentiment Showdown. -->
<!-- source: original figure -->

---

# Activity: Sentiment Showdown

<!-- img: visuals/activity-showdown.png -->
<!-- caption: Teams, programming competition, ~28 minutes. Highest F1 wins; sneakiest fooling review earns bonus glory. -->
<!-- source: weeks/week-02/class-02/exercise/sentiment.py -->

---

# Exercise: Sentiment Showdown

<!-- img: visuals/exercise-task.png -->
<!-- caption: Build a multinomial Naive Bayes classifier from scratch, no sklearn. -->
<!-- source: weeks/week-02/class-02/exercise/sentiment.py -->

---

# Exercise: the starter code

<!-- img: visuals/exercise-code.png -->
<!-- caption: Four TODOs in sentiment.py: train, score, predict, and evaluate. -->
<!-- source: weeks/week-02/class-02/exercise/sentiment.py -->

- Run it, then `pytest test_sentiment.py`

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- caption: All five test reviews classified correctly; precision, recall, and F1 all 1.00. -->
<!-- source: example run, solutions/sentiment.py -->

- Then break it: add a tricky review and watch the metrics move
