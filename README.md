# Mechanistic interpretability experiments
 
Small, controlled experiments on what neural networks represent internally and where, in a language model (**Gemma-2B**) and a self-supervised speech model (**HuBERT**). Each experiment states its predictions before running and compares against a baseline, so that a clean-looking pattern has to survive a control before it counts.
 
| # | Notebook | Question | Status |
|---|---|---|---|
| 1 | [`Bigrams_PreResearch_Experiment_1.ipynb`](Bigrams_PreResearch_Experiment_1.ipynb) | Do Gemma's SAE features pass information between tokens? (one-sentence pilot) | Done, with known limitations |
| 2 | [`Experiment_2_Position_Resolved_Bigrams.ipynb`](Experiment_2_Position_Resolved_Bigrams.ipynb) | Same question, measured per position on 18 matched sentence pairs with controls | Done: the pilot's links did not hold up |
| 3 | [`Experiment_3_HuBERT_Articulatory_Probing.ipynb`](Experiment_3_HuBERT_Articulatory_Probing.ipynb) | Which HuBERT layers encode how speech is physically produced? | Done for one speaker; silence-trimmed rerun pending |
 
---
 
## Part 1: Gemma-2B, do SAE features compute or just carry tokens forward?
 
Sparse autoencoders (`gemma-2b-res-jb`) on the residual stream after layers 6 and 12, with [SAE Lens](https://github.com/jbloomAus/SAELens) and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
 
I remove one layer-6 feature's contribution to the residual stream and measure which layer-12 features lose activation, and **at which token**.
 
- If a feature only affects later features **at the same token**, the residual stream is carrying that token's representation forward.
- If it affects features **at a later token**, information has moved between positions, which is a candidate for real computation.
### Experiment 1: pilot
 
**Setup.** Sentence: *"The quick brown fox jumps over the lazy dog"*. I took the 12 strongest layer-6 and layer-12 features, ablated each layer-6 feature in turn, and recorded how much each layer-12 feature's peak activation dropped.
 
**Result.** Of the 15 links with an effect above 0.5, **13 connected the same token at layer 6 and layer 12** (e.g. ' quick' → ' quick', effect 13.4). Two weaker links crossed tokens:
 
| Layer-6 feature | Layer-12 feature | Effect |
|---|---|---|
| 13987 (' brown', pos 3) | 5589 (' jumps', pos 5) | 1.82 |
| 4545 (' over', pos 6) | 10094 (' the', pos 7) | 0.53 |
 
**Limitations.** These are why the two cross-token links were hypotheses, not findings:
 
1. The effect was the change in a feature's maximum over the whole sentence, **including the BOS token at position 0**.
2. Effects were not measured per position; which token a link belonged to was inferred from where each feature peaked.
3. One sentence, and no baseline for how big an effect a random perturbation would cause.
### Experiment 2: position-resolved ablation with controls
 
**What's different**
 
- Each layer-6 feature is ablated **at one position only**, and the effect on every layer-12 feature is measured **at every position**. Position 0 is excluded everywhere.
- Effect size is the share of a layer-12 feature's peak activation that it loses.
- **18 matched pairs (36 sentences).** Each frame appears with a predictable first word and a merely adjacent one, with everything else identical:
  - *She spread the **peanut** butter on her toast.*
  - *She spread the **cold** butter on her toast.*
- **Random-direction control:** every ablation is repeated with a random vector of the same length.
- **Manipulation check:** confirms Gemma-2B actually finds word 2 more likely after the predictable word 1.
- **Next-token check:** does removing the word-1 feature change what the model predicts?
- **Causality sanity check:** positions before the ablation must not change.
**Predictions (written before running).** If the pilot's cross-token links reflect real computation tied to predictability:
 
- **(A)** cross-position links should be more common for real features than for random directions;
- **(B)** the word 1 → word 2 link should be stronger for predictable pairs than for their matched controls;
- **(C)** ablating that feature should lower P(word 2) more for predictable pairs.
**Results** (Gemma-2B, layers 6 → 12, top 5 features per position, strong link = at least 10% of peak lost)
 
| Check | Result |
|---|---|
| Manipulation check | Word 2 more likely after the predictable word 1 in **17 of 18 pairs** (median P = 0.82 vs 0.025) |
| (A) Where strong effects land | Real features: **91% same-token** (2,946 of 3,227). Random directions: **65% same-token** (2,517 of 3,899) |
| (A) Across thresholds | Cross-position share 19% / 9% / 2% for features vs 45% / 35% / 21% for random, at 5% / 10% / 25% |
| (B) Word 1 → word 2 link | Predictable > control in only **3 of 18 pairs** (Wilcoxon p = 0.001), but random directions show the same reversal (p = 0.004) |
| (C) Effect on P(word 2) | No difference between conditions (p = 0.15). Locally, removing the feature lowers log P(word 2) by a median of **0.60 vs 0.06** for a random direction (predictable pairs) |
| Causality check | 0.0 (earlier positions never change) |
 
![Experiment 2 summary](experiment2_summary.png)
 
**Interpretation.** None of the three predictions held. Real SAE features acted far more *locally* than random perturbations of equal size, at every threshold, so the pilot's cross-token links did not survive a controlled test. The features clearly matter at their own token: removing one cuts the model's prediction of the next word about ten times more than a random push. The reversal in (B) appears for random directions too, so it reflects how sensitive word 2 is to any disturbance at word 1, not specific features. It may also partly come from normalising by peak activation, which a raw residual-stream check would settle.
 
**A hypothesis for future work.** The strongest cross-position links ran from function words and verbs into the next word (*after → lunch*, *bought → paper*) or from control adjectives (*local → school*), never from a predictable first word. Cross-position links may appear where word 2 must be *combined* with word 1, not where it is *predicted*.
 
**Caveats.** Random directions may be off-distribution; covariance-matched baselines would be fairer. Only the top 5 features per position, one layer pair and 18 pairs were tested.
 
---
 
## Part 2: HuBERT, where is articulation encoded?
 
### Experiment 3: layer-wise probing against articulatory data
 
**Question.** HuBERT is trained only on audio, never on how speech is produced. Do its layers nonetheless encode the movements of the jaw, lips, tongue and velum, and in which layers?
 
**Data.** [MOCHA-TIMIT](https://www.cstr.ed.ac.uk/research/projects/artic/mocha.html), speaker `fsew0`: 460 sentences (27.8 min) of 16 kHz audio with electromagnetic articulography (EMA) at 500 Hz. 14 channels: x and y for the jaw, upper lip, lower lip, tongue tip, tongue blade, tongue dorsum and velum.
 
**Method.** For each of HuBERT Base's 13 layers (`facebook/hubert-base-ls960`), a **linear** ridge probe predicts the 14 articulator trajectories from frame-level features. The EMA is low-pass filtered at 20 Hz and aligned to HuBERT's 20 ms frames. The ridge penalty is chosen on validation sentences, and the score is the Pearson correlation on 46 held-out test sentences.
 
**Controls.**
- **MFCC baseline:** 13 MFCCs + deltas, ±5 frames of context. HuBERT has to beat plain acoustics.
- **Random-init HuBERT:** same architecture, untrained weights. This separates what was *learned* from what the architecture gives for free.
**Predictions (written before running).**
- **(P1)** The best HuBERT layer beats MFCC.
- **(P2)** Pretrained beats random init at most layers.
- **(P3)** Articulatory information peaks in the middle layers.
**Results** (speaker `fsew0`, silence included)
 
| | Mean correlation with measured EMA |
|---|---|
| HuBERT, best layer (12 of 12) | **0.825** |
| MFCC baseline | 0.618 |
| Random-init HuBERT, layer 12 | 0.391 |
 
| Jaw | Upper lip | Lower lip | Tongue tip | Tongue blade | Tongue dorsum | Velum |
|---|---|---|---|---|---|---|
| 0.77 | 0.75 | 0.77 | 0.87 | 0.84 | 0.83 | 0.95 |
 
![Experiment 3 probing](experiment3_probing.png)
 
- **(P1) Supported.** The best layer beats MFCC (0.83 vs 0.62). All 46 test sentences favoured HuBERT (Wilcoxon p = 2.8 × 10⁻¹⁴, the smallest possible with 46 sentences).
- **(P2) Supported.** Pretrained beats random init at **all 13 layers**, again in all 46 test sentences. The articulatory information is learned.
- **(P3) Not supported.** Articulatory information is strongest in the **final** layer, not the middle. This is plausible for HuBERT, which is trained to predict phone-like units from its last layer.
**Caveats.**
- **Silence at the start and end of sentences was included.** At rest, the velum and lips sit in a breathing position, so a probe can score well just by detecting speech vs silence. The velum's 0.95 (higher than the tongue's) is probably inflated this way, and the other values somewhat. The comparisons in P1 and P2 held in every sentence, so they are unlikely to change, but treat the absolute numbers as upper estimates.
- One speaker.
- A linear probe shows information is *decodable* from a layer, not that the model *uses* it.
This replicates, at small scale, the direction of [Cho et al., *Evidence of Vocal Tract Articulation in Self-Supervised Learning of Speech* (ICASSP 2023)](https://arxiv.org/abs/2210.11723).
 
---
 
## Running the notebooks
 
- **Experiments 1–2:** Colab with a GPU runtime (*Runtime → Change runtime type → T4 GPU*), or locally on a Mac with 16 GB+ RAM (CPU). You need a Hugging Face token with access to [`google/gemma-2b`](https://huggingface.co/google/gemma-2b). Versions are pinned to `sae-lens>=6,<7` and `transformer-lens>=3,<4`, because TransformerLens 4 removed `HookedTransformer`.
- **Experiment 3:** Colab free tier (T4), about 15 minutes. It downloads MOCHA-TIMIT (free for non-commercial use) and HuBERT automatically; no token needed.
## Next
 
- **Experiment 3:** rerun with silence trimmed, and on the second speaker (`msak0`), to check the numbers and the late-layer peak.
- **Experiment 3 → features:** train a sparse autoencoder on HuBERT's best layer and test whether individual features track specific articulators, using ablation with a random-direction control, as in Experiment 2.
- **Gemma:** test whether fixed compounds (*peanut butter*) are processed differently from compositional phrases (*cold butter*), with covariance-matched baselines and more pairs.
