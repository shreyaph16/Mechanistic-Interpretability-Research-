# Do early SAE features compute, or just carry tokens forward?
 
A small mechanistic-interpretability project on **Gemma-2B**, using sparse autoencoders (`gemma-2b-res-jb`) on the residual stream after layers 6 and 12, with [SAE Lens](https://github.com/jbloomAus/SAELens) and [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).
 
The question is how a layer-6 feature influences layer-12 features. I remove one layer-6 feature's contribution to the residual stream and measure which layer-12 features lose activation, and **at which token**.
 
- If a feature only affects later features **at the same token**, the residual stream is just carrying that token's representation forward.
- If it affects features **at a later token**, information has moved between positions, which is a candidate for real computation.
| Notebook | What it does | Status |
|---|---|---|
| [`Bigrams_PreResearch_Experiment_1.ipynb`](Bigrams_PreResearch_Experiment_1.ipynb) | Pilot on one sentence | Done, with known limitations |
| [`Experiment_2_Position_Resolved_Bigrams.ipynb`](Experiment_2_Position_Resolved_Bigrams.ipynb) | Fixes the pilot's measurement issues and tests 18 matched sentence pairs with controls | Ready to run |
 
## Experiment 1: pilot
 
**Setup.** Sentence: *"The quick brown fox jumps over the lazy dog"*. I took the 12 strongest layer-6 and layer-12 features, ablated each layer-6 feature in turn, and recorded how much each layer-12 feature's peak activation dropped.
 
**Result.** Of the 15 links with an effect above 0.5, **13 connected the same token at layer 6 and layer 12** (e.g. ' quick' → ' quick', effect 13.4). Two weaker links crossed tokens:
 
| Layer-6 feature | Layer-12 feature | Effect |
|---|---|---|
| 13987 (' brown', pos 3) | 5589 (' jumps', pos 5) | 1.82 |
| 4545 (' over', pos 6) | 10094 (' the', pos 7) | 0.53 |
 
This fits the residual stream mostly carrying tokens forward between these layers, with possible exceptions.
 
**Limitations.** These are why the two cross-token links are hypotheses, not findings:
 
1. The effect was the change in a feature's maximum over the whole sentence, **including the BOS token at position 0**.
2. Effects were not measured per position; which token a link belonged to was inferred from where each feature peaked.
3. One sentence, and no baseline for how big an effect a random perturbation would cause.
## Experiment 2: position-resolved ablation with controls
 
**What's different**
 
- Each layer-6 feature is ablated **at one position only**, and the effect on every layer-12 feature is measured **at every position**. Position 0 is excluded everywhere.
- Effect size is the share of a layer-12 feature's peak activation that it loses.
- **18 matched pairs (36 sentences).** Each frame appears with a predictable first word and a merely adjacent one, with everything else identical:
  - *She spread the **peanut** butter on her toast.*
  - *She spread the **cold** butter on her toast.*
- **Random-direction control:** every ablation is repeated with a random vector of the same length.
- **Manipulation check:** confirms Gemma-2B actually finds word 2 more likely after the predictable word 1.
- **Next-token check:** does removing the word-1 feature change what the model predicts?
- **Causality sanity check:** positions before the ablation must not change. In testing this was exactly 0.
**Predictions (written before running).** If the pilot's cross-token links reflect real computation tied to predictability:
 
- **(A)** cross-position links should be more common for real features than for random directions;
- **(B)** the word 1 → word 2 link should be stronger for predictable pairs than for their matched controls (Wilcoxon signed-rank test);
- **(C)** ablating that feature should lower P(word 2) more for predictable pairs.
**Results**
 
> _Not run yet. Paste the output of the notebook's "Summary" cell here, add `experiment2_summary.png`, and write two or three sentences on whether (A), (B) and (C) held. A null result counts._
 
## Running it
 
Open Experiment 2 in Colab with a GPU runtime (*Runtime → Change runtime type → T4 GPU*) and run all cells. You need a Hugging Face token with access to [`google/gemma-2b`](https://huggingface.co/google/gemma-2b).
 
Versions are pinned to `sae-lens>=6,<7` and `transformer-lens>=3,<4`, because TransformerLens 4 removed `HookedTransformer`.
 
## Next
 
- Test other layer pairs, and weaker features than the top 5 per position.
- Apply the same controlled-ablation approach to self-supervised speech models (wav2vec 2.0, HuBERT), where articulatory measurements can serve as ground truth for what features encode.
