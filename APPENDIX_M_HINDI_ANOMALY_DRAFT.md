# Appendix M: Diagnosing the Hindi Early-Epoch Anomaly

*(Draft — nối tiếp Appendix L, đánh số bảng tiếp theo Table 12 → Table 13.
Kiểm tra lại 2 số ở §M.5 (H3) trước khi paste vào bản nộp — xem ghi chú ở
đầu message.)*

---

The Limitations section notes that the Hindi branch peaks at epoch 1 and
declines monotonically thereafter (XQuAD-hi F1: 67.09 → 65.61 → 62.69 →
55.25), while SQuAD-EN improves over the same span. We test four candidate
mechanisms for this pattern, using the same checkpoints (static configuration,
seeds 42/43/44) reported in Table 9.

**H1 — Tokenization fragmentation.** We measure the mean subword-per-word
ratio (XLM-R tokenizer) on the validation context of each target language
(n=240 samples each). Vietnamese, Arabic, and Hindi yield 1.20, 1.85, and 1.59
respectively. Arabic — not Hindi — shows the highest fragmentation, yet Arabic
does not exhibit the early-peak-then-decline pattern (Table 8); we therefore
find no support for tokenization fragmentation as the driver, and the
direction of this result (Arabic highest, but Arabic unaffected) argues
against the hypothesis more directly than a simple magnitude comparison would.

**H4 — Data-distributional differences.** Comparing answer length, context
length, and answer position across the three validation sets (n=1,190 each),
token-based measures are broadly comparable: mean answer length in tokens is
5.25 (VI), 5.66 (AR), and 5.82 (HI); mean context length is 214.6, 214.3, and
228.9 tokens respectively — differences under 11% on every token-based metric.
(Word-count-based answer length differs more sharply, but we do not treat this
as informative, since whitespace-based word segmentation is not comparable
across writing systems with different compounding and postposition
conventions; the token-based measures, computed with the same tokenizer used
in training, are the metrics we report on.) We find no support for
data-distributional differences as the driver.

**H2 — Transport plan (Sinkhorn γ) entropy.** For each checkpoint, we compute
the mean row-entropy of the transport plan γ over the validation set. Table 13
reports the result across all three seeds and four epochs for Hindi and
Vietnamese.

*Table 13: Mean Sinkhorn transport-plan entropy by epoch and language
(mean over 3 seeds).*

| Language | Epoch 1 | Epoch 2 | Epoch 3 | Epoch 4 | Relative growth (ep1→ep4) |
|---|---|---|---|---|---|
| Vietnamese | 0.979 | 1.208 | 1.385 | 1.427 | 45.7% |
| Hindi | 1.216 | 1.660 | 1.958 | 2.158 | 77.4% |

Two distinct patterns emerge. First, an **offset**: Hindi's transport-plan
entropy is already ~24% higher than Vietnamese's at epoch 1, before the static
and dynamic margin schedules have diverged — indicating the OT-guided
cross-lingual correspondence for Hindi is less coherent from the very start of
Stage-2 adaptation, not merely by epoch 4. Second, a **compounding effect**:
Hindi's entropy grows 77.4% in relative terms over four epochs, versus 45.7%
for Vietnamese — the incoherence does not merely persist at a fixed gap but
widens with continued training.

**H3 — Representation drift.** Following the diagnostic protocol of Appendix
F (n=50 sampled pairs, paired t-test and Wilcoxon), we measure Euclidean
distance and cosine alignment before/after adaptation for Hindi (epoch 1→3,
matched to Vietnamese's selected checkpoint span for a fair comparison) and
Vietnamese (epoch 1→3). Summing the absolute Cohen's d of the two primary
drift metrics (Euclidean distance, cosine alignment), Hindi shows a
substantially larger total effect size (|d|=2.52) than Vietnamese (|d|=1.71)
over the identical two-epoch training span, indicating faster representation
drift for Hindi even when training duration is held constant between the two
languages.

**Synthesis.** Of the four candidate mechanisms, tokenization fragmentation
and data-distributional differences find no support, while transport-plan
entropy and representation drift both point toward the same underlying
picture: Hindi's cross-lingual OT alignment is less coherent from the start of
adaptation and degrades further, in both absolute and relative terms, faster
than Vietnamese's over the same number of training epochs. We stop short of
claiming a fully established causal chain from this representation-level
instability to the epoch-level QA metric decline reported in Table 9; these
diagnostics establish a converging association at the representation level,
not a proof of mechanism. Notably, this instability does not track subword
fragmentation in any simple way: Arabic has the highest fragmentation ratio of
the three languages yet shows neither the entropy/drift pattern nor the
epoch-level decline observed for Hindi — suggesting the source of Hindi's
instability lies elsewhere in its typological distance from English, or in
some interaction between OT alignment and Devanagari-script representations
that a token-count-based measure does not capture. Extending this same
diagnostic to Arabic checkpoints, to confirm its entropy/drift trajectory
patterns with Vietnamese rather than Hindi, is a natural next step we leave to
future work.
