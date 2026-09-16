import sys

with open(r'd:\URA\OT\code\back_up\ACL_latex\acl_latex.tex', 'r', encoding='utf-8') as f:
    text = f.read()

# Replacement 1
start1 = r'Figure~\ref{fig:alignment_diagnostics} summarizes these diagnostics'
end1 = r'underscoring the necessity of spatial-domain coordination.'
idx_start1 = text.find(start1)
idx_end1 = text.find(end1) + len(end1)
if idx_start1 != -1 and idx_end1 > idx_start1:
    new_text1 = r'''Figure~\ref{fig:alignment_diagnostics} summarizes these diagnostics, computed over $n=50$ sampled pairs. Panel (a) confirms that the OT-weighted correspondence is non-trivial: the OT-selected Vietnamese counterpart is substantially more similar to the true English answer token (median cosine $\approx 0.86$) than a randomly sampled Vietnamese token (median $\approx 0.63$), a large and highly significant gap (paired $t$: $p=1.6\text{e-}20$), ruling out a trivial anisotropy artifact.

Panels (b) and (c) show that Stage-2 adaptation moves the OT-weighted Vietnamese representations both closer to (43/50 pairs improved; paired $t$: $p=5.7\text{e-}08$) and more angularly aligned with (32/50 pairs improved; paired $t$: $p=0.0063$) their English counterparts. This is a more direct and intuitive picture than the tension we originally reported: rather than QA gains arising despite a majority-case drop in angular alignment, distance and angular alignment move together for a clear majority of pairs, so the geometric evidence here is consistent with -- rather than in tension with -- the causal ablation gains reported in Table~\ref{tab:main_results}. Panel (d) shows the accompanying norm changes for both languages are modest but statistically detectable.

Panel (e) reveals that raw per-layer cosine alignment is already near-saturated across the probed deeper layers (6--9), meaning the learned layer-mixing preference for deeper layers (Figure~\ref{fig:layer_weights}) is likely driven by syntactic/structural utility rather than alignment quality alone. Panel (f) shows that common-mode drift between the English and OT-weighted Vietnamese shift vectors is only weakly positive (mean cosine $=0.12$, one-sample $t$: $p=0.00061$): statistically distinguishable from zero, but a considerably smaller effect than a tightly coordinated shift, indicating that Stage-2 adaptation nudges the two languages' representations loosely in a common direction rather than moving them as a single rigid transformation.

Taken together, these diagnostics support a more direct account than we originally gave: OT-guided adaptation moves Vietnamese representations measurably closer to, and better angularly aligned with, their English counterparts for a clear majority of sampled pairs, consistent with -- though not proof of -- the mechanism underlying the framework's downstream QA gains (Section~\ref{sec:ablation_study}, Table~\ref{tab:main_results}).'''
    text = text[:idx_start1] + new_text1 + text[idx_end1:]
else:
    print('Failed to replace 1')

# Replacement 3
start3 = r'The held-out columns in Table~\ref{tab:source_preservation} reinforce this picture'
end3 = r'holds under held-out evaluation as well.'
idx_start3 = text.find(start3)
idx_end3 = text.find(end3) + len(end3)
if idx_start3 != -1 and idx_end3 > idx_start3:
    new_text3 = r'''The held-out columns in Table~\ref{tab:source_preservation} sharpen this picture. For the OT-only ablation (no $\mathcal{L}_{reg}$), removing the domain anchor leaves the in-domain SQuAD-EN score nearly unchanged for Vietnamese ($\Delta_{src}=-0.09$) but produces a clear drop on its held-out benchmarks ($-2.32$ on XQuAD-en, $-3.13$ on MLQA-en), with an even sharper XQuAD-en collapse for Arabic ($-4.74$). Our full configuration instead keeps $\Delta_{src}$ close to zero across all three languages on SQuAD-EN ($+1.18$ Vietnamese, $-0.27$ Arabic, $-0.18$ Hindi), and on the held-out XQuAD-en benchmark specifically, it does more than merely preserve English performance for two of the three languages: Arabic and Hindi both show a positive $\Delta_{src}$ ($+1.11$ and $+1.17$), meaning the adapted student answers English XQuAD questions more accurately than the frozen pre-alignment teacher. Vietnamese's XQuAD-en gap remains slightly negative ($-0.92$) but recovers sharply relative to its own no-$\mathcal{L}_{reg}$ ablation ($-2.32$), and MLQA-en stays within a narrow band around zero for all three languages ($-0.3$, $-0.28$, $+0.06$). This confirms that the domain-consistency anchor's benefit is not an artifact of the in-domain SQuAD-EN distribution: on held-out XQuAD-en in particular, it converts what would otherwise be a source-language loss into a net gain for the majority of our evaluated languages.'''
    text = text[:idx_start3] + new_text3 + text[idx_end3:]
else:
    print('Failed to replace 3')

# Replacement 4
start4 = r'whose individual contribution is isolated separately in Table'
end4 = r'consistent with catastrophic forgetting.'
idx_start4 = text.find(start4)
idx_end4 = text.find(end4) + len(end4)
if idx_start4 != -1 and idx_end4 > idx_start4:
    new_text4 = r'''whose individual contribution is isolated separately in Table~\ref{tab:source_preservation}, where removing it leaves in-domain SQuAD-EN nearly untouched but causes held-out English F1 to drop by up to $-3.13$ points (MLQA-en, Vietnamese branch) relative to the pre-alignment teacher, consistent with catastrophic forgetting concentrated on held-out rather than in-domain data.'''
    text = text[:idx_start4] + new_text4 + text[idx_end4:]
else:
    print('Failed to replace 4')

with open(r'd:\URA\OT\code\back_up\ACL_latex\acl_latex.tex', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done replacing.')
