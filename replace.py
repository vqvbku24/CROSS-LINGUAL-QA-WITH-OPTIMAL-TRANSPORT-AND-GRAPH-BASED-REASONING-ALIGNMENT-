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

# Replacement 2
start2 = r'We extend the geometric diagnostics'
end2 = r'appears to be a recurring pattern rather than a language-specific effect.'
idx_start2 = text.find(start2)
idx_end2 = text.find(end2) + len(end2)
if idx_start2 != -1 and idx_end2 > idx_start2:
    new_text2 = r'''We extend the geometric diagnostics of Appendix~\ref{sec:appendix_diagnostics} to Arabic and Hindi (Figure~\ref{fig:alignment_diagnostics_ar_hi}, $n=50$ pairs each). Anisotropy control (panels a, e) remains strong for both languages: the OT-weighted target counterpart is substantially more similar to the English answer token than a randomly sampled target token (paired $t$: $p=2\text{e-}12$ for Arabic, $p=6.1\text{e-}13$ for Hindi), ruling out a trivial anisotropy artifact.

Euclidean distance decreases for a majority of pairs in both languages (31/50 improved for Arabic, 31/50 for Hindi), and both shifts are statistically significant (Arabic: paired $t$ $p=0.016$; Hindi: paired $t$ $p=0.0076$), mirroring the Vietnamese pattern.

Angular alignment, however, shows a genuinely language-dependent pattern rather than a single universal effect: Vietnamese improves for a majority of pairs (Appendix~\ref{sec:appendix_diagnostics}), while Arabic shows a significant decrease for a majority of pairs (15/50 improved, i.e.\ 35/50 decreased; paired $t$: $p=0.0025$), and Hindi shows a similar-direction but non-significant decrease (20/50 improved; paired $t$: $p=0.064$, Wilcoxon: $p=0.11$).

Common-mode drift (panels d, h) is positive and highly significant for both languages (mean cosine $=0.35$ for Arabic, $p=4.4\text{e-}15$; $0.40$ for Hindi, $p=3.2\text{e-}18$) -- in fact noticeably stronger than the weak common-mode drift observed for Vietnamese (mean $=0.12$, Appendix~\ref{sec:appendix_diagnostics}), suggesting Arabic and Hindi representations shift in a more coordinated direction with English than Vietnamese's do, even though Vietnamese is the language where angular alignment itself improves most cleanly.

Overall, the geometric behaviour of the OT-guided alignment is directionally consistent across all three languages on distance and common-mode drift, but angular alignment is language-specific: it improves for Vietnamese, while it mildly declines for Arabic (significantly) and Hindi (not significantly) -- underscoring that no single geometric signature generalises uniformly across typologically distant target languages.'''
    text = text[:idx_start2] + new_text2 + text[idx_end2:]
else:
    print('Failed to replace 2')

with open(r'd:\URA\OT\code\back_up\ACL_latex\acl_latex.tex', 'w', encoding='utf-8') as f:
    f.write(text)
print('Done replacing.')
