# TODO

## Ask About Nitrogen Model

- Confirm whether `derivatives()` should pass `M[Ix.FON]` instead of `M[Ix.DON]`
  into `R_DON()`. The current call uses the DON mass as the first argument even
  though `R_DON()` names and documents that parameter as `M_FON`.
- Confirm whether `U_DIN()` should clamp uptake to zero when `0 < S_s < S_wp`.
  The current formula can produce negative uptake below wilting point.
- Confirm the intended temperature factor normalization. The `tempfactor()`
  docstring says it has value `2.73` at `20 C`, but the implementation returns
  `1.0` at `20 C`.
- Decide whether the placeholder source/sink functions `Q_SON()`, `Q_FON()`,
  and `Q_DIN()` should accept inputs now, or stay as zero-returning placeholders
  until source/sink processes are implemented.
