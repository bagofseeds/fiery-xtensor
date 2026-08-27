# Differences from xarray

`fiery.xtensor` is an [xarray](https://docs.xarray.dev)-like library, and it
follows xarray's conventions wherever it can: named **dimensions**, coordinate
**labels** keyed by dimension name, selection with `.sel` / `.isel`,
broadcasting and alignment **by name**, and a `set_options` context manager.
If you know xarray, `XTensor` should feel familiar.

The one structural difference colours everything else: an `XTensor` **is a
`torch.Tensor`** (a subclass), not a container that wraps an array. Autograd,
device, dtype, and `__torch_function__` all keep working, and every `torch.*`
op accepts an `XTensor`. xarray, by contrast, wraps a NumPy/Dask array and
exposes a curated API over it.

This page lists where the two diverge, in both directions.

## Added — things xarray does not have

- **A real tensor.** Gradients flow through named ops; `.cuda()`, `.double()`,
  `torch.stack`, `@`, and friends all work, because an `XTensor` *is* a tensor.
- **Numeric coordinates with an affine.** A dimension coordinate can be stored
  **compactly** as `spacing`/`origin` (materialised on demand), so a regular
  grid costs O(1) and its spacing can be a **learnable** tensor that gradients
  flow back to ([Proposal 0001](../proposals/0001-units.md)).
- **Two kinds of unit.** Beyond coordinate **position** units, a tensor's
  **values** carry a physical **data unit** with dimensional algebra and
  conversion, via an optional pint backend
  ([Proposal 0003](../proposals/0003-data-units.md)). xarray leaves units to
  the separate `pint-xarray` project.
- **Axis descriptors.** A name can be enriched with OME-NGFF-style `type`,
  `unit`, and `orientation` fields, and ops can resolve a `dim` by *type*
  ([Axis descriptors](descriptors.md)).
- **Unnamed dimensions.** A torch tensor often has anonymous axes (a batch dim,
  say); `XTensor` allows a dim to be unnamed (`None`). xarray requires every
  dimension to be named.
- **Attribute access to a label.** `x.red` reaches the position labelled
  `"red"` when that label is unambiguous — sugar over `.sel`.
- **Ergonomic constructors.** `xvector` / `xmatrix` and the `x*` factories
  (`xzeros`, `xones`, `xfill`, …) name and label axes at creation time.

## Missing — xarray features not (yet) here

- **No `Dataset`.** `XTensor` is the `DataArray` analogue only; there is no
  multi-variable container.
- **No split-apply-combine.** `groupby`, `resample`, `rolling`, `weighted`,
  and the rest of that suite are not implemented.
- **No IO / plotting / pandas bridge.** No NetCDF/Zarr readers, no `.plot`, no
  `.to_pandas()` / `MultiIndex`.
- **Curvilinear `.sel` is single-point and brute force; `.interp` is an
  approximate, iterative inverse.** A coordinate can span several dims at
  once — a `lat(y, x)`-style grid, jointly queried as
  `.sel(lat=.., lon=..)`/`.interp(lat=.., lon=..)`
  ([guide](coordinates.md#coordinates-spanning-several-dimensions),
  [Proposal 0005](../proposals/0005-multiple-coordinates.md), issue #82) —
  in two forms with very different capabilities. The compact **affine**
  form (`spacing`/`origin`, a linear map) inverts in closed form
  (`index = A⁻¹(world − origin)`), so both `.sel` and `.interp` are cheap
  and `.interp` supports a full vectorized/bulk query, collapsing the
  spanned dims into one new axis of sampled points. The explicit
  **curvilinear** form (an arbitrary array, no formula) has no such
  inverse: `.sel(lat=.., lon=..)` finds the nearest grid point by brute-force
  squared Euclidean distance over the queried coordinates' raw magnitudes
  (no scipy/sklearn dependency, so it stays GPU-capable) — unit-blind
  (mixing e.g. degrees and metres weights the nearer-magnitude one more
  heavily), and **single-point only**: it does **not** scale to
  *bulk regridding* (querying a whole new grid's worth of points at once),
  where the distance-matrix cost grows with the *product* of the source
  grid size and the query count — the reason every tree-based library in
  this space (xarray's `NDPointIndex`, `xoak`, `pyresample`) exists in the
  first place. `.interp` over a curvilinear coordinate is scoped to a
  **2-D** spanned coordinate (the common `lat(y, x)`/`lon(y, x)` case) and
  `method="nearest"`/`"linear"`: since there is no closed form, it seeds a
  fractional index from the same brute-force nearest lookup and refines it
  with a few Newton iterations against a finite-difference Jacobian, in
  plain `torch` (no extra dependency). Only that nearest-neighbor seed is
  computed one point at a time (no vectorized form of that lookup exists);
  the Newton solve and the final pull are both fully vectorized over the
  whole batch. It raises rather than guessing when a query is out of the
  grid's coordinate range, or lands close enough to a fold that the local
  Jacobian is itself singular there — away from the fold line itself it
  resolves to one of the (possibly several) valid preimages without warning
  that another exists — and does not carry gradients back to the query
  point or the coordinate arrays — only to the tensor's own data values,
  same as every other `interp` call. Convergence and singularity checks
  scale with the coordinates' own magnitude rather than a fixed absolute
  unit. A higher spline order, or more than 2 spanned dims, isn't
  implemented.
- **Alignment is inner-join only.** There is no `join="outer"/"left"/"right"`
  (which would need NaN fill); operands align to the **intersection** of their
  labels. (xarray's *default* `arithmetic_join` is also `"inner"`, so the
  common case matches.)

## Behaviour & vocabulary differences

- **`names`, not `dims`.** The dimension names live on `.names` (xarray:
  `.dims`); the values are a `torch.Tensor`, not a NumPy array.
- **Unnamed axes are allowed.** xarray has no unnamed dims; xtensor does (a
  plain tensor's anonymous batch axis). A pointwise op still aligns **by name**
  when the unnamed axes are all **leading** — the named suffix aligns by name,
  the anonymous prefix broadcasts positionally — and when both operands share
  the same `names`. An all-unnamed operand, a plain tensor, or a scalar
  broadcasts positionally; a `None` *after* a named axis on differently-named
  operands raises (ambiguous). See [broadcasting](broadcasting.md).
- **Selection vs interpolation are separate verbs.** Like xarray, `.sel` picks
  an existing position (exact, or snapped by `mode` with a `tolerance`) and
  `.interp` computes values off the grid, on both regular and irregular
  numeric coordinates
  ([Proposal 0004](../proposals/0004-numeric-selection.md)).
- **A range `.sel` is half-open, not inclusive.** `x.sel(t=slice(lo, hi))`
  selects `lo <= value < hi`, like ordinary Python/`__getitem__` slicing.
  xarray's label-based `.sel(x=slice(a, b))` is **inclusive on both ends**
  (`a <= value <= b`) — a deliberate divergence here, to keep one slicing
  rule across the whole library rather than a special inclusive-both-ends
  case for coordinate ranges (issue #109).
- **Name-as-`dim` is method-only.** `x.sum(dim="height")` works; the functional
  form `torch.sum(x, dim="height")` does not — a **PyTorch limitation** (its
  C-level argument parser rejects a string `dim` before our hook runs), not a
  choice of ours (see [Names & dimensions](names.md)).
