# Coordinates

Coordinate labels let you address a position along a dimension by a meaningful
name instead of an integer, and — like names — they follow their dimension
through permutes, transposes, and reductions. Coordinates can also be
**structured** (a descriptor per position) or **numeric** (a physical position
per element).

## Coordinate labels

Labels are keyed by dimension **name**, so — like xarray — they simply follow
their dimension through a `permute`/`transpose`/reduction with no bookkeeping:

```python
import torch
from fiery.xtensor import XTensor

# Coordinate labels: address positions along a dimension by label
m = XTensor(
    torch.arange(6).reshape(2, 3),
    names=("row", "channel"),
    coords={"channel": ("x", "y", "z")},
)
m.coords                        # {'channel': ('x', 'y', 'z')}
m.T.coords                      # {'channel': ('x', 'y', 'z')} — follows the dim
```

Four equivalent ways to select the position labelled `"y"` along `channel`:

=== "By label"

    ```python
    m.sel(channel="y")
    ```

=== "By attribute"

    ```python
    m.y
    ```

=== "Positional label"

    ```python
    m[:, "y"]   # label on the 2nd axis
    ```

=== "By integer position"

    ```python
    m.isel(channel=1)
    ```

A **positional** coordinate label works anywhere an integer index does, resolved
against the axis it sits on — so it composes with ints, slices, `...` and
newaxis:

```python
x[..., "r1", "y"]   # label the last two axes; a bare label drops its axis
x[:, ["w", "y"]]    # a list of labels is an advanced index (keeps the axis)
```

## Structured coordinates

A coordinate **label** can itself be a descriptor dict, so each *position* along
an axis is described — the position-level analogue of an
[axis descriptor](descriptors.md). A label's `"name"` is still its identity for
selection; the other fields are queryable:

```python
img = xtensor(data, names=("c", "y", "x"), coords={"c": [
    {"name": "DAPI", "type": "nucleus"},
    {"name": "GFP",  "type": "signal"},
    {"name": "RFP",  "type": "signal"},
]})

img.sel(c="GFP")            # by name — drops the axis (as before)
```

A **query** (a dict where a coordinate label is expected) selects every
*matching position* and keeps the axis — two equivalent spellings:

=== "By `[]`"

    ```python
    img[{"type": "signal"}]
    ```

=== "By `.sel`"

    ```python
    img.sel(c={"type": "signal"})
    ```

The matches become a `slice` when they are contiguous and an index list when
they are not; either way the axis is kept. This mirrors the descriptor query
for *axes*: a `{"type": ...}` dict picks **axes** in a `dim=`/`movedim` slot and
**positions** in a `[]`/`sel` slot, so the two never collide.

## Numeric coordinates

A coordinate can also be **numeric** — a physical position per element — given
compactly as `spacing` and/or `origin`, each a value with a unit
([Proposal 0001](../proposals/0001-units.md)):

```python
img = xtensor(data, names=("y", "x"),
              coords={"x": {"spacing": (0.5, "mm"), "origin": (-16, "mm")}})

img.coords["x"]["spacing"].unit   # "mm"  — the position unit
img.coords["x"]["value"]          # tensor([-16, -15.5, …]) with .units == "mm"
```

Only one of `spacing`/`origin` needs to be given — the other defaults, and the
default always takes the unit of the one you did give:

```python
xtensor(data, names=("x",), coords={"x": {"spacing": (0.5, "mm")}})
# origin defaults to 0 mm  -> positions 0, 0.5, 1, …

xtensor(data, names=("x",), coords={"x": {"origin": (-16, "mm")}})
# spacing defaults to 1 mm -> positions -16, -15, -14, …
```

`["value"]` materializes `origin + i·spacing` on demand (differentiable — a
learnable `spacing` tensor keeps its gradient). An **irregular** axis instead
takes an explicit unitful tensor of positions — three equivalent spellings:

=== "Bare tensor"

    ```python
    sig = xtensor(trace, names=("t",),
                  coords={"t": xtensor([0., 0.5, 2., 4.], units="s")})
    ```

=== "Dict form"

    ```python
    sig = xtensor(trace, names=("t",),
                  coords={"t": {"value": xtensor([0., 0.5, 2., 4.], units="s")}})
    ```

=== "Bare list of numbers"

    ```python
    # dimensionless, unless you use one of the unitful forms above -- this
    # is a numeric coordinate, not a set of labels that happen to be numbers
    sig = xtensor(trace, names=("t",), coords={"t": (0., 0.5, 2., 4.)})
    ```

Numeric coordinates slice **affinely** (`img[..., 2:]` shifts the origin,
`img[..., ::2]` scales the spacing) and convert with `img.coords["x"].to("um")`.
The position unit is separate from the *data* unit of the tensor's values (see
[Data units](data-units.md)).

A `slice` selector on `.sel` picks a **value range** instead of a single tick
— unit-aware, and half-open like ordinary Python slicing (`lo <= value <
hi`, **not** xarray's inclusive-both-ends convention):

```python
sig.sel(t=slice("1s", "5s"))   # every tick with 1s <= value < 5s
sig.sel(t=slice(None, "2s"))   # value < 2s
```

Bounds are compared numerically regardless of the order they're given in or
of the coordinate's own direction — `slice(lo, hi)` and `slice(hi, lo)`
select the same range. An out-of-range or empty result is a well-formed
empty axis, not an error.

### Selecting near a value

A bare `.sel(t=v)` (no `mode`) is **exact** — it raises if `v` isn't an
existing tick. Pass `mode` to snap to a nearby one instead:

=== "Nearest (round)"

    ```python
    sig.sel(t=1.2, mode="round")   # the nearest tick by value
    ```

=== "Floor / ceil"

    ```python
    sig.sel(t=1.2, mode="floor")   # largest tick <= 1.2
    sig.sel(t=1.2, mode="ceil")    # smallest tick >= 1.2
    ```

=== "Prev / next"

    ```python
    sig.sel(t=1.2, mode="prev")   # neighbouring tick at the lower index
    sig.sel(t=1.2, mode="next")   # neighbouring tick at the higher index
    ```

A `mode` alone snaps with no limit on the distance; add `tolerance` to cap
how far it's allowed to go:

```python
sig.sel(t=1.2, mode="round", tolerance=0.1)   # raises: nearest tick is farther away than 0.1
```

## Interpolating between ticks

Where `.sel` picks an existing position, `.interp` computes a value at an
arbitrary one:

```python
sig.interp(t=1.25)               # one point -> drops the axis
sig.interp(t=[1.0, 1.25, 1.5])   # several   -> keeps the axis, the queried
                                  # values becoming its new coordinate
```

`method` picks the interpolation order: `"nearest"` works out of the box;
anything higher (`"linear"` *(default)*, `"quadratic"`, `"cubic"`, or a plain
integer order) needs the optional interpolation extra,
`pip install fiery-xtensor[interp]`. A **regular** coordinate (`spacing`/
`origin`) supports every order; an **irregular** one only `"nearest"`/
`"linear"`.

An out-of-range query is governed by `bound` (default `"replicate"`, which
clamps to the edge value) and `extrapolate`, settable per call or as a
standing default with [`set_options`][fiery.xtensor.set_options]:

```python
sig.interp(t=10.0, bound="replicate")   # clamps to the last tick

with set_options(interp_bound="replicate"):
    sig.interp(t=10.0)   # every call in this block clamps too
```

## Coordinates spanning several dimensions

A coordinate can also span **more than one axis** at once — a `lat`/`lon`
grid over `(y, x)`, not a separate coordinate per axis. Key it by a name that
isn't a dim, paired with the **tuple of dims** it spans:

```python
img = xtensor(data, names=("y", "x"), coords={
    "lat": (("y", "x"), {"spacing": ([1.0, 0.0], "deg"), "origin": (10.0, "deg")}),
    "lon": (("y", "x"), {"spacing": ([0.0, 2.0], "deg"), "origin": (20.0, "deg")}),
})
```

This is the same compact `spacing`/`origin` form as a 1-D numeric coordinate,
just with one `spacing` component **per spanned dim** (here, `lat` only
varies along `y`, `lon` only along `x` — a plain lat/lon grid; a rotated or
sheared grid would give both components non-zero values). Query every
spanned name **together** in one `.sel`/`.interp` call — a *joint* query,
solved as a small linear system (`index = A⁻¹(world − origin)`) rather than
axis-by-axis:

```python
img.sel(lat=11.0, lon=24.0)                    # exact match -> a scalar
img.sel(mode="round", lat=11.4, lon=23.6)      # snaps to the nearest position
img.interp(lat=11.4, lon=23.6)                 # interpolates -> a scalar
```

A query with a **list** collapses the spanned dims into **one new axis** of
sampled points (not an outer-product grid — the dims are coupled, so you
move through both at once), carrying the sampled `lat`/`lon` values back as
a riding coordinate on it. Name the new axis with `name=`:

```python
pts = img.interp(lat=[10.0, 11.0, 12.0], lon=[20.0, 22.0, 24.0], name="pts")
pts.shape            # (3,)
pts.coords["lat"]    # the three sampled lat values, riding along "pts"
```

Composes with ordinary 1-D indexers in the same call
(`img.sel(t=1.0, lat=11.0, lon=24.0)`), but a dim resolved by the joint query
can't *also* be given directly in the same call. Only `mode="round"` (`.sel`)
is supported for the joint query; `.interp` keeps its usual `method=`/
`bound=`/`extrapolate=`.

A coordinate can also span several dims **without** an affine formula — an
explicit array, one value per grid point, for a genuinely curved
(**curvilinear**) grid that a compact `spacing`/`origin` can't describe:

```python
grid = xtensor(data, names=("y", "x"), coords={
    "lat": (("y", "x"), lat_array),   # an (y, x)-shaped tensor of latitudes
    "lon": (("y", "x"), lon_array),
})
grid.sel(lat=52.1, lon=4.3, method="nearest")   # nearest grid point, by raw distance
```

Unlike the affine form, this has no inverse formula to solve — `.sel` finds
the nearest point by brute-force distance instead (unit-blind, and a
single-point query only; it isn't meant for bulk regridding).

`.interp` over a curvilinear coordinate computes a genuinely interpolated
value, not just the nearest tick — it seeds a fractional position from that
same nearest-point lookup, then refines it with a few Newton iterations
against the coordinate's locally estimated slope:

```python
grid.interp(lat=52.13, lon=4.28)                # method="nearest" or "linear"
```

This is scoped to a **2-D** spanned coordinate (the `lat(y, x)`/`lon(y, x)`
case above) and to `method="nearest"`/`"linear"` — a higher spline order, or
more than 2 spanned dims, isn't implemented. A query outside the grid's
coordinate range, or landing close enough to a fold that the local Jacobian
is itself singular there, raises rather than returning a silently wrong
answer. That detection is local: away from the fold line itself, a query
still resolves to one of the (possibly several) valid preimages — whichever
its nearest-neighbor seed lands closest to — without warning that another
equally-valid one exists. Convergence and singularity checks scale with the
coordinates' own magnitude rather than a fixed absolute unit, so this works
the same whether the coordinate values are e.g. degrees or metres. Gradients
flow through the tensor's own data values, as with any other `interp` call,
but not back through the query point or the coordinate arrays themselves.

## Multiple coordinates per axis

An axis can carry more than one coordinate at once — its own index, plus any
number of **non-dimension coordinates** riding alongside it. Key a coordinate
by any name that isn't a dim, paired with the dim(s) it rides:

```python
sig = xtensor(
    trace, names=("t",),
    coords={
        "t": {"spacing": 0.5, "origin": 0.0},                  # the index
        "season": ("t", ("w", "w", "sp", "sp", "su", "su")),   # rides along "t"
    },
)
sig.sel(t=1.0)        # selects by the index
sig.sel(season="su")  # ValueError: not an index coordinate
```

`.sel`/`.isel` only ever resolve against a dim's own index — a non-dimension
coordinate isn't selectable directly. Promote one to be the index with
`swap_dims`, which renames the axis to the promoted coordinate's name and
keeps the old index around, now riding alongside under its own key:

```python
sig.swap_dims({"t": "season"}).sel(season="su")
```

`swap_dims_` is the in-place variant.
