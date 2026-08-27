"""The core data model: `XTensor` (the named/coordinate-aware tensor
subclass), `Coordinate` (a numeric coordinate, compact or explicit), and
`as_xtensor`, plus every helper that is mutually recursive with them --
coordinate construction, numeric/curvilinear/affine selection, and
interpolation. These stay in one module because `XTensor`'s own bespoke
methods (`sel`, `interp`, ...) call them directly, and they in turn need
`XTensor`/`Coordinate` back (see CLAUDE.md, "Layout").

Every other operator module (`_shape`, `_reduce`, `_slice`, `_combine`,
`_gather`, `_pointwise`) imports `XTensor` from here and registers its
overrides via `XTensor.overrides()`; this module never imports from any
of them.
"""

from __future__ import annotations

# stdlib
import math

# dependencies
import torch
import typing_extensions as tx
from torch import Tensor

# internals
from fiery.xtensor import _arrayutils as arrayutils
from fiery.xtensor import _units
from fiery.xtensor._arrayutils import SmartSlicerT, _SmartSlicerT
from fiery.xtensor._common import (
    AxisT,
    CoordsT,
    LabelsT,
    _carry,
    _either_dict_or_kwargs,
    _expand_name_ellipsis,
    _parse_axes,
    _resolve_axis,
)
from fiery.xtensor._extended import ExtendedTensor
from fiery.xtensor._options import get_option as _get_option
from fiery.xtensor._selection import (
    _check_curvilinear_shape,
    _check_sel_tolerance,
    _check_unambiguous_coord_spec,
    _closed_form_sel_index,
    _ClosedFormMiss,
    _compact_range_slice,
    _explicit_range_slice,
    _is_compact_coord,
    _is_explicit_coord,
    _is_label_index,
    _is_pure_number,
    _label_name,
    _label_unit,
    _match_positions,
    _pack_coord,
    _pick_sel_index,
    _positions_to_index,
    _reconcile_origin_unit,
    _resolve_sel_mode,
    _single_source,
    _slice_labels,
)


class XTensor(ExtendedTensor):
    """
    A tensor with named dimensions and, optionally, per-dimension coordinate
    **labels** -- an [xarray](https://docs.xarray.dev)-like `DataArray` over a
    live `torch.Tensor`.

    - **Dimensions** are named through `names` (self-managed in `_axis_names`,
      independent of PyTorch's experimental builtin named-tensor feature, so
      the class works even where that API has been removed).
    - **Coordinates** label the positions along a named dimension. They live
      in `coords` -- a mapping *dim name -> labels* -- keyed by dimension
      **name**, so they follow their dimension through reshaping/reordering
      with no positional bookkeeping. A labelled dimension must be named.
    - **Axis descriptors** may enrich a name with extra fields -- any custom
      key you like (`type` is the OME-NGFF convention shown in examples;
      `orientation` is the one field with built-in behaviour) -- passed as a
      dict in place of a bare name (`{"name": "x", "type": "space"}`). `names`
      stays the ergonomic view (bare names); `axes` returns the full
      descriptors. The extra fields live in `_axis_meta`, keyed by dimension
      name, so they follow the dimension like coordinates do.

    Select by label with `sel`, by integer position with `isel`, or reach a
    single label by attribute (`x.red`).
    """

    _ATTRS = {
        "_axis_names",
        "_coords",
        "_axis_meta",
        "_data_units",
    }

    def __new__(cls, *args, **kwargs) -> tx.Self:
        # NOTE: remove arguments that `Tensor.__new__` does not support.
        kwargs.pop("names", None)
        kwargs.pop("coords", None)
        kwargs.pop("axes", None)
        kwargs.pop("units", None)
        # Wrapping an existing tensor via `Tensor.__new__(cls, t)` is not
        # portable: some PyTorch versions reject a non-default dtype there
        # (e.g. a Long tensor raises "expected Float"). `as_subclass` re-tags
        # an existing tensor as this subclass across versions without a copy.
        if len(args) == 1 and not kwargs and isinstance(args[0], Tensor):
            return args[0].as_subclass(cls)
        return super().__new__(cls, *args, **kwargs)

    def __init__(self, *args, **kwargs) -> None:
        # NOTE: Tensor does not implement `__init__` (only `__new__`), but we
        # add support for the `names` / `coords` arguments here.
        super().__init__()  # This actually calls `object.__init__`
        # `names=` takes bare strings; `axes=` is the general per-axis
        # container (descriptor dicts: name + coord/labels + free-form fields).
        axes = kwargs.pop("axes", None)
        names = kwargs.pop("names", None)
        coords = kwargs.pop("coords", None)
        units = kwargs.pop("units", None)
        coord_specs = {}
        if axes is not None:
            axis_names, meta, coord_specs = _parse_axes(tuple(axes), self.ndim)
            self._axis_names = axis_names
            self._axis_meta = meta
        if names is not None:
            self.names = names
        if coords is not None:
            # an explicit `coords=` merges onto (and overrides) any coordinates
            # embedded in `axes=` descriptors.
            coord_specs = {**coord_specs, **dict(coords)}
        if coord_specs:
            self.coords = coord_specs
        if units is not None:
            self.units = units

    # -- dimensions --------------------------------------------------------

    @property
    def names(self) -> tuple[str | None, ...]:
        """
        The name of each axis (`None` for unnamed axes). On assignment a single
        `...` expands to a run of unnamed axes, so `x.names = ("b", ..., "w")`
        names only the ends and leaves the middle unnamed.
        """
        names = self.__dict__.get("_axis_names", None)
        # Fall back to all-unnamed if metadata is missing or stale (e.g. it
        # was propagated onto the output of an op that changed the rank but
        # has no name-aware override yet).
        if names is None or len(names) != self.ndim:
            return (None,) * self.ndim
        return names

    @names.setter
    def names(self, value: tx.Optional[tx.Sequence[AxisT]]) -> None:
        if value is None:
            self.__dict__.pop("_axis_names", None)
            self.__dict__.pop("_axis_meta", None)
            return
        value = tuple(value)
        # A single `...` fills the unspecified middle with unnamed axes, so
        # `names=("b", ..., "x")` on a 4-D tensor -> ("b", None, None, "x").
        value = _expand_name_ellipsis(value, self.ndim, (None,) * self.ndim)
        if len(value) != self.ndim:
            raise ValueError(
                f"Expected {self.ndim} names, got {len(value)}: {value}"
            )
        # `names=` takes bare strings (or `None`); richer axis descriptors --
        # `coord`/`labels` and free-form fields -- go through `axes=` instead.
        for item in value:
            if not (item is None or isinstance(item, str)):
                raise TypeError(
                    "names= takes strings (or None); pass a descriptor dict "
                    f"through axes= instead of {item!r}"
                )
        self._axis_names = value

    # -- axis descriptors --------------------------------------------------

    @property
    def axes(self) -> tuple[dict | None, ...]:
        """
        Each axis as a descriptor dict ``{"name": ..., **extra}`` (or `None`
        for an unnamed axis). The extra fields -- any custom key (`type` by
        OME-NGFF convention, `orientation`, ...) -- come from `_axis_meta`,
        keyed by dimension name.
        """
        meta = self._valid_axis_meta()
        return tuple(
            None if name is None else {"name": name, **meta.get(name, {})}
            for name in self.names
        )

    @axes.setter
    def axes(self, value: tx.Optional[tx.Sequence]) -> None:
        if value is None:
            for attr in ("_axis_names", "_axis_meta"):
                self.__dict__.pop(attr, None)
            return
        names, meta, coord_specs = _parse_axes(tuple(value), self.ndim)
        self._axis_names = names
        self._axis_meta = meta
        # A descriptor may embed its coordinate under `coord` (numeric) or
        # `labels` (categorical); apply those, leaving other coords untouched.
        if coord_specs:
            self.coords = coord_specs

    def _valid_axis_meta(self) -> dict[str, dict]:
        """`_axis_meta` filtered to dimensions still named on this tensor."""
        stored = self.__dict__.get("_axis_meta") or {}
        names = self.names
        return {name: extra for name, extra in stored.items() if name in names}

    # -- coordinates -------------------------------------------------------

    @property
    def coords(self) -> dict[str, LabelsT]:
        """
        The coordinates, as a `{dim name: coordinate}` dict. A coordinate is a
        tuple of **labels**, or a compact numeric coordinate (`{spacing[,
        origin]}`, whose `["value"]` key materialises the positions).

        Only entries that are still valid are returned -- every dim a
        coordinate spans must still be named on this tensor (and, for
        labels, its size must match the label count) -- so stale metadata
        propagated onto a shape-changing op is hidden.

        A **dimension** coordinate spans just the dim it is keyed under (it
        *is* that dim's index, so `.sel(name=...)` works); a
        **non-dimension** coordinate (its key is not itself a dim name)
        rides along some other dim(s) instead, and is not an index. A
        non-dimension coordinate may span **several** dims at once: a
        compact **affine** map (`spacing` is a vector, one component per
        spanned dim, `origin` a single scalar shared across them), or an
        explicit **grid** of values with no regular spacing (e.g. `lat(y,
        x)`), one tensor axis per spanned dim.
        """
        names = self.names
        valid = {}
        stored = self.__dict__.get("_coords") or {}
        for name, (dims, coord) in stored.items():
            if any(dim not in names for dim in dims):
                continue
            if len(dims) > 1:
                # The grid is laid out in **this tensor's** axis order, not in
                # `dims` order: the two differ when `dims` was given in
                # another order, or once an axis-reordering op (`permute` /
                # `transpose` / `movedim`) has moved them, and `["value"]` is
                # a bare array with no dims of its own -- so materialising in
                # `dims` order would silently misalign it with the data.
                axes = [names.index(dim) for dim in dims]
                order = sorted(range(len(dims)), key=axes.__getitem__)
                if isinstance(coord, Coordinate) and coord._compact():
                    valid[name] = coord._bound_axes(
                        tuple(
                            (component, self.shape[axes[component]])
                            for component in order
                        )
                    )
                elif isinstance(coord, Coordinate):
                    # explicit curvilinear array: it does not itself update
                    # when a spanned dim is sliced/narrowed (like a 1-D
                    # explicit non-dimension coordinate, it rides through a
                    # slice unchanged and is only kept here if its shape still
                    # matches) -- so it is dropped, not resliced, once any
                    # spanned dim's size has moved on without it.
                    raw = dict.__getitem__(coord, "value")
                    if tuple(raw.shape) == tuple(
                        self.shape[axes[i]] for i in range(len(dims))
                    ):
                        valid[name] = coord._bound_curvilinear(order)
                continue
            size = self.shape[names.index(dims[0])]
            if isinstance(coord, Coordinate):
                if coord._compact():
                    valid[name] = coord._bound(size)
                elif len(dict.__getitem__(coord, "value")) == size:
                    valid[name] = coord  # explicit: kept if length matches
            elif len(coord) == size:
                valid[name] = coord
        return valid

    @coords.setter
    def coords(self, value: tx.Optional[CoordsT]) -> None:
        if value is None:
            self.__dict__.pop("_coords", None)
            return
        names = self.names
        unified = {}
        for key, spec in dict(value).items():
            if key not in names:
                # a coord keyed by a non-axis name is a **non-dimension**
                # coordinate (Proposal 0005): given as `(dim, values)`, it
                # rides along `dim` rather than indexing it -- or, spanning
                # several dims, `(dims, {spacing, ...})` is a compact
                # **affine** coordinate (step 3); that form materialises and
                # re-slices exactly, so (unlike a single-dim non-dimension
                # coordinate) it has no fixed "length" to check on input.
                dims, coord = _parse_nondim_coord(key, spec, names)
                if len(dims) == 1:
                    size = self.shape[names.index(dims[0])]
                    _check_nondim_len(key, dims[0], coord, size)
                elif isinstance(coord, Coordinate) and not coord._compact():
                    _check_curvilinear_shape(
                        key, coord, dims, self.shape, names
                    )
                unified[key] = dims, coord
                continue
            if _is_compact_coord(spec) or _is_explicit_coord(spec):
                coord = _make_coordinate(spec)
                if not coord._compact():
                    size = self.shape[names.index(key)]
                    length = _nondim_coord_len(coord)
                    if length != size:
                        raise ValueError(
                            f"coords: dim {key!r} has {length} values "
                            f"for size {size}"
                        )
                unified[key] = _pack_coord(key, coord)
                continue
            size = self.shape[names.index(key)]
            labels = tuple(spec)
            # a bare sequence of plain numbers is a numeric coordinate, not
            # labels that happen to be numbers (issue #107) -- auto-promote
            # it through the same explicit-coordinate path a tensor spec
            # already takes, so it gets real `.sel` support (mode/tolerance/
            # units) instead of becoming a silently uncomparable label. The
            # length check applies here too -- promotion must not bypass the
            # #95/#97 validation a plain label sequence already gets below.
            promoted = _promote_numeric_labels(key, labels)
            if isinstance(promoted, Coordinate):
                if len(labels) != size:
                    raise ValueError(
                        f"coords: dim {key!r} has {len(labels)} values "
                        f"for size {size}"
                    )
                unified[key] = _pack_coord(key, promoted)
                continue
            # `...` fills the middle with unlabelled positions.
            if Ellipsis in labels:
                labels = tuple(arrayutils._unroll(labels, size))
            if len(labels) != size:
                raise ValueError(
                    f"coords: dim {key!r} has {len(labels)} labels "
                    f"for size {size}"
                )
            unified[key] = _pack_coord(key, labels)
        self._coords = unified

    # -- data unit ---------------------------------------------------------

    @property
    def units(self) -> tx.Optional[str]:
        """
        The physical unit of the tensor's **values** (the *data* unit), or
        `None`. Assigning *annotates* (it never changes the data);
        `to_units` converts. Under `unit_backend="pint"` the unit is
        validated and normalised on set; with the default
        `unit_backend=None` it is an opaque string that is simply carried
        through operations.
        """
        return self.__dict__.get("_data_units")

    @units.setter
    def units(self, value: tx.Optional[str]) -> None:
        if value is None:
            self.__dict__.pop("_data_units", None)
            return
        self._data_units = _units.normalise(value)

    @property
    def dimensionality(self) -> str:
        """
        The physical dimensionality of this tensor's unit (e.g. `"[length]"`,
        `"[mass] * [length] ** 2 / [time] ** 3"`), or `""` if there's no unit
        at all. An explicitly dimensionless unit (`.units == ""`) has its own
        non-empty dimensionality string (`"dimensionless"`) -- see
        `.dimensionless`/`.unitless` for the boolean questions. Requires
        `unit_backend="pint"`.
        """
        return _units.dimensionality(self.units)

    @property
    def dimensionless(self) -> bool:
        """
        Whether this tensor's unit is dimensionless (or unset). With no unit
        backend this is `not bool(self.units)` -- `True` for no unit or an
        explicitly empty one, `False` for any opaque unit string, since
        there's no dimensionality system to consult otherwise.
        """
        return _units.dimensionless(self.units)

    @property
    def unitless(self) -> bool:
        """
        Whether this tensor has literally **no unit at all** -- a stricter
        check than `dimensionless`. Angle units are the case that splits
        them: `xtensor(x, units="rad").dimensionless` is `True` (an angle is
        dimensionally trivial) but `unitless` is `False` (it still names a
        unit). With no backend the two coincide exactly
        (`not bool(self.units)` for both), since there's no dimensionality
        system to tell them apart. Requires `unit_backend="pint"` for the two
        to actually differ.
        """
        return _units.unitless(self.units)

    def is_compatible_with(self, unit: str) -> bool:
        """
        Whether this tensor's unit shares a dimensionality with `unit` (so
        `to_units(unit)` would succeed). `False` if this tensor has no unit.
        Requires `unit_backend="pint"`.
        """
        current = self.units
        if current is None:
            return False
        return _units.compatible(current, _units.normalise(unit))

    def to_units(self, unit: str) -> tx.Self:
        """
        Convert the data to `unit`, rescaling the values by the conversion
        factor (requires a unit already set and `unit_backend="pint"`).
        """
        current = self.units
        if current is None:
            raise ValueError("to_units: this tensor has no unit to convert")
        unit = _units.normalise(unit)
        scaled = Tensor.mul(self, _units.factor(current, unit))
        return _carry(self, scaled, _data_units=unit)

    def to_units_(self, unit: str) -> tx.Self:
        """
        Convert to `unit` **in place** -- rescales the data and updates the
        unit annotation on `self`, returning `self`. Same restrictions as any
        other in-place op (`mul_`, `add_`, ...): raises on a leaf tensor that
        requires grad, and raises if the scale factor can't be applied
        without changing dtype. Requires a unit already set and
        `unit_backend="pint"`.
        """
        current = self.units
        if current is None:
            raise ValueError("to_units_: this tensor has no unit to convert")
        unit = _units.normalise(unit)
        # Rescale first: nothing above this line has touched `self`, so a
        # failure (bad unit, incompatible dimensions, a grad-requiring leaf,
        # an integer dtype) leaves the annotation as it was.
        self.mul_(_units.factor(current, unit))
        self._data_units = unit
        return self

    @property
    def magnitude(self) -> tx.Self:
        """
        The tensor with its **data unit dropped** -- the bare values, still
        an `XTensor` with the same names and coordinates. A view (no data
        copy); the original is unchanged. `x.magnitude.units` is always
        `None`. (To get a plain `torch.Tensor`, use
        `x.as_subclass(torch.Tensor)`.)

        This drops the *unit annotation* -- it is not the mathematical
        modulus. For the absolute value of a (possibly complex) tensor, use
        `x.abs()` / `torch.abs(x)`.
        """
        return _carry(self, self.as_subclass(type(self)), _data_units=None)

    def m_as(self, unit: str) -> tx.Self:
        """
        Convert to `unit` and drop the annotation in one step -- sugar for
        `x.to_units(unit).magnitude`. Still an `XTensor` (see `magnitude`'s
        own note on getting a plain `torch.Tensor`).
        """
        return self.to_units(unit).magnitude

    # `.m`/`.u` (pint's own short aliases for `.magnitude`/`.units`) were
    # deliberately not added: they would silently shadow a coordinate label
    # named "m"/"u" (`__getattr__`'s label lookup never runs once a real
    # attribute of that name exists) -- and single-letter physics variable
    # names (velocity components `u`/`v`/`w`, mass `m`) are exactly the kind
    # of label this library expects to see.

    # -- unit simplification (Proposal 0006 §2.7) --------------------------
    #
    # Each of these asks the backend which unit to land in, then reuses the
    # ordinary `to_units` / `to_units_` conversion. `to_compact` is the one
    # whose answer depends on the *values* (it picks the prefix that keeps
    # them near 1), so it feeds the backend a representative magnitude.

    def _simplification_target(
        self, name: str, operation: tx.Callable, **kwargs: tx.Any
    ) -> str:
        """The unit one of the `to_*` simplifications should convert to."""
        current = self.units
        if current is None:
            raise ValueError(f"{name}: this tensor has no unit to convert")
        return operation(current, **kwargs)

    def _compact_target(self, name: str) -> str:
        # Unlike the other three, the compact unit depends on how big the
        # values actually are, so the backend is given the largest magnitude
        # present as a stand-in for the whole tensor. A NaN/inf value must
        # not veto the whole tensor's answer -- only fall back to 1.0 when
        # there is no finite value at all (an empty tensor, or all NaN/inf).
        # There is no torch.nanmax across the torch versions this library
        # supports, so the finite values are masked out by hand -- an O(n)
        # pass that allocates a boolean mask plus a filtered copy, on top of
        # the abs()/detach() copies already taken. Worth knowing before
        # calling `to_compact`/`to_compact_` on a large tensor, despite the
        # name suggesting something cheap.
        magnitude = 1.0
        if self.numel():
            values = self.as_subclass(Tensor).detach().abs()
            finite = values[torch.isfinite(values)]
            if finite.numel():
                magnitude = finite.max().item()
        return self._simplification_target(
            name, _units.compact_units, magnitude=magnitude
        )

    def to_base_units(self) -> tx.Self:
        """
        Convert to the backend's base units (SI base units under pint).
        Requires a unit already set and `unit_backend="pint"`.
        """
        return self.to_units(
            self._simplification_target("to_base_units", _units.base_units)
        )

    def to_base_units_(self) -> tx.Self:
        """In-place variant of `to_base_units`."""
        return self.to_units_(
            self._simplification_target("to_base_units_", _units.base_units)
        )

    def to_reduced_units(self) -> tx.Self:
        """
        Convert to the backend's reduced (simplified) form of this unit.
        Requires a unit already set and `unit_backend="pint"`.
        """
        return self.to_units(
            self._simplification_target(
                "to_reduced_units", _units.reduced_units
            )
        )

    def to_reduced_units_(self) -> tx.Self:
        """In-place variant of `to_reduced_units`."""
        return self.to_units_(
            self._simplification_target(
                "to_reduced_units_", _units.reduced_units
            )
        )

    def to_compact(self) -> tx.Self:
        """
        Convert to the unit these values read most compactly in -- the prefix
        that keeps them near 1 (`200e-9 s` becomes `200 ns`), picked from the
        largest magnitude present. Requires a unit already set and
        `unit_backend="pint"`. Unlike the other three simplifications, this
        one has to look at the data itself (not just the unit), which costs
        an extra full pass over the tensor -- worth knowing before calling it
        in a hot loop over a large tensor.
        """
        return self.to_units(self._compact_target("to_compact"))

    def to_compact_(self) -> tx.Self:
        """In-place variant of `to_compact`."""
        return self.to_units_(self._compact_target("to_compact_"))

    def to_preferred(
        self, preferred_units: tx.Optional[tx.List[str]] = None
    ) -> tx.Self:
        """
        Convert to whichever unit the backend's preferred-units logic picks.
        `preferred_units` is a list of unit strings to guide it; omit it to
        use the backend registry's own default, which raises if none is
        configured. Requires a unit already set and `unit_backend="pint"`.

        ```python
        force.to_preferred(["N"])          # 5000 g·mm/s² -> 0.005 N
        ```
        """
        return self.to_units(
            self._simplification_target(
                "to_preferred",
                _units.preferred_units,
                preferred=preferred_units,
            )
        )

    def to_preferred_(
        self, preferred_units: tx.Optional[tx.List[str]] = None
    ) -> tx.Self:
        """In-place variant of `to_preferred`."""
        return self.to_units_(
            self._simplification_target(
                "to_preferred_",
                _units.preferred_units,
                preferred=preferred_units,
            )
        )

    # -- dtype / device / metadata conversion (Proposal 0006 §2.5-2.6) -----

    def _parse_to_units(
        self, name: str, args: tuple, kwargs: dict, units: tx.Any
    ) -> tx.Tuple[tuple, tx.Any]:
        """
        Split `.to()`/`.to_()`'s arguments into `(args, unit)`: a lone
        positional backend `Unit`/`Quantity` is sugar for `units=`, and a
        backend object given either way reduces to its unit string. Other
        `.to()` keywords (`copy=`, `non_blocking=`, `memory_format=`) still
        apply alongside it -- once `args[0]` is recognised as a unit there is
        no ambiguity left for them to create, so they simply pass through to
        the dtype/device call unchanged.
        """
        if (
            args
            and _units.is_unit_like(args[0])
            and units is arrayutils._UNSET
            and len(args) == 1
        ):
            units, args = args[0], ()
        if units is None:
            # `units=` converts, and there is no such thing as converting
            # *to* no unit -- clearing an annotation is a different verb.
            raise ValueError(
                f"{name}: units=None is not a conversion target; assign "
                "`x.units = None` to clear a tensor's unit instead"
            )
        if _units.is_unit_like(units):
            _, units = _units.split_quantity(units)
        return args, units

    def to(
        self,
        *args: tx.Any,
        units: tx.Any = arrayutils._UNSET,
        names: tx.Any = arrayutils._UNSET,
        coords: tx.Any = arrayutils._UNSET,
        **kwargs: tx.Any,
    ) -> tx.Self:
        """
        Same as `torch.Tensor.to` (dtype/device, positional or keyword), plus
        `units=`/`names=`/`coords=` overrides.

        ```python
        x.to(torch.float64)              # exactly as before
        x.to(units="mm")                 # convert the data unit
        x.to(ureg.mm)                    # ... same, from a backend unit
        x.to(torch.float64, names=("b", "t"))
        ```

        `units=` **converts** (like `to_units`, so a unit must already be
        set), matching what `.to()` means everywhere else -- unlike
        `as_xtensor(x, units=...)`, which *annotates*. `names=`/`coords=` are
        the instance-method form of
        [`as_xtensor`][fiery.xtensor.as_xtensor]'s overrides of the same
        name, and replace wholesale. A bare positional backend
        `Unit`/`Quantity` is sugar for `units=` with that same object; unlike
        the positional form, `units=` also accepts a plain unit string (a
        positional string stays `torch.Tensor.to`'s own device spelling,
        `"cuda"`/`"cpu"`).
        """
        args, units = self._parse_to_units("to", args, kwargs, units)
        result = Tensor.to(self, *args, **kwargs)
        if (
            units is arrayutils._UNSET
            and names is arrayutils._UNSET
            and coords is arrayutils._UNSET
        ):
            return result
        if units is not arrayutils._UNSET:
            result = result.to_units(units)
        return as_xtensor(result, names=names, coords=coords)

    def to_(
        self,
        *args: tx.Any,
        units: tx.Any = arrayutils._UNSET,
        names: tx.Any = arrayutils._UNSET,
        coords: tx.Any = arrayutils._UNSET,
        **kwargs: tx.Any,
    ) -> tx.Self:
        """
        In-place `.to()`, with the same `units=`/`names=`/`coords=`
        overrides, each going through an already in-place path -- but "in
        place" doesn't mean "always succeeds": a dtype/device change raises
        unless the request already matches this tensor's current dtype and
        device (a no-op), even if forced through `copy=True`; `units=`
        inherits `to_units_`'s own restrictions (no unit set, or a leaf
        tensor requiring grad); `names=`/`coords=` inherit their setters'
        own validation (a length mismatch, an invalid coordinate spec).
        Applied `names=`, then `coords=`, then `units=` last -- each of the
        first two either fully applies or raises before mutating anything,
        so the data rescale from `units=` never happens unless every other
        override already succeeded.
        """
        args, units = self._parse_to_units("to_", args, kwargs, units)
        result = Tensor.to(self, *args, **kwargs)
        if result.dtype != self.dtype or result.device != self.device:
            raise ValueError(
                "to_: in-place .to() cannot change dtype or device "
                f"(would produce dtype={result.dtype}, "
                f"device={result.device})"
            )
        if names is not arrayutils._UNSET:
            self.names = names
        if coords is not arrayutils._UNSET:
            self.coords = coords
        if units is not arrayutils._UNSET:
            self.to_units_(units)
        return self

    # -- attaching a unit by multiplication (Proposal 0003 §2.4) -----------
    #
    # `x * u.mm` / `x / u.s`: a backend `Unit`/`Quantity` operand attaches or
    # derives a data unit. This must be caught at the operator dunder, because
    # Python's protocol otherwise lets the unit library's reflected `__rmul__`
    # intercept `x * <unit>` first (yielding a wrapped object, never an
    # `XTensor`). A non-unit operand falls straight back to the normal path,
    # so name/unit algebra for ordinary operands is untouched.

    def __mul__(self, other: tx.Any) -> tx.Any:
        if _units.is_unit_like(other):
            return _attach_unit(self, other, "mul")
        return Tensor.__mul__(self, other)

    def __rmul__(self, other: tx.Any) -> tx.Any:
        if _units.is_unit_like(other):
            return _attach_unit(self, other, "mul")
        return Tensor.__rmul__(self, other)

    def __truediv__(self, other: tx.Any) -> tx.Any:
        if _units.is_unit_like(other):
            return _attach_unit(self, other, "div")
        return Tensor.__truediv__(self, other)

    def __rtruediv__(self, other: tx.Any) -> tx.Any:
        # `unit / x` is normally handled by the unit library itself before we
        # are consulted; this only fires for e.g. a scalar left operand.
        return Tensor.__rtruediv__(self, other)

    # -- renaming ----------------------------------------------------------

    def _resolve_rename(
        self, names: tuple, rename_map: dict
    ) -> tuple[str | None, ...]:
        """Compute the new axis-name tuple for `rename` / `rename_`."""
        if rename_map:
            if names:
                raise ValueError(
                    "rename: cannot mix positional names and a rename map"
                )
            current = list(self.names)
            for old, new in rename_map.items():
                if old not in current:
                    raise ValueError(
                        f"rename: no axis named {old!r} in {tuple(current)}"
                    )
                current[current.index(old)] = new
            return tuple(current)
        if len(names) == 1 and names[0] is None:
            return (None,) * self.ndim
        if len(names) == 1 and isinstance(names[0], (tuple, list)):
            names = tuple(names[0])
        # A single `...` keeps the axes it spans unchanged (`rename` modifies,
        # so an unspecified run is left as-is, not unnamed).
        new_names = _expand_name_ellipsis(
            tuple(names), self.ndim, tuple(self.names)
        )
        if len(new_names) != self.ndim:
            raise ValueError(
                f"rename: expected {self.ndim} names, got {len(new_names)}"
            )
        return new_names

    def _remap_named(self, attr: str, new_names: tuple) -> dict:
        """Re-key a `{dim name: ...}` dict attribute from current names."""
        stored = self.__dict__.get(attr) or {}
        if not stored:
            return {}
        remapped = {}
        for old, new in zip(self.names, new_names):
            if old in stored and new is not None:
                remapped[new] = stored[old]
        return remapped

    def _remap_coords(self, new_names: tuple) -> dict:
        """
        Coordinates re-keyed from the current names to `new_names`. A
        **dimension** coordinate (its key is one of `self.names`) is re-keyed
        like the axis, and dropped if that axis is unnamed; a
        **non-dimension** coordinate (Proposal 0005 -- its key is its own
        name, not a dim) keeps its key. Either way, every dim in its `dims`
        is remapped the same way, and the coordinate drops if any of them is
        unnamed.
        """
        stored = self.__dict__.get("_coords") or {}
        if not stored:
            return {}
        current = self.names
        rename_of = {
            old: new for old, new in zip(current, new_names) if new is not None
        }
        remapped = {}
        for old_key, (dims, coord) in stored.items():
            if old_key in current:
                new_key = rename_of.get(old_key)
                if new_key is None:
                    continue
            else:
                new_key = old_key
            if any(dim not in rename_of for dim in dims):
                continue
            if new_key in remapped:
                raise ValueError(
                    f"rename: coordinate name collision on {new_key!r} "
                    "(a renamed axis now matches an existing coordinate's "
                    "name); choose a name that doesn't collide"
                )
            new_dims = tuple(rename_of[dim] for dim in dims)
            if new_key in new_names and new_dims != (new_key,):
                # Renaming an axis onto a **multi-dim** coordinate's key would
                # leave an entry whose key *is* a dim but which is not that
                # dim's index -- breaking the `dims == (name,)` <=> dimension
                # coordinate invariant every consumer relies on (`sel`,
                # `__getitem__`'s dimension-coordinate pass, `flip`/`roll`
                # would then treat the vector `spacing` as a 1-D one and
                # corrupt it). Refuse, like any other name collision.
                raise ValueError(
                    f"rename: coordinate name collision on {new_key!r} "
                    "(a renamed axis now matches a multi-dim coordinate's "
                    f"name, which spans {new_dims}); choose a name that "
                    "doesn't collide"
                )
            remapped[new_key] = (new_dims, coord)
        return remapped

    def rename(self, *names: str | None, **rename_map: str) -> tx.Self:
        """
        Return a view with renamed axes (self-managed; not the builtin op).

        Call positionally (`x.rename("a", "b")`), with `None` to clear all
        names (`x.rename(None)`), or with a mapping to rename specific axes
        (`x.rename(old="new")`). A single `...` keeps the axes it spans
        unchanged (`x.rename("A", ..., "Z")`). Coordinates follow their
        (renamed) dimension.
        """
        new_names = self._resolve_rename(names, rename_map)
        # `as_subclass` returns a view but does not copy `__dict__`, so carry
        # the subclass metadata over explicitly.
        out = self.as_subclass(type(self))
        out.__dict__.update(self.__dict__)
        out._coords = self._remap_coords(new_names)
        out._axis_meta = self._remap_named("_axis_meta", new_names)
        out._axis_names = new_names
        return out

    def rename_(self, *names: str | None, **rename_map: str) -> tx.Self:
        """In-place variant of `rename`."""
        new_names = self._resolve_rename(names, rename_map)
        coords = self._remap_coords(new_names)
        meta = self._remap_named("_axis_meta", new_names)
        self._coords = coords
        self._axis_meta = meta
        self._axis_names = new_names
        return self

    def _swap_dims_state(self, dims_map: dict) -> tuple:
        """
        Validate a `swap_dims` mapping and compute its `(new_names,
        new_coords)`. Each `{old_dim: new_name}` pair promotes `new_name` --
        an existing non-dimension coordinate riding `old_dim` **alone** -- to
        be `old_dim`'s replacement index, renaming the axis to `new_name` in
        the process (xarray's `swap_dims`).

        This is *not* a rename with extra steps: `rename` would re-key
        `old_dim`'s own dimension coordinate onto `new_name` too, colliding
        with the very coordinate being promoted (the same collision `rename`
        already raises on). `swap_dims` never re-keys a coordinate -- it only
        remaps `dims` tuples through the axis substitution, exactly like
        `rename` does for a coordinate's `dims` -- so which entry counts as
        *the* dimension coordinate falls out structurally afterwards (`dims
        == (key,)`): `new_name`'s entry becomes `(new_name,)` (now the
        index), and `old_dim`'s former entry becomes `(new_name,)` too but
        keyed `old_dim` (now a rider, since its key no longer matches its own
        `dims`) -- exactly xarray's "old index survives under its old name,
        riding the renamed axis" behaviour.
        """
        names = self.names
        stored = self.__dict__.get("_coords") or {}
        for old_dim, new_name in dims_map.items():
            if old_dim not in names:
                raise ValueError(f"swap_dims: no axis named {old_dim!r}")
            if new_name in names and new_name != old_dim:
                raise ValueError(
                    f"swap_dims: {new_name!r} is already an axis name"
                )
            entry = stored.get(new_name)
            if entry is None or entry[0] != (old_dim,):
                raise ValueError(
                    f"swap_dims: {new_name!r} must be an existing "
                    f"non-dimension coordinate riding {old_dim!r} alone, "
                    "to be promoted to its index"
                )
        new_names = tuple(dims_map.get(n, n) for n in names)
        seen = [n for n in new_names if n is not None]
        if len(set(seen)) != len(seen):
            raise ValueError(
                "swap_dims: the result would have duplicate axis names "
                f"{tuple(new_names)}"
            )
        new_coords = {
            key: (tuple(dims_map.get(d, d) for d in dims), coord)
            for key, (dims, coord) in stored.items()
        }
        return new_names, new_coords

    def swap_dims(
        self, dims_map: tx.Optional[dict] = None, **kwargs: str
    ) -> tx.Self:
        """
        Promote a non-dimension coordinate to be its dim's index, demoting
        the previous index to ride along under its old key -- xarray's
        `swap_dims`. `{old_dim: new_name}` (positionally or as keywords):
        `new_name` must already be a non-dimension coordinate riding
        `old_dim` alone (`coords={..., new_name: (old_dim, values)}`).

        ```python
        da.swap_dims({"time": "label"}).sel(label="c")   # promote, then select
        ```

        The axis itself is renamed `old_dim -> new_name` (so `.names` and any
        axis descriptor follow, like [`rename`][fiery.xtensor.XTensor.rename]);
        every other coordinate riding `old_dim` keeps its own key and simply
        rides the renamed axis.
        """
        mapping = dict(dims_map or {})
        mapping.update(kwargs)
        if not mapping:
            return self
        new_names, new_coords = self._swap_dims_state(mapping)
        out = self.as_subclass(type(self))
        out.__dict__.update(self.__dict__)
        out._coords = new_coords
        out._axis_meta = self._remap_named("_axis_meta", new_names)
        out._axis_names = new_names
        return out

    def swap_dims_(
        self, dims_map: tx.Optional[dict] = None, **kwargs: str
    ) -> tx.Self:
        """In-place variant of `swap_dims`."""
        mapping = dict(dims_map or {})
        mapping.update(kwargs)
        if not mapping:
            return self
        new_names, new_coords = self._swap_dims_state(mapping)
        self._coords = new_coords
        self._axis_meta = self._remap_named("_axis_meta", new_names)
        self._axis_names = new_names
        return self

    # -- indexing / selection ---------------------------------------------

    def _resolve_label_slicer(self, slicer: SmartSlicerT) -> SmartSlicerT:
        # Resolve any *positional* coordinate-label index to an integer (or a
        # list of integers) against the axis it sits on, so `x[..., "y", "z"]`
        # addresses the last two axes by label. A bare `str` or list-of-`str`
        # element is a label index; everything else is left untouched.
        items = slicer if isinstance(slicer, tuple) else (slicer,)
        if not any(_is_label_index(v) for v in items):
            return slicer
        # Axes consumed by the explicit (non-newaxis, non-ellipsis) items; the
        # ellipsis, if any, fills the remaining axes in the middle. A label
        # index (str / list-of-str / query dict) consumes exactly one axis.
        consumed = sum(
            1 if _is_label_index(v) else arrayutils._count_input_axes((v,))
            for v in items
            if v is not None and v is not ...
        )
        gap = self.ndim - consumed
        resolved, axis = [], 0
        for value in items:
            if value is ...:
                axis += gap
                resolved.append(value)
            elif value is None:
                resolved.append(value)
            elif _is_label_index(value):
                resolved.append(self._label_to_index(axis, value))
                axis += 1
            else:
                resolved.append(value)
                axis += 1
        return tuple(resolved)

    def _label_to_index(self, axis: int, value: tx.Any) -> tx.Any:
        """
        Resolve a positional label index against `axis`:

        - a `str` -> the integer position whose label **identity** matches
          (drops the axis, like an int);
        - a list of `str` -> the list of such positions;
        - a `dict` -> a *query* over structured labels, giving a `slice`
          (contiguous) or index list of the matches (keeps the axis).

        Raises if the axis is unlabelled or a named label is absent.
        """
        name = self.names[axis % self.ndim]
        labels = self.coords.get(name) if name is not None else None
        if labels is None:
            raise KeyError(
                f"axis {name!r} has no coordinates for label {value!r}"
            )
        if isinstance(value, dict):
            return _positions_to_index(_match_positions(labels, value))

        identities = [_label_name(label) for label in labels]

        def _one(label: str) -> int:
            try:
                return identities.index(label)
            except ValueError:
                raise KeyError(
                    f"no label {label!r} on axis {name!r}"
                ) from None

        if isinstance(value, str):
            return _one(value)
        return [_one(label) for label in value]

    def __getitem__(self, slicer: SmartSlicerT) -> tx.Self:
        # A positional coordinate label (`x[..., "y"]`) resolves to an integer
        # index against the axis it indexes before ordinary indexing runs.
        slicer = self._resolve_label_slicer(slicer)
        # A 0-D integer tensor index (`x[torch.tensor(1)]`) behaves exactly
        # like the plain `int` it's equivalent to; normalising it up front
        # means the slicer-classification helpers below never have to
        # special-case a tensor with no `len()`.
        slicer = arrayutils._normalize_scalar_tensor_index(slicer)
        # The underlying tensor carries no builtin names, so basic indexing
        # (including newaxis via `None`) works directly.
        out = Tensor.__getitem__(self, slicer)
        # Re-tag as this subclass: normal dispatch preserves it, but when
        # __getitem__ is reached from inside a `_no_dispatch` override (e.g.
        # `select` / `narrow` / `split`) the raw op returns a plain Tensor.
        if type(out) is not type(self):
            out = out.as_subclass(type(self))
        # Map each output axis back to its source axis to carry names across.
        # A single advanced index reports its source as a length-1 tuple; a
        # broadcast of several input axes reports a longer tuple (ambiguous ->
        # unnamed).
        in_names = self.names
        axis_map = arrayutils._map_axes(slicer, self.ndim)
        sources = [_single_source(src) for src in axis_map]
        out_names = tuple(
            in_names[src] if src is not None else None for src in sources
        )
        out._axis_names = out_names
        # Carry every coordinate through by default (dimension or
        # non-dimension, unified `{name: (dims, coord)}` storage, Proposal
        # 0005): the loop below overwrites each *surviving* axis' own
        # **dimension** coordinate with its properly sliced value (or drops
        # it if the slicer can't be applied). A **non-dimension** coordinate
        # is never explicitly re-sliced here -- it rides through unchanged,
        # and the `coords` property's own dim/size validation drops it once
        # the dim it rides on is removed or resized.
        stored = self.__dict__.get("_coords") or {}
        if stored:
            unrolled = arrayutils._unroll_slicer(slicer, self.ndim)
            new_stored = dict(stored)
            for out_axis, src in enumerate(sources):
                name = out_names[out_axis]
                if src is None or name is None:
                    continue
                in_name = in_names[src]
                entry = stored.get(in_name)
                if entry is None:
                    continue
                _, coord = entry
                piece = arrayutils._get_slicer_by_index(unrolled, src)
                if isinstance(coord, Coordinate):
                    adjusted = _slice_coordinate(coord, piece, self.shape[src])
                    if adjusted is not None:
                        new_stored[name] = (name,), adjusted
                    else:
                        new_stored.pop(in_name, None)
                else:
                    sliced = _slice_labels(coord, piece)
                    if sliced is not None:
                        new_stored[name] = (name,), tuple(sliced)
                    else:
                        new_stored.pop(in_name, None)
            # A compact non-dimension coordinate -- one whose key is not the
            # dim it rides on, so the loop above never looks it up -- *is*
            # explicitly re-sliced when it spans a multi-dim compact
            # **affine** coordinate (Proposal 0005 step 3): exact per-
            # component, like a dimension coordinate's own basic-slice
            # update, just applied once per spanned dim.
            for key, (dims, coord) in stored.items():
                if len(dims) == 1 and dims[0] == key:
                    continue  # a dimension coordinate; handled above
                if not (isinstance(coord, Coordinate) and coord._compact()):
                    continue  # labels / explicit: ride through unchanged
                if any(dim not in in_names for dim in dims):
                    continue  # already invalid; the coords property drops it
                pieces = {}
                sizes = {}
                for dim in dims:
                    src = in_names.index(dim)
                    pieces[dim] = arrayutils._get_slicer_by_index(
                        unrolled, src
                    )
                    sizes[dim] = self.shape[src]
                result = _slice_affine_coordinate(coord, dims, pieces, sizes)
                if result is None:
                    new_stored.pop(key, None)
                else:
                    new_stored[key] = result
            # Selecting a single position on a unit-carrying axis collapses
            # that axis away; its per-position data unit folds into the base
            # data unit (effective unit = base * product of coord units).
            if _units.active():
                folded = self.__dict__.get("_data_units")
                kept = {src for src in sources if src is not None}
                changed = False
                for ax, in_name in enumerate(in_names):
                    if ax in kept or in_name is None:
                        continue
                    piece = arrayutils._get_slicer_by_index(unrolled, ax)
                    if not isinstance(piece, int):
                        continue
                    entry = stored.get(in_name)
                    labels = (
                        entry[1]
                        if entry is not None
                        and not isinstance(entry[1], Coordinate)
                        else None
                    )
                    unit = _label_unit(labels[piece]) if labels else None
                    if unit is not None:
                        folded = _units.mul(folded, unit)
                        changed = True
                if changed:
                    out._data_units = folded
            out._coords = new_stored
        return out

    def isel(self, **indexers: tx.Any) -> tx.Self:
        """
        Select by integer position along **named** dimensions.

        `x.isel(row=0, col=slice(1, 3))` indexes `row` at position 0 and `col`
        at positions 1..2, leaving the other axes untouched.
        """
        slicer = [slice(None)] * self.ndim
        for name, index in indexers.items():
            slicer[_resolve_axis(self.names, name)] = index
        return self[tuple(slicer)]

    def sel(
        self,
        *indexers_positional: tx.Mapping,
        mode: tx.Optional[str] = None,
        tolerance: tx.Any = None,
        method: tx.Optional[str] = None,
        **indexers_kwargs: tx.Any,
    ) -> tx.Self:
        """
        Select by coordinate **label** (or numeric value) along named dims.

        `x.sel(channel="red")` selects the position whose label is `"red"`. A
        list of labels selects several positions; a single label drops the
        dimension (like integer indexing). For **structured** coordinates, a
        `str` matches a label's `"name"`, and a **dict** queries the labels'
        fields (`x.sel(channel={"type": "signal"})`), keeping the axis and
        selecting every match.

        On a **numeric** coordinate, the selector is a value (`x.sel(t="2s")`).
        `mode` chooses which tick an inexact value snaps to:

        - `"round"` *(default)* — the nearest tick by value;
        - `"floor"` / `"ceil"` — the largest tick `<=` / smallest tick `>=`
          the value (**value** space, robust to a descending coordinate);
        - `"prev"` / `"next"` — the neighbouring tick at the lower / higher
          **index** (tick order; needs a monotonic coordinate).

        `tolerance` (a value in the position unit) caps the allowed gap. A
        **bare** `.sel(t=v)` is **exact** (`tolerance=0`); passing a `mode`
        implies an unbounded snap unless a `tolerance` is given.

        A **`slice(lo, hi)`** on a numeric coordinate is a **value range**,
        unit-aware, resolving to a contiguous integer `slice` — half-open
        like ordinary Python indexing (`lo <= value < hi`), **not** xarray's
        inclusive-both-ends convention (see the "Differences from xarray"
        guide). Bounds are compared numerically regardless of order or of
        the coordinate's own direction: `t=slice(1, 5)` and `t=slice(5, 1)`
        select the same range. A one-sided range keeps the bound in the
        slot it was given (`slice(1, None)` -> `value >= 1`; `slice(None,
        5)` -> `value < 5`); an out-of-range or empty result is a
        well-formed empty axis, not an error. `slice.step` is not supported
        (`mode`/`tolerance` don't apply to a range either).

        A **joint query over several dims sharing one coordinate** (e.g. a
        `lat`/`lon` pair that together locate a point on a 2-D grid) picks
        all of those dims' positions in one shot: pass a value for *every*
        coordinate name that spans the same `dims` (`x.sel(lat=52.1,
        lon=4.3)`) — no dedicated syntax, ordinary keyword arguments that
        happen to share `dims` are recognised as one joint system. Only a
        **square** query is supported (exactly one coordinate value per
        spanned dim); an under- or over-determined query raises rather than
        guessing. Only `mode="round"` (the default) applies to a joint
        query — `floor`/`ceil`/`prev`/`next` have no well-defined meaning
        across several coupled dims at once. `tolerance` still applies, per
        queried coordinate name (a bare query is exact by default), checked
        against the chosen position's own value.

        On an **irregular** grid coordinate (an explicit multi-dim array
        with no regular spacing, e.g. an irregular satellite-swath
        `lat`/`lon`), a joint query works the same way — one value per
        coordinate name spanning the same `dims` — but resolves to the
        single **nearest** grid point across the queried coordinates' raw
        magnitudes. Mixing coordinates with very different units (degrees
        and metres, say) weighs the nearer one more heavily, same as any
        unnormalised distance always does. Only a single point is
        supported per call, not a vectorized query over many points at
        once. `tolerance`/`mode`/`method` behave the same as the joint case
        above (only the default "nearest" mode applies; a gap over
        `tolerance` raises).

        Pass `indexers` as an explicit mapping (`x.sel({"mode": "red"})`)
        instead of keyword arguments when a dim's name collides with one
        of `sel`'s own keyword parameters (`mode`, `tolerance`, `method`)
        — xarray's own escape hatch for exactly this, since a keyword
        argument matching one of those names is always bound to the
        parameter, never reaching the indexers. Passing both raises.
        """
        indexers = _either_dict_or_kwargs(
            indexers_positional, indexers_kwargs, "sel"
        )
        if mode is not None and method is not None:
            raise ValueError("sel: pass either 'mode' or 'method', not both")
        raw = mode if mode is not None else method
        sel_mode = _resolve_sel_mode(raw)
        if tolerance is None:
            # a bare sel is exact; asking for a mode implies an unbounded snap
            tolerance = 0 if raw is None else None
        elif isinstance(tolerance, float) and tolerance == float("inf"):
            tolerance = None  # explicit unbounded
        positional, consumed = self._affine_sel_groups(
            indexers, sel_mode, tolerance
        )
        curv_positional, curv_consumed = self._curvilinear_sel_groups(
            indexers, sel_mode, tolerance
        )
        for dim in curv_positional:
            if dim in positional:
                raise ValueError(
                    f"sel: dim {dim!r} is set by both a joint affine and a "
                    "joint curvilinear query in the same call -- pass one "
                    "or the other"
                )
        positional.update(curv_positional)
        consumed.update(curv_consumed)
        coords = self.coords
        for name, label in indexers.items():
            if name in consumed:
                continue
            if name in positional:
                # `name` is itself a dim already resolved by a joint affine
                # or curvilinear query over a *different* coordinate group
                # spanning it -- e.g. `x.sel(lat=.., lon=.., y=..)` where
                # `lat`/`lon` span `y` too. Silently letting this loop
                # overwrite that result would discard the joint solve
                # without any signal (#82 phase 1 review); pick one or the
                # other instead.
                raise ValueError(
                    f"sel: dim {name!r} is set both by a joint affine or "
                    "curvilinear query over its coordinate group and "
                    "directly in the same call -- pass one or the other"
                )
            if name not in coords:
                raise ValueError(f"sel: dim {name!r} has no coordinates")
            if name not in self.names:
                # a non-dimension coordinate (Proposal 0005) is not an index
                raise ValueError(
                    f"sel: {name!r} is not an index coordinate; "
                    "promote it with swap_dims first"
                )
            labels = coords[name]
            if isinstance(labels, Coordinate):
                if isinstance(label, slice):
                    positional[name] = _numeric_select_range(
                        labels, label, name
                    )
                    continue
                positional[name] = _numeric_select(
                    labels, label, sel_mode, tolerance, name
                )
                continue
            if isinstance(label, dict):
                positional[name] = _positions_to_index(
                    _match_positions(labels, label)
                )
                continue
            identities = [_label_name(one) for one in labels]
            is_many = isinstance(label, (list, tuple))
            wanted = list(label) if is_many else [label]
            positions = []
            for one in wanted:
                # the selector needs the same identity extraction the
                # stored labels already got, so `.sel(season=Season.WINTER)`
                # and `.sel(season="WINTER")` resolve identically (#107) --
                # falling back to the raw selector only when it doesn't
                # resolve to an identity of its own (e.g. it's already a
                # plain string, where extraction is a no-op).
                target = _label_name(one)
                if target is None:
                    target = one
                try:
                    positions.append(identities.index(target))
                except ValueError:
                    raise ValueError(
                        f"sel: no label {one!r} on dim {name!r}"
                    ) from None
            positional[name] = positions if is_many else positions[0]
        return self.isel(**positional)

    def _affine_sel_groups(
        self,
        indexers: tx.Mapping[str, tx.Any],
        sel_mode: str,
        tolerance: tx.Optional[float],
    ) -> tuple:
        """
        Resolve every **joint affine query** among `.sel`'s `indexers` --
        `{dim: integer position}` for each spanned dim, plus the set of
        indexer names consumed this way (issue #82 phase 1). A coordinate
        NAME present in `indexers` that spans several dims at once
        (Proposal 0005 step 3) is grouped with every other queried name
        sharing the exact same `dims`; each group must supply exactly
        `len(dims)` values (one per dim) to be square and invertible.
        """
        stored = self.__dict__.get("_coords") or {}
        groups: dict = {}
        for name in indexers:
            entry = stored.get(name)
            if entry is None:
                continue
            dims, coord = entry
            if (
                len(dims) > 1
                and isinstance(coord, Coordinate)
                and coord._compact()
            ):
                groups.setdefault(dims, []).append(name)
        positional: dict = {}
        consumed: set = set()
        for dims, names_in_group in groups.items():
            if len(names_in_group) != len(dims):
                raise ValueError(
                    f"sel: a joint affine query over {dims!r} needs "
                    f"exactly {len(dims)} coordinate value(s) (one per "
                    f"dim), got {len(names_in_group)} "
                    f"({sorted(names_in_group)!r}) -- square systems only "
                    "(#82 phase 1), no least-squares fallback"
                )
            positional.update(
                _affine_sel_indices(
                    self,
                    dims,
                    names_in_group,
                    indexers,
                    sel_mode,
                    tolerance,
                )
            )
            consumed.update(names_in_group)
        return positional, consumed

    def _curvilinear_sel_groups(
        self,
        indexers: tx.Mapping[str, tx.Any],
        sel_mode: str,
        tolerance: tx.Optional[float],
    ) -> tuple:
        """
        Resolve every **joint curvilinear query** among `.sel`'s `indexers` --
        nearest-neighbor lookup over a general (non-affine) multi-dim
        explicit coordinate (issue #82 phase 2, e.g. `x.sel(lat=52.1,
        lon=4.3)` for a 2-D curvilinear `lat`/`lon`). Grouped exactly like
        `_affine_sel_groups`: every coordinate NAME queried that spans the
        same `dims` is one group, and each group must supply exactly one
        value per spanned dim (a single point -- there is no vectorized
        "many points" form here, see `_curvilinear_sel_indices`).
        """
        stored = self.__dict__.get("_coords") or {}
        valid = self.coords
        groups: dict = {}
        for name in indexers:
            entry = stored.get(name)
            if entry is None:
                continue
            dims, coord = entry
            if (
                len(dims) > 1
                and isinstance(coord, Coordinate)
                and not coord._compact()
            ):
                if name not in valid:
                    # dropped (not resliced) by a previous op that changed
                    # one of its spanned dims' sizes -- leave it out of the
                    # group entirely so it falls through to the generic
                    # per-indexer loop below, which raises the usual "has
                    # no coordinates" error instead of a bare KeyError.
                    continue
                groups.setdefault(dims, []).append(name)
        positional: dict = {}
        consumed: set = set()
        for dims, names_in_group in groups.items():
            if len(names_in_group) != len(dims):
                raise ValueError(
                    f"sel: a joint curvilinear query over {dims!r} needs "
                    f"exactly {len(dims)} coordinate value(s) (one per "
                    f"dim), got {len(names_in_group)} "
                    f"({sorted(names_in_group)!r})"
                )
            positional.update(
                _curvilinear_sel_indices(
                    self,
                    dims,
                    names_in_group,
                    indexers,
                    sel_mode,
                    tolerance,
                )
            )
            consumed.update(names_in_group)
        return positional, consumed

    def interp(
        self,
        *indexers_positional: tx.Mapping,
        method: tx.Any = "linear",
        bound: tx.Any = None,
        extrapolate: tx.Any = None,
        name: tx.Optional[str] = None,
        **indexers_kwargs: tx.Any,
    ) -> tx.Self:
        """
        Interpolate onto new coordinate values along named dims.

        Where [`sel`][fiery.xtensor.XTensor.sel] *picks* existing positions,
        `interp` *computes* values at arbitrary positions of a **numeric**
        coordinate, the xarray way:

        ```python
        x.interp(t=2.5)                   # one point -> drops the axis
        x.interp(t=[0.0, 0.5, 1.0])       # several  -> keeps the axis
        x.interp(t="2.5s")                # unitful (backend converts)
        x.interp(t=q, method="cubic")     # a query tensor (grads flow)
        ```

        `method` is the interpolation order -- `"nearest"` (built in) or a
        higher order (`"linear"` *(default)*, `"quadratic"`, `"cubic"`, or an
        int), which needs the optional `fiery.interpol` backend
        (`pip install fiery-xtensor[interp]`). An out-of-range query follows
        `bound` (default: the `interp_bound` option -- `"replicate"` clamps
        to the edge) and `extrapolate` (default: the `interp_extrapolate`
        option); both can be set with
        [`set_options`][fiery.xtensor.set_options].

        A **scalar** query drops the axis (like `sel`); a **list**/tensor keeps
        it, its coordinate becoming the queried positions. A **regular**
        (evenly-spaced) coordinate supports every `method`; an **irregular**
        (explicit values) one only supports `"nearest"`/`"linear"`, both
        exact, since the map between value space and index space is locally
        linear between two bracketing ticks. A higher order needs a true
        non-uniform spline in *value* space, which isn't currently supported
        for an irregular coordinate.

        A **joint query over several dims that share one coordinate** (e.g. a
        `lat`/`lon` pair spanning several dims at once) resolves to a
        **fractional** position -- never rounded -- across all of them at
        once, then interpolates in that many dimensions together (falling
        back to a built-in nearest gather for `method="nearest"`, no extra
        backend needed). A query with every name given as a **scalar** is a
        single point: all the spanned dims drop, like the 1-D scalar case
        above. Any name given as a **list**/tensor makes it "many": every
        name's query broadcasts to a common length `N`, and the spanned dims
        collapse into **one new axis** of `N` sampled points -- not an
        outer-product grid, since the dims are coupled and you can't vary
        one queried name without moving through every spanned dim at once
        (mirroring xarray's own vectorized/pointwise-indexing convention for
        a value-based query on a multi-dim coordinate). The new axis is
        named `name` if given, else the shared name of any query that is
        itself a named 1-D `XTensor` -- `x.interp(lat=XTensor([...],
        names=("pts",)), lon=[...])` needs no `name=` at all, mirroring how
        xarray derives the result's new dimension from the *indexer*
        arrays' own shared dim name -- else unnamed (matching
        [`xstack`][fiery.xtensor.xstack]'s convention for a brand-new axis
        with nothing to infer from). When a name *is* resolved, the axis
        carries every queried name's own sampled values as a riding
        coordinate -- an unnamed axis can't be keyed, so it has none. Only
        **one** such joint group is supported per call; call `interp`
        again for a second group.

        A joint query over a **curvilinear** coordinate (a `lat(y, x)`-style
        array with no analytic formula, rather than a compact affine
        `spacing`/`origin` map) is also supported, for a 2-D spanned
        coordinate and `method="nearest"`/`"linear"` only:

        ```python
        grid.interp(lat=52.13, lon=4.28)   # nearest-neighbor seed + Newton
        ```

        There is no closed-form inverse for an arbitrary curvilinear map, so
        this seeds an initial guess from the nearest grid point (the same
        brute-force lookup `.sel` uses), then refines it with a few Newton
        iterations against a locally-estimated Jacobian, entirely in plain
        `torch` (no extra dependency). A query outside the grid's coordinate
        range raises rather than returning a silently wrong answer, as does
        one landing close enough to a fold (or other locally non-invertible
        patch) that the local Jacobian is itself singular there. That
        detection is necessarily local, though: a fold means at least two
        different positions share the same world coordinates, and *away*
        from the fold line itself (where the two branches are still locally
        distinct and individually invertible) the solve converges to
        *whichever* of the valid preimages its nearest-neighbor seed happens
        to land closest to, without warning that another equally-valid
        preimage exists. The Newton solve itself does not carry gradients
        back to the query point or the coordinate arrays -- only to the
        tensor's own **data** values, same as every other `interp` path.

        Pass `indexers` as an explicit mapping (`x.interp({"method":
        5.0})`) instead of keyword arguments when a dim's name collides
        with one of `interp`'s own keyword parameters (`method`, `bound`,
        `extrapolate`, `name`) -- xarray's own escape hatch for exactly
        this, since a keyword argument matching one of those names is
        always bound to the parameter, never reaching the indexers.
        Passing both raises.
        """
        indexers = _either_dict_or_kwargs(
            indexers_positional, indexers_kwargs, "interp"
        )
        if name is not None and not isinstance(name, str):
            # `name=` binds to this parameter before a same-named indexer
            # ever reaches `**indexers_kwargs` -- so `interp(name=3.0)` on a
            # dim literally called "name" would otherwise silently query
            # nothing at all (name=3.0 stored as an axis name never used,
            # since interp is numeric-only, no query needs a string here).
            # Catching the type mismatch turns that into a loud error;
            # reach the dim via the indexers dict instead.
            raise TypeError(
                f"interp: name= must be a str or None, got {name!r} -- "
                "pass interp({'name': ...}) to query a dim literally "
                "called 'name'"
            )
        _check_no_affine_curvilinear_dims_conflict(
            self.__dict__.get("_coords") or {}, indexers
        )
        out, consumed = self._affine_interp_group(
            indexers, method, bound, extrapolate, name
        )
        out, curv_consumed = out._curvilinear_interp_group(
            indexers, method, bound, extrapolate, name
        )
        # `consumed`/`curv_consumed` can never overlap: a name's stored
        # `Coordinate` is either `_compact()` (affine) or not (curvilinear),
        # never both, so the two groups can never both claim the same name.
        # The dims-level version of this conflict (same `dims` spanned by
        # one of each kind) is caught upfront by
        # `_check_no_affine_curvilinear_dims_conflict`, before either group
        # runs.
        consumed = consumed | curv_consumed
        for key, target in indexers.items():
            if key in consumed:
                continue
            out = out._interp_axis(key, target, method, bound, extrapolate)
        return out

    def _interp_axis(
        self,
        name: str,
        target: tx.Any,
        method: tx.Any,
        bound: tx.Any,
        extrapolate: tx.Any,
    ) -> tx.Self:
        """Interpolate a single named axis onto `target` (see `interp`)."""
        axis = _resolve_axis(self.names, name)
        coord = self.coords.get(name)
        if not isinstance(coord, Coordinate):
            raise ValueError(f"interp: dim {name!r} has no numeric coordinate")
        order = _interp_order(method)
        if coord._compact():
            spacing = dict.__getitem__(coord, "spacing")
            origin = dict.get(coord, "origin")
            unit = spacing["unit"]
            step = spacing["value"]
            base = origin["value"] if origin is not None else 0
            query, is_many = _query_values(target, unit)
            frac = (query - base) / step
        else:
            if order >= 2:
                raise NotImplementedError(
                    f"interp(method={method!r}) on the irregular coordinate "
                    f"{name!r}: only nearest/linear are supported on an "
                    "irregular coordinate (#73) -- a higher order would need "
                    "a true non-uniform spline in value space, which this "
                    "architecture cannot provide (not a missing feature, "
                    "see #81)"
                )
            stored_values = dict.__getitem__(coord, "value")
            unit = stored_values.units
            query, is_many = _query_values(target, unit)
            frac = _irregular_frac(
                stored_values.as_subclass(Tensor), query, name
            )
        if frac.numel() == 0:
            # an empty query -> an empty axis, the same way an empty
            # advanced index (`x[[]]`) already behaves, rather than the
            # backend's internal reshape choking on a zero-sized grid (#96).
            empty_index = torch.empty(0, dtype=torch.long, device=self.device)
            raw = self.as_subclass(Tensor).index_select(axis, empty_index)
        else:
            eff_bound = _get_option("interp_bound") if bound is None else bound
            eff_extrap = (
                _get_option("interp_extrapolate")
                if extrapolate is None
                else extrapolate
            )
            raw = _interp_pull(
                self.as_subclass(Tensor),
                axis,
                frac,
                order,
                eff_bound,
                eff_extrap,
            )
        out = _carry(self, raw)
        # the interpolated axis now sits at the queried positions: give it an
        # explicit coordinate (dropping whatever `name` held before -- labels
        # or numeric -- plus any non-dimension coordinate riding on it, since
        # neither corresponds to the new positions; Proposal 0005).
        new_coords = _coords_dropping(self, name)
        explicit = Coordinate(value=XTensor(query, units=unit))
        new_coords[name] = (name,), explicit
        out._coords = new_coords
        if not is_many:
            # a scalar query drops the axis (like integer indexing / sel)
            out = out.isel(**{name: 0})
        return out

    def _affine_interp_group(
        self,
        indexers: tx.Mapping[str, tx.Any],
        method: tx.Any,
        bound: tx.Any,
        extrapolate: tx.Any,
        name: tx.Optional[str],
    ) -> tuple:
        """
        Resolve a **joint affine interp** among `interp`'s indexers (issue
        #82 phase 2): a multi-dim compact affine coordinate queried jointly
        by every coordinate name spanning it, returning `(result,
        consumed_names)` -- `self` unchanged and an empty set when no such
        group is present, so every existing single-dim `interp` call is
        untouched. Mirrors `.sel`'s `_affine_sel_groups`, but computes a
        **fractional** index (never rounded) and performs a genuine N-D
        pull rather than picking one integer position.
        """
        stored = self.__dict__.get("_coords") or {}
        groups: dict = {}
        for nm in indexers:
            entry = stored.get(nm)
            if entry is None:
                continue
            dims, coord = entry
            if (
                len(dims) > 1
                and isinstance(coord, Coordinate)
                and coord._compact()
            ):
                groups.setdefault(dims, []).append(nm)
        if not groups:
            return self, set()
        if len(groups) > 1:
            raise NotImplementedError(
                "interp: a joint affine query over more than one "
                "coordinate group in the same call isn't supported yet "
                "(#82 phase 2) -- call interp() once per group"
            )
        (dims, names_in_group) = next(iter(groups.items()))
        if len(names_in_group) != len(dims):
            raise ValueError(
                f"interp: a joint affine query over {dims!r} needs "
                f"exactly {len(dims)} coordinate value(s) (one per dim), "
                f"got {len(names_in_group)} ({sorted(names_in_group)!r}) "
                "-- square systems only (#82 phase 2), no least-squares "
                "fallback"
            )
        out = _affine_interp_pull(
            self,
            dims,
            names_in_group,
            indexers,
            method,
            bound,
            extrapolate,
            name,
        )
        return out, set(names_in_group)

    def _curvilinear_interp_group(
        self,
        indexers: tx.Mapping[str, tx.Any],
        method: tx.Any,
        bound: tx.Any,
        extrapolate: tx.Any,
        name: tx.Optional[str],
    ) -> tuple:
        """
        Resolve a **joint curvilinear interp** among `interp`'s indexers
        (issue #82): a multi-dim *explicit* (non-affine) coordinate queried
        jointly by every coordinate name spanning it, returning `(result,
        consumed_names)` -- `self` unchanged and an empty set when no such
        group is present, so every existing affine/single-dim `interp` call
        is untouched. Mirrors `_affine_interp_group`, but the coordinate map
        has no closed-form inverse: `_curvilinear_interp_pull` seeds a
        fractional index from the existing brute-force `.sel` nearest
        lookup and refines it with Newton's method.
        """
        stored = self.__dict__.get("_coords") or {}
        valid = self.coords
        groups: dict = {}
        for nm in indexers:
            entry = stored.get(nm)
            if entry is None:
                continue
            dims, coord = entry
            if (
                len(dims) > 1
                and isinstance(coord, Coordinate)
                and not coord._compact()
            ):
                if nm not in valid:
                    # dropped (not resliced) by a previous op that changed
                    # one of its spanned dims' sizes -- fall through to the
                    # generic per-indexer loop, which raises the usual "has
                    # no coordinates" error instead of a bare KeyError.
                    continue
                groups.setdefault(dims, []).append(nm)
        if not groups:
            return self, set()
        if len(groups) > 1:
            raise NotImplementedError(
                "interp: a joint curvilinear query over more than one "
                "coordinate group in the same call isn't supported yet "
                "(#82) -- call interp() once per group"
            )
        (dims, names_in_group) = next(iter(groups.items()))
        if len(names_in_group) != len(dims):
            raise ValueError(
                f"interp: a joint curvilinear query over {dims!r} needs "
                f"exactly {len(dims)} coordinate value(s) (one per dim), "
                f"got {len(names_in_group)} ({sorted(names_in_group)!r}) "
                "-- square systems only (#82), no least-squares fallback"
            )
        out = _curvilinear_interp_pull(
            self,
            dims,
            names_in_group,
            indexers,
            method,
            bound,
            extrapolate,
            name,
        )
        return out, set(names_in_group)

    def _dims_with_label(self, label: str) -> list:
        """Named dims a label **identity** appears on (usually 0 or 1)."""
        return [
            dim
            for dim, labels in self.coords.items()
            if any(_label_name(one) == label for one in labels)
        ]

    def __getattr__(self, name: str) -> tx.Self:
        # Only consulted when normal attribute lookup fails, so real methods
        # and attributes always win. Private / dunder names are never labels.
        if name.startswith("_"):
            raise AttributeError(name)
        hits = self._dims_with_label(name)
        if len(hits) == 1:
            # positional (not **kwargs): `x.<label>` must still resolve a
            # dim literally named "mode"/"tolerance"/"method".
            return self.sel({hits[0]: name})
        if len(hits) > 1:
            raise AttributeError(
                f"label {name!r} is ambiguous across dims {hits}"
            )
        raise AttributeError(name)

    # -- transposes --------------------------------------------------------

    @property
    def T(self) -> tx.Self:
        dims = reversed(range(self.ndim))
        return self.permute(*dims)

    @property
    def mT(self) -> tx.Self:
        """Transpose of the last two dimensions (names included)."""
        return self.transpose(-2, -1)

    # -- builtin named-tensor API, re-implemented self-managed -------------

    def refine_names(self, *names: str | None) -> tx.Self:
        """
        Return a view with names assigned to (only) the unnamed axes.

        Naming an already-named axis to a *different* name is an error; a
        given `None` keeps the current name. A single `...` keeps the names
        of the axes it spans. Self-managed (not the builtin op).
        """
        current = list(self.names)
        # A single `...` keeps the names of the axes it spans (refine only
        # touches the *unnamed* axes; the spanned run rides through unchanged).
        names = _expand_name_ellipsis(names, self.ndim, tuple(current))
        if len(names) != self.ndim:
            raise ValueError(
                f"refine_names: expected {self.ndim} names, got {len(names)}"
            )
        refined = []
        for cur, given in zip(current, names):
            if given is None:
                refined.append(cur)
            elif cur is not None and cur != given:
                raise ValueError(
                    f"refine_names: cannot rename axis {cur!r} to {given!r}"
                )
            else:
                refined.append(given)
        return self.rename(*refined)

    def _align_order(self, names: tuple) -> list:
        """Resolve a name order (possibly with `...`) to a permutation."""
        current = list(self.names)
        if Ellipsis in names:
            explicit = [n for n in names if n is not Ellipsis]
            rest = [n for n in current if n not in explicit]
            i = names.index(Ellipsis)
            names = names[:i] + tuple(rest) + names[i + 1 :]
        if len(names) != self.ndim:
            raise ValueError(
                f"align: expected {self.ndim} names, got {len(names)}"
            )
        return [_resolve_axis(self.names, n) for n in names]

    def align_to(self, *names: str) -> tx.Self:
        """
        Return a view with the axes permuted into the given name order.

        A single `...` stands for all the other axes, in their current
        order (e.g. `x.align_to(..., "channel")`). Self-managed.
        """
        return self.permute(*self._align_order(names))

    def align_as(self, other: XTensor) -> tx.Self:
        """
        Return a view aligned to `other`'s named axes.

        This tensor's axes are permuted into `other`'s order, and a size-1
        axis is inserted for every name that only `other` has. Every axis of
        `self` must be named and present in `other`. Self-managed.
        """
        target = _names_of(other)
        mine = self.names
        for name in mine:
            if name is None or name not in target:
                raise ValueError(
                    f"align_as: axis {name!r} is not in the target names "
                    f"{tuple(target)}"
                )
        # permute self's axes into the order they appear in `other`
        order = [mine.index(name) for name in target if name in mine]
        out = self.permute(*order)
        # insert size-1 axes for the names only `other` has
        for pos, name in enumerate(target):
            if name not in mine:
                out = out.unsqueeze(pos)
        return out.rename(*target)


class Coordinate(_units.MagicDict):
    """
    A **numeric coordinate** -- a magic dict in one of two forms:

    - **compact / regular** -- `{spacing[, origin]}` (each a
      [`Unitful`][fiery.xtensor._units.Unitful]); `["value"]` is a **derived**
      key materialising `origin + i * spacing` **fresh each access** (no cache,
      so a learnable spacing never goes stale and gradients flow back);
    - **explicit / irregular** -- `{"value": <unitful 1-D tensor>}` (a bare
      tensor is equivalent sugar for the same thing); `["value"]` returns
      the stored array.

    The **position** unit (`["value"].units`) is distinct from the tensor's
    own data unit.
    """

    def _compact(self) -> bool:
        """Whether this is a compact (spacing/origin) coordinate."""
        return "spacing" in self or "origin" in self

    def _bound(self, size: int) -> "Coordinate":
        """A copy that knows its axis `size`, so `["value"]` materialises."""
        out = Coordinate(self)
        out._size = size
        return out

    def _bound_axes(self, axes: tuple) -> "Coordinate":
        """
        A copy bound to several axes -- `((spacing component, axis size),
        ...)`, **in the host tensor's axis order** -- so `["value"]`
        materialises an N-D **affine** grid laid out like the tensor
        (Proposal 0005 step 3: `spacing` is a vector with one component per
        spanned dim, but `dims` need not be in the tensor's own axis order).
        """
        out = Coordinate(self)
        out._axes = tuple(axes)
        return out

    def _bound_curvilinear(self, order: tuple) -> "Coordinate":
        """
        A copy bound to a permutation (one raw-tensor axis index per host
        axis, ascending) so `["value"]` returns an explicit **curvilinear**
        coordinate's stored array reordered to the host tensor's own axis
        order (issue #82) -- the same "laid out like the tensor, not like
        `dims`" rule `_bound_axes` follows for the compact affine form.
        """
        out = Coordinate(self)
        out._curv_order = tuple(order)
        return out

    def __getitem__(self, key: tx.Any) -> tx.Any:
        if key == "value" and self._compact():
            if "_axes" in self.__dict__:
                return self._materialise_axes()
            return self._materialise()
        if key == "value" and "_curv_order" in self.__dict__:
            return self._materialise_curvilinear()
        return dict.__getitem__(self, key)

    def _materialise(self) -> "XTensor":
        spacing = dict.__getitem__(self, "spacing")
        origin = dict.get(self, "origin")
        step = spacing["value"]
        start = origin["value"] if origin is not None else 0
        index = torch.arange(self._size)
        if isinstance(step, Tensor):
            index = index.to(step)
        values = start + index * step
        return XTensor(values, units=spacing["unit"])

    def _materialise_axes(self) -> "XTensor":
        """
        Materialise a compact **affine** coordinate (Proposal 0005 step 3)
        over its bound `_axes`: `origin + sum_d spacing[d] * index_d`, via a
        broadcast `arange` per spanned dim -- an N-D grid, still
        differentiable w.r.t. `spacing`/`origin` (no dense grid is stored,
        only assembled fresh on each access, exactly like the 1-D case). The
        axes come in the host tensor's order (see `_bound_axes`), each paired
        with the `spacing` component it draws on, so the grid lines up with
        the data whatever order `dims` is in.
        """
        spacing = dict.__getitem__(self, "spacing")
        origin = dict.get(self, "origin")
        components = spacing["value"]
        total = origin["value"] if origin is not None else 0
        ndim = len(self._axes)
        for axis, (component_index, size) in enumerate(self._axes):
            component = components[component_index]
            index = torch.arange(size)
            if isinstance(component, Tensor):
                index = index.to(component)
            shape = [1] * ndim
            shape[axis] = size
            total = total + index.view(shape) * component
        return XTensor(total, units=spacing["unit"])

    def _materialise_curvilinear(self) -> "XTensor":
        """
        Reorder an explicit **curvilinear** coordinate's stored array (issue
        #82) from its construction axis order to the host tensor's own axis
        order (see `_bound_curvilinear`) -- a data-preserving permutation, no
        interpolation or recomputation, since (unlike the affine form) there
        is no formula to re-derive from.
        """
        raw = dict.__getitem__(self, "value")
        order = self._curv_order
        if order == tuple(range(len(order))):
            return raw
        return raw.permute(*order)

    def to(self, unit: tx.Any) -> "Coordinate":
        """
        Convert the coordinate's **position** unit, rescaling
        `spacing`/`origin` (compact) or the stored `value` (explicit). Needs a
        backend. Carries over whatever axis binding this coordinate already
        had, so `coords[name].to(unit)["value"]` still materialises
        correctly for the tensor it came from.
        """
        if self._compact():
            out = Coordinate()
            out["spacing"] = dict.__getitem__(self, "spacing").to(unit)
            if "origin" in self:
                out["origin"] = dict.__getitem__(self, "origin").to(unit)
        else:
            out = Coordinate(
                value=dict.__getitem__(self, "value").to_units(unit)
            )
        if "_size" in self.__dict__:
            out._size = self._size
        if "_axes" in self.__dict__:
            out._axes = self._axes
        if "_curv_order" in self.__dict__:
            out._curv_order = self._curv_order
        return out


def as_xtensor(
    value: tx.Any,
    *,
    dtype: tx.Any = None,
    device: tx.Any = None,
    units: tx.Any = arrayutils._UNSET,
    names: tx.Any = arrayutils._UNSET,
    coords: tx.Any = arrayutils._UNSET,
) -> XTensor:
    """
    Coerce `value` (a bare Python number, a plain `Tensor`, or an `XTensor`)
    into an `XTensor` -- the `XTensor` analogue of `torch.as_tensor`:
    **graph-safe** (`torch.as_tensor(value)` with no `dtype=`/`device=` is a
    strict identity passthrough for an already-a-tensor `value` -- the *same
    object*, never a detaching copy, unlike `torch.tensor(existing_tensor)`'s
    well-known footgun of silently returning a fresh, non-differentiable
    leaf), and metadata-preserving: `units`/`names`/`coords` ride through
    untouched unless a keyword **explicitly** overrides them -- mirroring how
    `torch.as_tensor(t, dtype=..., device=...)` only converts what you pass.
    A given override **replaces wholesale**, never merges (`coords={...}`
    discards whatever coordinates `value` already had, rather than combining
    the two).

    `dtype=`/`device=` extend `torch.as_tensor`'s own conversion, applied
    *before* the metadata is settled (so e.g. an axis-typed vs. numeric
    dtype affects nothing about the labels themselves). `None` (the default
    for both) means "leave as is" -- the same convention `torch.as_tensor`
    and `.to()` use.

    A genuine dtype/device conversion always keeps the result's metadata:
    plain `torch.as_tensor(an_xtensor, dtype=...)` silently degrades to a
    **plain `Tensor`** whenever it actually has to convert something,
    stripping every bit of metadata in the process -- `as_xtensor` avoids
    that pitfall.

    `value`'s own tensor is never mutated: when nothing is overridden and
    `value` is already an `XTensor`, it is returned as-is (the same object,
    metadata included); otherwise the result is always a **fresh** view (no
    data copy) before any override is applied, so overriding e.g. `units=`
    never reaches back and changes `value`'s own unit as a side effect.
    """
    base = torch.as_tensor(value)
    # Skip `.to()` entirely when neither actually changes anything, rather
    # than trusting its own no-op-returns-self behaviour: passing `device=`
    # explicitly (even as `None`) alongside `dtype=` defeats that fast path
    # on old torch (verified on 1.7/1.8 CI) even when both already match --
    # this way, identity is guaranteed by construction, not by a version-
    # dependent internal optimisation.
    if (dtype is not None and dtype != base.dtype) or (
        device is not None and torch.device(device) != base.device
    ):
        base = base.to(dtype=dtype, device=device)
    if isinstance(base, XTensor) and (
        units is arrayutils._UNSET
        and names is arrayutils._UNSET
        and coords is arrayutils._UNSET
    ):
        return base
    out_cls = type(base) if isinstance(base, XTensor) else XTensor
    out = base.as_subclass(out_cls)
    if isinstance(base, XTensor):
        # copy the *raw* stored metadata directly, not through the
        # `names`/`coords` property setters -- those validate against
        # `out`'s already-current names/shape, which is premature when
        # `names` is being overridden in the same call: a coordinate keyed
        # by the *old* name would fail that validation outright instead of
        # simply going stale (`.coords`'s own getter already treats a
        # coordinate whose dim isn't a current name as invalid and silently
        # drops it -- direct assignment to `.names` on an existing `XTensor`
        # has exactly this same behaviour today, so this matches it rather
        # than introducing a new failure mode).
        out.__dict__.update(base.__dict__)
    if names is not arrayutils._UNSET:
        out.names = names
    if units is not arrayutils._UNSET:
        out.units = units
    if coords is not arrayutils._UNSET:
        out.coords = coords
    return out


def is_xtensor(obj: tx.Any) -> bool:
    """Whether `obj` is an `XTensor` (the `XTensor` analogue of
    `torch.is_tensor`)."""
    return isinstance(obj, XTensor)


def _as_unitful(obj: tx.Any) -> tx.Any:
    """Coerce a spacing/origin input to a `Unitful`, preserving a tensor."""
    if isinstance(obj, XTensor):
        unit = obj.units
        if unit is None:
            return _units.Unitful(value=obj, unit=_units.normalise(""))
        return _units.Unitful(value=obj.magnitude, unit=unit)
    return _units.as_unitful(obj)


def _as_unitful_vector(obj: tx.Any, ndims: int) -> tx.Any:
    """
    Coerce an affine coordinate's `spacing` input (Proposal 0005 step 3) to a
    `Unitful` wrapping a 1-D tensor of `ndims` components -- one per spanned
    dim. Any component given as a tensor (e.g. a learnable 0-rank tensor) is
    preserved via `torch.stack` rather than `torch.as_tensor`, so it keeps its
    autograd graph; a component given as a bare number is not learnable.
    """
    unitful = _as_unitful(obj)
    value = unitful["value"]
    if isinstance(value, Tensor):
        vec = value
    elif isinstance(value, (list, tuple)) and any(
        isinstance(v, Tensor) for v in value
    ):
        vec = torch.stack(
            [
                v
                if isinstance(v, Tensor)
                else torch.as_tensor(v, dtype=torch.get_default_dtype())
                for v in value
            ]
        )
    else:
        vec = torch.as_tensor(value, dtype=torch.get_default_dtype())
    if vec.ndim != 1 or vec.shape[0] != ndims:
        raise ValueError(
            "coords: an affine coordinate's spacing must have one component "
            f"per dim ({ndims} here), got shape {tuple(vec.shape)}"
        )
    return _units.Unitful(value=vec, unit=unitful["unit"])


def _as_unitful_origin(obj: tx.Any) -> tx.Any:
    """
    Coerce a coordinate's `origin` input to a `Unitful`, requiring it be a
    **scalar** -- unlike `spacing`, `origin` is always a single value shared
    across every spanned dim, even for a multi-dim affine coordinate (step
    3). Catches a non-scalar origin here with a clear message instead of
    deferring to an opaque broadcast-shape error at materialisation time.
    """
    unitful = _as_unitful(obj)
    value = unitful["value"]
    if isinstance(value, Tensor):
        shape = tuple(value.shape)
    elif isinstance(value, (list, tuple)):
        shape = (len(value),)
    else:
        shape = ()
    if shape:
        raise ValueError(
            f"coords: a coordinate's origin must be a scalar, got shape "
            f"{shape}"
        )
    return unitful


def _promote_numeric_labels(key: str, labels: tuple) -> tx.Any:
    """
    Auto-promote an all-numeric label sequence to a numeric `Coordinate`
    (issue #107) -- shared by dimension and non-dimension coordinate
    parsing, so a bare tuple/list of plain numbers is a numeric coordinate
    everywhere it can appear, not just on a dim's own index. Raise if
    numbers are mixed with categorical values (`bool`/`Enum`/`str`/dict/
    `None`/`...` are always categorical, never a position) -- otherwise
    return `labels` unchanged, for the caller's own label handling
    (Ellipsis-unroll, length check, ...).
    """
    if labels and all(_is_pure_number(label) for label in labels):
        try:
            values = torch.as_tensor(labels)
        except RuntimeError as exc:
            raise ValueError(
                f"coords: {key!r} -- {labels!r} isn't representable as a "
                f"numeric coordinate: {exc}"
            ) from None
        return _make_coordinate(values)
    if any(_is_pure_number(label) for label in labels):
        raise ValueError(
            f"coords: {key!r} mixes numeric values with categorical ones "
            f"in {labels!r} -- bool/Enum/str/dict/None/Ellipsis are "
            "always categorical (never a position); give a compact "
            "{'spacing': ...}/explicit tensor for a numeric coordinate, "
            "or make every value a label"
        )
    return labels


def _make_coordinate(spec: tx.Any) -> Coordinate:
    """Build a `Coordinate` from a compact spec, an explicit tensor, or a
    `{"value": ...}` mapping wrapping one."""
    _check_unambiguous_coord_spec(spec)
    if _is_explicit_coord(spec):
        if isinstance(spec, tx.Mapping):
            spec = spec["value"]
        if isinstance(spec, XTensor) and spec.units is not None:
            values = as_xtensor(spec)  # preserve its own unit, graph-safe
        else:
            # force dimensionless -- via the override kwarg, not a post-hoc
            # mutation, so a unit-less `spec` is never changed in place.
            # (Benign behaviour change vs. the old `XTensor(spec, unit=...)`
            # call this replaced: if `spec` is itself an XTensor with
            # `unit=None`, its own `names`/`coords` now ride along instead of
            # being silently dropped -- verified inert for every existing
            # caller, since a bare Tensor/number spec has none to preserve.)
            values = as_xtensor(spec, units=_units.normalise(""))
        if values.ndim != 1:
            raise ValueError(
                "coords: a numeric coordinate must be 1-D, got shape "
                f"{tuple(values.shape)}"
            )
        return Coordinate(value=values)
    coord = Coordinate()
    if "origin" in spec:
        coord["origin"] = _as_unitful_origin(spec["origin"])
    if "spacing" in spec:
        coord["spacing"] = _as_unitful(spec["spacing"])
    else:
        # symmetric to an omitted `origin` defaulting to 0 in `spacing`'s
        # unit (see `_materialise`): an omitted `spacing` defaults to 1 in
        # `origin`'s unit -- `_is_compact_coord` guarantees at least one of
        # the two is present, so this only runs with `origin` given.
        origin_unit = coord["origin"]["unit"] if "origin" in coord else ""
        coord["spacing"] = _units.Unitful(value=1, unit=origin_unit)
    _reconcile_origin_unit(coord)
    return coord


def _make_affine_coordinate(spec: tx.Mapping, ndims: int) -> Coordinate:
    """
    Build a compact **affine** `Coordinate` spanning `ndims` dims (Proposal
    0005 step 3) -- a generalisation of the 1-D compact form (0001) where
    `spacing` is a **vector**, one component per dim, and `origin` stays a
    single scalar shared across them: `value[i_0,...] = origin +
    sum_d spacing[d] * i_d`.
    """
    _check_unambiguous_coord_spec(spec)
    if "spacing" not in spec:
        raise ValueError(
            "coords: an affine (multi-dim) coordinate requires 'spacing'"
        )
    coord = Coordinate()
    coord["spacing"] = _as_unitful_vector(spec["spacing"], ndims)
    if "origin" in spec:
        coord["origin"] = _as_unitful_origin(spec["origin"])
    _reconcile_origin_unit(coord)
    return coord


def _make_curvilinear_coordinate(
    key: str, spec: tx.Any, dims: tuple
) -> Coordinate:
    """
    Build an explicit **curvilinear** `Coordinate` spanning `dims` (issue
    #82) -- the general counterpart of `_make_affine_coordinate` for a
    multi-dim non-dimension coordinate with no analytic form: an N-D tensor
    of values, one axis per dim in `dims` order, e.g. `lat(y, x)` on an
    irregular grid. Unlike the affine form there is no formula to fold a
    slice/index through, so (per `_parse_nondim_coord`'s docstring) this
    coordinate simply drops once a spanned dim's size no longer matches.
    """
    _check_unambiguous_coord_spec(spec)
    if not _is_explicit_coord(spec):
        raise ValueError(
            f"coords: {key!r} over several dims {dims!r} must be given as "
            "either a compact {'spacing': ...} affine map or an explicit "
            "tensor of values (or a {'value': ...} mapping wrapping one), "
            f"got {spec!r}"
        )
    if isinstance(spec, tx.Mapping):
        spec = spec["value"]
    if isinstance(spec, XTensor) and spec.units is not None:
        values = as_xtensor(spec)
    else:
        values = as_xtensor(spec, units=_units.normalise(""))
    if values.ndim != len(dims):
        raise ValueError(
            f"coords: {key!r} spans {len(dims)} dims {dims!r}, so its "
            f"values must be {len(dims)}-D (one axis per dim), got shape "
            f"{tuple(values.shape)}"
        )
    return Coordinate(value=values)


def _affine_sel_indices(
    tensor: "XTensor",
    dims: tuple,
    names_in_group: list,
    indexers: tx.Mapping[str, tx.Any],
    sel_mode: str,
    tolerance: tx.Optional[float],
) -> dict:
    """
    Solve the closed-form affine inverse for one joint `.sel` query (issue
    #82 phase 1): given a target world value for each of `len(dims)`
    coordinate names spanning the same `dims`, `index = A^-1 (world -
    origin)`, then snap to the nearest integer position along each dim --
    never materialising the affine grid (`spacing`/`origin` alone are
    enough, mirroring the 1-D compact `.sel` fast path, #110).

    `A`'s rows are each queried coordinate's `spacing` vector (already
    ordered along `dims`, Proposal 0005 step 3); `names_in_group`'s order
    only has to line up between `A`'s rows and the right-hand side, not
    match any particular canonical order.
    """
    if sel_mode != "round":
        raise NotImplementedError(
            f"sel: mode={sel_mode!r} isn't supported for a joint affine "
            "query over several coupled dims (#82 phase 1) -- only the "
            "default 'round' is; floor/ceil/prev/next don't have a "
            "well-defined meaning jointly across several dims"
        )
    stored = tensor.__dict__.get("_coords") or {}
    per_name = []
    for name in names_in_group:
        _, coord = stored[name]
        spacing = dict.__getitem__(coord, "spacing")
        origin = dict.get(coord, "origin")
        vec = spacing["value"]
        # solved in float64 regardless of the spacing's own/default dtype --
        # matching `_numeric_select_compact`'s closed-form convention -- so a
        # genuinely float64-precision-dependent spacing (or an int one) isn't
        # silently downcast to float32 and solved wrong (review finding #3).
        if isinstance(vec, Tensor):
            vec = vec.to(torch.float64)
        else:
            vec = torch.as_tensor(vec, dtype=torch.float64)
        base = float(origin["value"]) if origin is not None else 0.0
        target = _selector_value(indexers[name], spacing["unit"])
        per_name.append((name, vec, base, target, spacing["unit"]))
    matrix = torch.stack([vec for _, vec, _, _, _ in per_name])
    # built on `matrix`'s own device (not implicitly CPU): a spacing tensor
    # that lives off-CPU must not force a cross-device op here (review
    # finding #2).
    vector = torch.tensor(
        [target - base for _, _, base, target, _ in per_name],
        dtype=matrix.dtype,
        device=matrix.device,
    )
    try:
        index = torch.inverse(matrix) @ vector
    except RuntimeError as exc:
        # narrowed to the actual singular-matrix message, so an unrelated
        # failure (e.g. a device mismatch) isn't misattributed as "not
        # invertible" (review finding #2).
        if "singular" not in str(exc).lower():
            raise
        raise ValueError(
            f"sel: the affine map over {dims!r} ({sorted(names_in_group)!r}) "
            f"isn't invertible: {exc}"
        ) from None
    rounded = index.round().long().tolist()
    result = {}
    for dim, position in zip(dims, rounded):
        size = tensor.shape[_resolve_axis(tensor.names, dim)]
        if not 0 <= position < size:
            raise ValueError(
                f"sel: the joint affine query resolves dim {dim!r} to "
                f"index {position}, out of range for size {size}"
            )
        result[dim] = position
    # `tolerance` was silently ignored for a joint query (review finding #4)
    # -- re-evaluate the forward map at the rounded index for each queried
    # coordinate NAME (not dim: the gap is meaningful per coordinate, since
    # several can share the same dims) and enforce it the same way the 1-D
    # path does, so a bare `.sel(lat=.., lon=..)` stays exact by default too.
    # Built once on `matrix`'s own device/dtype (not implicitly CPU) -- a
    # per-name `torch.tensor(rounded, ...)` with no `device=` would silently
    # reintroduce the exact cross-device bug fix #2 (above) already closed.
    rounded_t = torch.tensor(rounded, dtype=matrix.dtype, device=matrix.device)
    for name, vec, base, target, unit in per_name:
        predicted = base + float(vec @ rounded_t)
        gap = abs(predicted - target)
        tol = None if tolerance is None else _selector_value(tolerance, unit)
        _check_sel_tolerance(gap, tol, target, sel_mode, indexers[name], name)
    return result


#: Distance computation's working dtype for `_curvilinear_sel_indices`
#: (issue #82 phase 2) -- always float64 regardless of the grid's own dtype,
#: matching `_affine_sel_indices`'s "solved in float64" convention. This
#: also sidesteps a real footgun a plain squared-distance sum would have in
#: float32 for realistic coordinate magnitudes (lat ~52 deg, UTM northings
#: ~6e6 m): squaring and summing large values first loses the precision a
#: small grid spacing needs (independent review finding -- the same
#: cancellation `torch.cdist`'s default `use_mm_for_euclid_dist` compute
#: mode has above 25 points, avoided here by not using `cdist` at all).
_CURVILINEAR_SEL_DTYPE = torch.float64

#: Distance-computation size guard for `_curvilinear_sel_indices` (issue
#: #82 phase 2) -- brute force (no tree index), so a query against a grid
#: this large is rejected rather than left to run for a long time or
#: exhaust memory. Accounts for the `(n, k)` stacked point cloud plus the
#: `(n,)` distance vector, both in `_CURVILINEAR_SEL_DTYPE` -- the actual
#: peak allocation for a single query point -- so this is far above what
#: any realistic single-point lookup needs; it exists as a backstop against
#: a pathologically large grid, not a routine limit.
_CURVILINEAR_SEL_MAX_BYTES = 2 * 1024**3


def _curvilinear_sel_indices(
    tensor: "XTensor",
    dims: tuple,
    names_in_group: list,
    indexers: tx.Mapping[str, tx.Any],
    sel_mode: str,
    tolerance: tx.Optional[float],
) -> dict:
    """
    Exact nearest-neighbor lookup for one joint `.sel` query over a general
    **curvilinear** coordinate (issue #82 phase 2): stack the queried
    coordinates' values into one `(*grid_shape, k)` point cloud and find the
    grid point closest to the target by direct squared distance -- brute
    force (`O(grid size)` memory/time), since an arbitrary curvilinear grid
    has no closed-form inverse the way the affine case does. Deliberately
    torch-native (no scipy/sklearn KD-tree dependency, so it stays
    GPU-capable) and deliberately single-point only: a query vectorized over
    many points at once (the shape a bulk regrid needs) would multiply the
    point cloud by the query count, which stops being brute-forceable well
    before a tree index would even notice -- see `vs-xarray.md`.
    """
    if sel_mode != "round":
        raise NotImplementedError(
            f"sel: mode={sel_mode!r} isn't supported for a joint "
            "curvilinear query (#82 phase 2) -- only nearest-neighbor "
            "(the default, or method='nearest') is"
        )
    for name in names_in_group:
        target = indexers[name]
        if not isinstance(target, (int, float)) and not (
            isinstance(target, Tensor) and target.ndim == 0
        ):
            raise TypeError(
                f"sel: a joint curvilinear query over {dims!r} (#82 phase "
                f"2) only supports a single point -- {name}={target!r} "
                "must be a bare number (or 0-D tensor), not a list/slice/"
                "multi-element tensor; there is no vectorized multi-point "
                "form yet, see vs-xarray.md"
            )
    # `coords_bound[name]["value"]` materialises in **ascending host axis
    # order** among `dims` (see `_bound_curvilinear`), which need not be
    # `dims`'s own given order -- `sorted_dims` matches that same order, so
    # unraveling the flat argmin index below lines up with the grid's actual
    # axes instead of silently transposing two same-size spanned dims.
    axes = [_resolve_axis(tensor.names, dim) for dim in dims]
    sorted_dims = tuple(
        dims[i] for i in sorted(range(len(dims)), key=axes.__getitem__)
    )
    coords_bound = tensor.coords
    grids = []
    units = []
    grid_shape = None
    for name in names_in_group:
        grid = coords_bound[name]["value"]
        if grid_shape is None:
            grid_shape = tuple(grid.shape)
        grids.append(grid.as_subclass(Tensor))
        units.append(grid.units)
    n = 1
    for size in grid_shape:
        n *= size
    if n == 0:
        raise ValueError(
            f"sel: a joint curvilinear query over {dims!r} has an empty "
            "grid (a spanned dim has size 0) -- there is no nearest point"
        )
    k = len(names_in_group)
    itemsize = torch.tensor([], dtype=_CURVILINEAR_SEL_DTYPE).element_size()
    if n * itemsize * (k + 1) > _CURVILINEAR_SEL_MAX_BYTES:
        raise ValueError(
            f"sel: a joint curvilinear query over {dims!r} needs a "
            f"brute-force nearest-neighbor search over {n} grid points "
            f"(~{n * itemsize * (k + 1) / 1e9:.2f} GB) -- too large; "
            "see vs-xarray.md for the general-curvilinear scaling limit"
        )
    points = torch.stack(
        [g.reshape(-1).to(_CURVILINEAR_SEL_DTYPE) for g in grids], dim=-1
    )
    targets = torch.tensor(
        [
            _selector_value(indexers[name], unit)
            for name, unit in zip(names_in_group, units)
        ],
        dtype=points.dtype,
        device=points.device,
    )
    # squared distance, not `torch.cdist`: `cdist`'s default compute mode
    # switches to the `||a||^2 + ||b||^2 - 2 a.b` identity above 25 points,
    # which catastrophically cancels in float32 for realistic coordinate
    # magnitudes (independent review finding) -- a direct sum has no such
    # cliff, and NaN grid points (masked/fill swath cells) are pushed to
    # +inf here rather than winning the argmin by NaN-propagation.
    sq_dist = (points - targets).pow(2).sum(-1)
    sq_dist = torch.where(
        torch.isnan(sq_dist), torch.full_like(sq_dist, float("inf")), sq_dist
    )
    flat_index = int(sq_dist.argmin())
    unraveled = []
    remaining = flat_index
    for size in reversed(grid_shape):
        unraveled.append(remaining % size)
        remaining //= size
    unraveled.reverse()
    result = dict(zip(sorted_dims, unraveled))
    for i, (name, unit) in enumerate(zip(names_in_group, units)):
        predicted = float(points[flat_index, i])
        target = _selector_value(indexers[name], unit)
        gap = abs(predicted - target)
        tol = None if tolerance is None else _selector_value(tolerance, unit)
        if not math.isfinite(gap):
            gap = float("inf")  # a NaN grid point never satisfies a bound
        _check_sel_tolerance(gap, tol, target, sel_mode, indexers[name], name)
    return result


#: Newton solve controls for `_curvilinear_interp_pull` (issue #82, general
#: curvilinear `.interp`) -- fixed, small budgets rather than adaptive
#: control: the forward map sampled here is piecewise-**bilinear** in the
#: fractional index (exact within one grid cell), and the nearest-neighbor
#: seed already starts within one cell of the true answer for a reasonably
#: sampled grid, so a handful of iterations either converges or signals a
#: genuinely bad query (out of range, or a non-invertible/folded patch).
_CURVILINEAR_INTERP_MAX_ITER = 50
#: Convergence check: `residual.abs() < atol + rtol * |target|`, mirroring
#: `torch.allclose`'s own atol/rtol combination (review of #166, finding
#: #3). A bare absolute tolerance is only reachable up to coordinate
#: magnitudes of a few million in float64 (spacing is `~2.2e-16 * M`), so
#: UTM/ECEF-scale (metres, 1e6-1e7) or millimetre-precision coordinates
#: would spuriously "fail to converge" well before they actually ran out of
#: representable precision. `_CURVILINEAR_INTERP_ATOL` alone still governs
#: small-magnitude coordinates (its previous role as the sole tolerance).
_CURVILINEAR_INTERP_ATOL = 1e-9
_CURVILINEAR_INTERP_RTOL = 1e-10
_CURVILINEAR_INTERP_FD_STEP = 1e-4
#: Singularity check: compare `|det(J)|` against `rtol * ||J[:, 0]|| *
#: ||J[:, 1]||` rather than a bare constant (review of #166, finding #2).
#: `det(J)` carries units of `world0 * world1 / index^2`, so a fixed
#: absolute threshold is only sensible for O(1)-magnitude coordinates: the
#: same geometry in fine-resolution degrees (`|det J| ~ 1e-9`) or in
#: large-magnitude metres (`|det J|` inflated by `~1e6`) would push a fixed
#: threshold to false-positive or false-negative respectively. Normalizing
#: by the column norms turns the test into `|sin(theta)| < rtol`, where
#: `theta` is the angle between the Jacobian's two column vectors -- a
#: dimensionless, scale-free measure of how close the local map is to
#: folding (theta -> 0), which is what "isn't locally invertible" actually
#: means. `<=` (not `<`) so a Jacobian with an exactly-zero column (e.g. a
#: degenerate size-1 axis, where both norms and the determinant are all
#: exactly 0) is still caught rather than passing a `0 < 0` test.
_CURVILINEAR_INTERP_SINGULAR_RTOL = 1e-6
#: How far (in index units) a solved position may sit outside `[0, size -
#: 1]` before it counts as out-of-bounds rather than an edge cell -- half a
#: grid cell either side, mirroring `bound="replicate"`'s own edge handling.
_CURVILINEAR_INTERP_OOB_MARGIN = 0.5


def _bilinear_sample_2d(
    grid: Tensor, ij: Tensor, *, clamp_query: bool = True
) -> Tensor:
    """
    Bilinearly sample a 2-D `grid` (shape `(h, w)`) at fractional row/column
    positions `ij` (`(..., 2)`). The forward-map evaluator the curvilinear
    Newton solve (`_curvilinear_interp_pull`) repeatedly calls --
    deliberately independent of the optional `fiery.interpol` backend (pure
    `torch`, no extra dependency), since the inversion needs to evaluate it
    many times per query point and a size-1 axis is a real, if degenerate,
    case (no interpolation possible along it, so that axis's contribution
    is a flat zero).

    `clamp_query=True` (the default) clamps `ij` to the valid `[0, size -
    1]` range first (replicate at the border) -- the right behavior for
    evaluating the map itself, including at/near the boundary, so a
    genuinely out-of-range query still degrades to a flattened derivative
    there and gets caught (see `_curvilinear_newton_indices`).

    `clamp_query=False` skips that clamp: `i`/`j` (and therefore `di`/`dj`)
    are allowed to go negative or past `size - 1`, which -- since `i0`/`j0`
    (the gather indices) are still clamped to a valid cell -- linearly
    *extrapolates* along that edge cell's own slope instead of flattening.
    This is what `_curvilinear_newton_indices` uses for its finite-
    difference Jacobian: a central difference straddling index 0 or `size -
    1` must see the true local derivative, not one artificially halved by
    one of its two sample points collapsing back onto the clamped boundary
    (review of #166, finding #1) -- the nearest-neighbor Newton seed is
    *always* an exact integer index by construction, so boundary rows/
    columns (a third or more of any grid's nodes) are hit on essentially
    every query, not as some rare edge case.
    """
    h, w = grid.shape[-2], grid.shape[-1]
    i = ij[..., 0]
    j = ij[..., 1]
    if clamp_query:
        i = i.clamp(0, h - 1)
        j = j.clamp(0, w - 1)
    i0 = i.floor().long().clamp(0, max(h - 2, 0))
    j0 = j.floor().long().clamp(0, max(w - 2, 0))
    i1 = (i0 + 1).clamp(max=h - 1)
    j1 = (j0 + 1).clamp(max=w - 1)
    di = i - i0.to(i.dtype)
    dj = j - j0.to(j.dtype)
    if h == 1:
        di = torch.zeros_like(di)
    if w == 1:
        dj = torch.zeros_like(dj)
    g00 = grid[i0, j0]
    g01 = grid[i0, j1]
    g10 = grid[i1, j0]
    g11 = grid[i1, j1]
    top = g00 + (g01 - g00) * dj
    bot = g10 + (g11 - g10) * dj
    return top + (bot - top) * di


def _curvilinear_out_of_bounds(
    index: Tensor,
    sizes: tuple,
    dims: tuple,
    names_in_group: list,
    active: tx.Optional[Tensor] = None,
) -> None:
    """
    Raise a clear `ValueError` if any row of `index` (`(n, k)`) sits outside
    `[0, size - 1]` (per column) by more than
    `_CURVILINEAR_INTERP_OOB_MARGIN` -- shared between the mid-solve check
    (an out-of-range query degrades to a singular clamped-boundary Jacobian,
    see `_curvilinear_newton_indices`) and the final post-convergence check.

    `active` (`(n,)` bool, default all-`True`) restricts which rows are
    checked, while keeping the reported row number relative to the full
    batch -- the mid-solve call passes the not-yet-converged mask, so an
    already-converged point sitting near the margin (nothing wrong with it)
    can never be the one this raises about (review of #166, finding #4).
    """
    if active is None:
        active = torch.ones(
            index.shape[0], dtype=torch.bool, device=index.device
        )
    for col in range(index.shape[1]):
        size = sizes[col]
        lo = -_CURVILINEAR_INTERP_OOB_MARGIN
        hi = size - 1 + _CURVILINEAR_INTERP_OOB_MARGIN
        # non-strict: `index` is clamped into `[lo, hi]` after every Newton
        # step, so a point that actually diverged past the margin settles
        # exactly *at* the boundary, not beyond it -- `<=`/`>=` still catches
        # that (an interior solution landing exactly on the boundary by
        # genuine coincidence is vanishingly unlikely).
        out_of_bounds = active & (
            (index[:, col] <= lo) | (index[:, col] >= hi)
        )
        if bool(out_of_bounds.any()):
            bad = int(out_of_bounds.nonzero()[0])
            raise ValueError(
                f"interp: a joint curvilinear query over {dims!r} "
                f"({sorted(names_in_group)!r}) resolves query point {bad} "
                f"to index {index[bad].tolist()}, out of range for size "
                f"{size} along dim {dims[col]!r} -- the target is outside "
                "the grid's coordinate range"
            )


def _curvilinear_newton_indices(
    grids: list,
    targets: Tensor,
    init_index: Tensor,
    sizes: tuple,
    dims: tuple,
    names_in_group: list,
) -> Tensor:
    """
    Invert a 2-D curvilinear coordinate map by Newton's method (issue #82):
    given `targets` (`(n, 2)`, one row per query point, columns matching
    `names_in_group`) and `init_index` (`(n, 2)`, the nearest-neighbor
    seed), refine towards the fractional index where the bilinearly
    sampled `grids` equal `targets`, via a local Jacobian estimated by
    central finite differences (`_CURVILINEAR_INTERP_FD_STEP`) and an
    explicit 2x2 solve (this path is scoped to exactly 2 spanned dims, see
    `_curvilinear_interp_pull`). Detached throughout -- an iterative,
    data-dependent root find has no well-defined gradient to carry back to
    `targets`/`grids`; only the tensor's own data values stay differentiable
    once this fractional index is handed to the actual pull.

    Raises `ValueError` (naming the offending query point) when the local
    Jacobian is singular (the map isn't locally invertible there -- e.g. a
    fold or a degenerate/size-1 axis), when the solve doesn't converge
    within the iteration budget, or when the final position lands outside
    the grid's index range by more than half a cell -- rather than
    returning an extrapolated or otherwise unreliable position.

    A point that has already converged is masked out of every later
    iteration's singularity test (and its Newton update) -- otherwise its
    outcome would depend on whichever *other*, still-iterating point
    happens to be batched alongside it: a point that converges immediately
    (e.g. it seeds exactly onto a fold's apex, where the true Jacobian is
    itself singular *at the solution*) would still have that singular
    Jacobian re-tested -- and raise -- on every subsequent iteration driven
    by a slower batch-mate, even though queried alone it succeeds
    (review of #166, finding #4).
    """
    h_step = _CURVILINEAR_INTERP_FD_STEP
    index = init_index.clone()
    n, k = index.shape
    lo = torch.tensor(
        [-_CURVILINEAR_INTERP_OOB_MARGIN] * k,
        dtype=index.dtype,
        device=index.device,
    )
    hi = torch.tensor(
        [size - 1 + _CURVILINEAR_INTERP_OOB_MARGIN for size in sizes],
        dtype=index.dtype,
        device=index.device,
    )
    converged = torch.zeros(n, dtype=torch.bool, device=index.device)
    for _iteration in range(_CURVILINEAR_INTERP_MAX_ITER):
        world = torch.stack(
            [_bilinear_sample_2d(g, index) for g in grids], dim=-1
        )  # (n, k)
        residual = targets - world
        tol = (
            _CURVILINEAR_INTERP_ATOL + _CURVILINEAR_INTERP_RTOL * targets.abs()
        )
        converged = (residual.abs() < tol).all(dim=-1)
        if bool(converged.all()):
            break
        active = ~converged
        jac = torch.empty(n, k, k, dtype=index.dtype, device=index.device)
        for col in range(k):
            step = torch.zeros_like(index)
            step[:, col] = h_step
            plus = torch.stack(
                [
                    _bilinear_sample_2d(g, index + step, clamp_query=False)
                    for g in grids
                ],
                dim=-1,
            )
            minus = torch.stack(
                [
                    _bilinear_sample_2d(g, index - step, clamp_query=False)
                    for g in grids
                ],
                dim=-1,
            )
            jac[:, :, col] = (plus - minus) / (2 * h_step)
        a, b = jac[:, 0, 0], jac[:, 0, 1]
        c, d = jac[:, 1, 0], jac[:, 1, 1]
        det = a * d - b * c
        col0_norm = torch.hypot(a, c)
        col1_norm = torch.hypot(b, d)
        singular_tol = (
            _CURVILINEAR_INTERP_SINGULAR_RTOL * col0_norm * col1_norm
        )
        singular = active & (det.abs() <= singular_tol)
        if bool(singular.any()):
            # an out-of-range query degrades to a singular Jacobian for a
            # mundane reason (the forward sampler clamps/flattens beyond
            # the grid edge, so its derivative vanishes there) -- check that
            # first, so the error names the actual cause (out of range)
            # rather than the more alarming but less useful "singular"
            # diagnosis whenever the two coincide. Scoped to `active` (not
            # yet converged) rows, same as `singular` itself.
            _curvilinear_out_of_bounds(
                index, sizes, dims, names_in_group, active
            )
            bad = int(singular.nonzero()[0])
            raise ValueError(
                "interp: the local Jacobian of the curvilinear coordinate "
                f"map over {dims!r} ({sorted(names_in_group)!r}) is "
                f"singular at query point {bad} (near index "
                f"{index[bad].tolist()}) -- the map isn't locally "
                "invertible there (a fold or a degenerate cell)"
            )
        inv_det = 1.0 / det
        r0, r1 = residual[:, 0], residual[:, 1]
        delta = torch.stack(
            [(d * r0 - b * r1) * inv_det, (-c * r0 + a * r1) * inv_det],
            dim=-1,
        )
        # freeze already-converged rows: `delta` for them may be garbage
        # (e.g. `inv_det` computed from a since-irrelevant near-singular
        # Jacobian at the fold apex above), but it is never selected here.
        delta = torch.where(
            active.unsqueeze(-1), delta, torch.zeros_like(delta)
        )
        index = torch.max(torch.min(index + delta, hi), lo)
    else:
        n_bad = int((~converged).sum())
        raise ValueError(
            "interp: the curvilinear coordinate map over "
            f"{dims!r} ({sorted(names_in_group)!r}) did not converge "
            f"within {_CURVILINEAR_INTERP_MAX_ITER} Newton iterations for "
            f"{n_bad} of {n} query point(s) -- the target may be outside "
            "the grid's coordinate range, or the map may not be locally "
            "invertible nearby"
        )
    _curvilinear_out_of_bounds(index, sizes, dims, names_in_group)
    return index


def _curvilinear_interp_pull(
    tensor: "XTensor",
    dims: tuple,
    names_in_group: list,
    indexers: tx.Mapping[str, tx.Any],
    method: tx.Any,
    bound: tx.Any,
    extrapolate: tx.Any,
    name: tx.Optional[str],
) -> "XTensor":
    """
    Interpolate a joint `interp` query over a general **curvilinear**
    coordinate (issue #82): given a target world value for each of
    `len(dims)` coordinate names spanning the same non-affine `dims` (e.g.
    `lat(y, x)`/`lon(y, x)`), invert the coordinate map with
    `_curvilinear_newton_indices` -- seeded from the existing brute-force
    nearest-neighbor `.sel` lookup (`_curvilinear_sel_indices`), one query
    point at a time -- to a fractional N-D index, then hand it to the same
    pull-and-wrap tail `_affine_interp_pull` uses
    (`_nd_interp_pull_and_wrap`).

    Scoped to exactly **2** spanned dims (the common `lat(y, x)`/`lon(y,
    x)` case; a higher-dimensional curvilinear coordinate raises
    `NotImplementedError` rather than a half-working generalisation) and to
    `method="nearest"`/`"linear"` (order 0/1) -- a higher spline order would
    need a true N-D fit to a scattered, non-uniform coordinate map, out of
    scope here (mirrors the irregular 1-D coordinate's own
    nearest/linear-only limit, #73).
    """
    if len(dims) != 2:
        raise NotImplementedError(
            f"interp: a joint curvilinear query over {dims!r} "
            f"({len(dims)} dims) isn't supported yet -- only the 2-D case "
            "(e.g. lat(y, x)/lon(y, x)) is implemented (#82)"
        )
    order = _interp_order(method)
    if order not in (0, 1):
        raise NotImplementedError(
            f"interp(method={method!r}) on a curvilinear coordinate "
            f"{sorted(names_in_group)!r}: only 'nearest'/'linear' are "
            "supported for a general (non-affine) curvilinear coordinate "
            "(#82) -- a higher order would need a true N-D spline fit to a "
            "scattered coordinate map, which isn't implemented"
        )
    bound = _get_option("interp_bound") if bound is None else bound
    extrapolate = (
        _get_option("interp_extrapolate")
        if extrapolate is None
        else extrapolate
    )

    axes = [_resolve_axis(tensor.names, d) for d in dims]
    sorted_dims = tuple(
        dims[i] for i in sorted(range(len(dims)), key=axes.__getitem__)
    )
    coords_bound = tensor.coords
    grids = []
    units = []
    sizes = None
    for nm in names_in_group:
        grid = coords_bound[nm]["value"]
        if sizes is None:
            sizes = tuple(grid.shape)
        grids.append(
            grid.as_subclass(Tensor).detach().to(_CURVILINEAR_SEL_DTYPE)
        )
        units.append(grid.units)

    per_name = []
    lengths = set()
    for nm, unit in zip(names_in_group, units):
        query, is_many = _query_values(indexers[nm], unit)
        lengths.add(query.numel())
        per_name.append((nm, query, is_many, unit))
    lengths.discard(1)
    if len(lengths) > 1:
        raise ValueError(
            f"interp: a joint curvilinear query over {dims!r} "
            f"({sorted(names_in_group)!r}) needs every coordinate's query "
            "to have the same length (a length-1 query broadcasts) -- got "
            f"lengths {sorted(lengths)!r}"
        )
    n = next(iter(lengths), 1)
    is_many_group = n > 1 or any(is_many for _, _, is_many, _ in per_name)

    # mirrors `_affine_interp_pull`'s own new-axis-name inference: an
    # indexer that is itself a named 1-D XTensor lends its name to the new
    # axis, an explicit `name=` wins outright, and disagreeing indexer
    # names with no override raise.
    inferred_name = None
    conflicting = None
    for nm in names_in_group:
        target = indexers[nm]
        if (
            isinstance(target, XTensor)
            and target.ndim == 1
            and target.names[0] is not None
        ):
            tname = target.names[0]
            if inferred_name is None:
                inferred_name = tname
            elif inferred_name != tname:
                conflicting = tname
    if name is None and conflicting is not None:
        raise ValueError(
            f"interp: a joint curvilinear query over {dims!r} "
            f"({sorted(names_in_group)!r}) was given named indexers that "
            f"disagree on the new axis's name ({inferred_name!r} vs. "
            f"{conflicting!r}) -- pass an explicit name= to resolve it"
        )
    name = name if name is not None else inferred_name

    broadcasted = []
    for nm, query, _, unit in per_name:
        if query.numel() == 1 and n != 1:
            query = query.expand(n)
        broadcasted.append((nm, query, unit))

    if n == 0:
        frac = torch.empty(0, len(dims), dtype=_CURVILINEAR_SEL_DTYPE)
    else:
        targets = torch.stack(
            [query.to(_CURVILINEAR_SEL_DTYPE) for _, query, _ in broadcasted],
            dim=-1,
        ).detach()  # (n, k)
        init = torch.empty(n, len(dims), dtype=_CURVILINEAR_SEL_DTYPE)
        # seeded one point at a time via the existing brute-force `.sel`
        # nearest lookup (issue #82's "reuse, don't reimplement" -- there is
        # no vectorized form of that lookup, see `_curvilinear_sel_indices`).
        for i in range(n):
            point_indexers = {
                nm: float(targets[i, j]) for j, nm in enumerate(names_in_group)
            }
            seed = _curvilinear_sel_indices(
                tensor, dims, names_in_group, point_indexers, "round", None
            )
            for j, d in enumerate(sorted_dims):
                init[i, j] = seed[d]
        frac = _curvilinear_newton_indices(
            grids, targets, init, sizes, dims, names_in_group
        )

    return _nd_interp_pull_and_wrap(
        tensor,
        dims,
        sorted_dims,
        frac,
        order,
        bound,
        extrapolate,
        name,
        is_many_group,
        broadcasted,
    )


def _nondim_coord_len(coord: tx.Any) -> int:
    """The number of positions in a non-dimension coordinate's values."""
    if isinstance(coord, Coordinate):
        return len(dict.__getitem__(coord, "value"))
    return len(coord)


def _parse_nondim_coord(key: str, spec: tx.Any, names: tuple) -> tuple:
    """
    Parse a `(dim(s), values)` non-dimension coordinate spec into `(dims,
    coord)`. `dim(s)` is a single dim name (1-D, rides along that one dim), or
    a sequence of several dim names for a coordinate spanning **several**
    dims at once: a compact **affine** map (Proposal 0005 step 3, `spacing` a
    vector, one component per dim) or an explicit **curvilinear** array with
    one tensor axis per dim (issue #82, e.g. arbitrary `lat(y, x)` values).

    A **1-D compact** spec isn't supported: unlike a dimension coordinate, a
    single-dim non-dimension one isn't re-sliced when its dim is (there is no
    per-component affine to update against just one dim's slicer the way
    the multi-dim form is) -- for an explicit or label coordinate that's
    caught by the length check on resize, but a compact coordinate binds to
    *any* size, so it would silently rebind to the wrong affine after a
    non-trivial slice instead of raising or dropping. Rejecting it here
    avoids that silent-wrong-values trap. A multi-dim explicit (curvilinear)
    coordinate has the same "rides through a slice unchanged" behaviour, but
    is fine with it: it is simply dropped (not rebound) once a spanned dim's
    size moves on without it, the same rule an ordinary 1-D explicit
    non-dimension coordinate already follows.
    """
    if not (isinstance(spec, tuple) and len(spec) == 2):
        raise ValueError(
            f"coords: {key!r} is not an axis; a non-dimension coordinate must "
            "be given as (dim, values) or (dims, values) for a multi-dim "
            "coordinate"
        )
    dims_spec, raw = spec
    if isinstance(dims_spec, str):
        dims = (dims_spec,)
    elif (
        isinstance(dims_spec, (list, tuple))
        and dims_spec
        and all(isinstance(d, str) for d in dims_spec)
    ):
        dims = tuple(dims_spec)
    else:
        raise ValueError(
            f"coords: {key!r} -- expected a dim name or a sequence of dim "
            f"names, got {dims_spec!r}"
        )
    for dim in dims:
        if dim not in names:
            raise ValueError(
                f"coords: no axis named {dim!r} in {tuple(names)}"
            )
    if len(set(dims)) != len(dims):
        raise ValueError(f"coords: {key!r} repeats a dim in {dims!r}")
    if _is_compact_coord(raw):
        if len(dims) == 1:
            raise NotImplementedError(
                f"coords: {key!r} -- a compact (spacing/origin) "
                "non-dimension coordinate over a single dim isn't supported "
                "yet (it wouldn't survive slicing its dim correctly); use "
                "an explicit tensor of values instead"
            )
        return dims, _make_affine_coordinate(raw, len(dims))
    if len(dims) > 1:
        return dims, _make_curvilinear_coordinate(key, raw, dims)
    if _is_explicit_coord(raw):
        coord = _make_coordinate(raw)
    else:
        coord = _promote_numeric_labels(key, tuple(raw))
    return dims, coord


def _check_nondim_len(key: str, dim: str, coord: tx.Any, size: int) -> None:
    """Validate a non-dimension coordinate's length against its dim's size."""
    length = _nondim_coord_len(coord)
    if length != size:
        raise ValueError(
            f"coords: non-dimension coordinate {key!r} has {length} values "
            f"for dim {dim!r} of size {size}"
        )


def _coords_of(tensor: tx.Any) -> dict:
    """The coordinate labels of `tensor` (empty for a plain / non tensor)."""
    if isinstance(tensor, XTensor):
        return tensor.coords
    return {}


def _coords_for(input: XTensor, result_names: tuple) -> dict:
    """
    Keep the coordinates (dimension or non-dimension, labels or numeric) all
    of whose `dims` survive (by name) into `result_names`. A merged / split /
    removed axis loses its name, so any coordinate keyed on it -- or merely
    *riding* on it (Proposal 0005) -- drops automatically. `key in valid`
    first restricts to `input`'s own currently-valid coordinates (already
    filtered by size/name on `input`); the raw (unbound) coordinate is kept
    rather than `valid`'s bound copy, since a survivor's dim size is by
    construction unchanged, so the result rebinds it identically on read.
    """
    kept = {name for name in result_names if name is not None}
    valid = _coords_of(input)
    stored = input.__dict__.get("_coords") or {}
    return {
        key: (dims, coord)
        for key, (dims, coord) in stored.items()
        if key in valid and all(dim in kept for dim in dims)
    }


def _coords_dropping(input: XTensor, *dims: tx.Optional[str]) -> dict:
    """
    `input`'s coordinates, packed into unified storage, minus every
    coordinate that touches any of `dims` -- its own dimension coordinate, or
    a non-dimension coordinate (Proposal 0005) *riding* on it. For ops whose
    positions along `dims` no longer correspond to the stored ones (sort,
    flip, roll, gather, index_select, ...): the caller re-adds a transformed
    dimension coordinate for a touched dim itself when it can track one (e.g.
    flip reverses, roll rotates); a rider is conservatively dropped outright.
    """
    touched = set(dims)
    valid = _coords_of(input)
    stored = input.__dict__.get("_coords") or {}
    return {
        key: (entry_dims, coord)
        for key, (entry_dims, coord) in stored.items()
        if key in valid and not touched & set(entry_dims)
    }


def _selector_value(selector: tx.Any, unit: tx.Optional[str]) -> float:
    """
    A numeric selector as a plain float in the coordinate's position `unit`. A
    bare number is taken as already in that unit; a unitful selector (`"2mm"`,
    `(2, "mm")`, a pint quantity, ...) has its magnitude/unit split regardless
    of a backend -- only the actual **conversion** into a *different* unit
    (e.g. `"2000ms"` onto a `"s"` coordinate) needs one active.
    """
    if isinstance(selector, (int, float)):
        return float(selector)
    quantity = _as_unitful(selector)
    value, sel_unit = quantity["value"], quantity["unit"]
    if (
        unit
        and sel_unit
        and _units.active()
        and not _units.equal(sel_unit, unit)
    ):
        value = value * _units.factor(sel_unit, unit)
    return float(value)


def _numeric_select_compact(
    coord: "Coordinate",
    selector: tx.Any,
    mode: str,
    tolerance: tx.Any,
    name: str,
) -> tx.Any:
    """
    `_numeric_select` for a **compact** coordinate: `origin`/`spacing` give an
    O(1) closed-form inverse (`index = (value - origin) / spacing`), so this
    never materialises `["value"]` or searches it (issue #110) -- the whole
    point of the compact representation is to avoid exactly that for a large
    regular grid. `coord` must already be size-bound (`coord._bound(size)`,
    what `.coords` always returns), so `coord._size` is available. Falls back
    to materialising and searching (still correct, just not O(1)) only for
    the rare target `_closed_form_sel_index` can't resolve locally
    (`_ClosedFormMiss`) -- see that function's docstring.
    """
    spacing = dict.__getitem__(coord, "spacing")
    origin = dict.get(coord, "origin")
    unit = spacing["unit"]
    step = float(spacing["value"])
    base = float(origin["value"]) if origin is not None else 0.0
    size = coord._size
    is_many = isinstance(selector, list)
    wanted = list(selector) if is_many else [selector]
    tol = None if tolerance is None else _selector_value(tolerance, unit)
    # a single-tick (or empty) coordinate has no direction of its own to
    # speak of -- match `_numeric_select`'s explicit-coordinate convention
    # of defaulting to ascending in that case, rather than trusting a
    # declared negative spacing that has nothing to actually order.
    ascending = True if size <= 1 else step > 0
    materialised_values = None  # lazily materialised only on a fallback
    positions = []
    for one in wanted:
        target = _selector_value(one, unit)
        if math.isnan(target):
            raise ValueError(f"sel: target {target!r} is not a number")
        if step == 0:
            # degenerate: every tick sits at `base` -- round always matches
            # it (index 0, the same tie-break `argmin` gives an all-equal
            # array, which is ascending per `(diffs >= 0).all()`, so
            # prev->floor, next->ceil); floor/ceil are valid only from the
            # matching side.
            if mode == "prev":
                eff_mode = "floor"
            elif mode == "next":
                eff_mode = "ceil"
            else:
                eff_mode = mode
            if (
                eff_mode == "round"
                or (eff_mode == "floor" and base <= target)
                or (eff_mode == "ceil" and base >= target)
            ):
                j = 0
            else:
                j = None
        else:
            try:
                j = _closed_form_sel_index(
                    base, step, target, mode, ascending, size
                )
            except _ClosedFormMiss:
                if materialised_values is None:
                    # built directly in float64 -- matching the closed-form
                    # walk's own arithmetic -- rather than materialising
                    # via `coord["value"]` (which computes in the tensor's
                    # default, float32, dtype: `torch.arange(size)*step`
                    # already loses precision there) and upcasting
                    # afterwards, which cannot recover what's already lost.
                    # This regime is exactly where that precision gap
                    # matters (an astronomically large `|base/step|`, the
                    # only way this fallback is ever reached).
                    materialised_values = (
                        base + torch.arange(size, dtype=torch.float64) * step
                    )
                j = _pick_sel_index(
                    materialised_values, target, mode, ascending
                )
        if j is None:
            raise ValueError(f"sel: no {mode} tick for {one!r} on {name!r}")
        gap = abs(base + j * step - target)
        _check_sel_tolerance(gap, tol, target, mode, one, name)
        positions.append(j)
    return positions if is_many else positions[0]


def _numeric_select(
    coord: "Coordinate",
    selector: tx.Any,
    mode: str,
    tolerance: tx.Any,
    name: str,
) -> tx.Any:
    """
    Resolve a value-based selector against a numeric `Coordinate` to integer
    position(s) (Proposal 0004), snapping per `mode` (see `sel`). `tolerance`
    (a delta in the position unit) caps the gap; `None` is unbounded, `0` is
    exact (up to float epsilon). A **compact** coordinate resolves in closed
    form (`_numeric_select_compact`, issue #110); an **explicit** one
    materialises and searches, below.
    """
    if coord._compact():
        return _numeric_select_compact(coord, selector, mode, tolerance, name)
    materialised = coord["value"]
    values = materialised.as_subclass(Tensor)
    unit = materialised.units
    # a `list` selects several positions; a `tuple` is a unitful (value, unit)
    is_many = isinstance(selector, list)
    wanted = list(selector) if is_many else [selector]
    tol = None if tolerance is None else _selector_value(tolerance, unit)
    ascending = True
    if mode in ("prev", "next") and values.numel() > 1:
        diffs = values[1:] - values[:-1]
        if bool((diffs >= 0).all()):
            ascending = True
        elif bool((diffs <= 0).all()):
            ascending = False
        else:
            raise ValueError(
                f"sel: mode={mode!r} needs a monotonic coordinate on {name!r}"
            )
    positions = []
    for one in wanted:
        target = _selector_value(one, unit)
        j = _pick_sel_index(values, target, mode, ascending)
        if j is None:
            raise ValueError(f"sel: no {mode} tick for {one!r} on {name!r}")
        gap = float((values[j] - target).abs())
        _check_sel_tolerance(gap, tol, target, mode, one, name)
        positions.append(j)
    return positions if is_many else positions[0]


def _numeric_select_range(
    coord: "Coordinate", selector: slice, name: str
) -> slice:
    """
    Resolve a `slice(lo, hi)` value-range selector against a numeric
    `Coordinate` to an integer position `slice` (#109) -- half-open
    (`lo <= value < hi`), unit-aware, on both compact and explicit
    coordinates. Bounds are compared **numerically**, independent of the
    order they're given in or of the coordinate's own direction:
    `slice(lo, hi)` and `slice(hi, lo)` are the same request. A single bound
    is positional (`slice.start` alone -> `value >= start`; `slice.stop`
    alone -> `value < stop`); an out-of-range (including `+/-inf`) bound
    clamps to an empty or full range rather than raising (#96's empty-axis
    precedent); a `nan` bound raises, since no comparison to it is
    well-formed. `step` has no value-range meaning here and is rejected
    outright, rather than repurposed to signal an open/closed bound -- that
    would overload one field with two unrelated meanings, the trap #93
    already fixed.
    """
    if selector.step is not None:
        raise ValueError(
            f"sel: a range selector on {name!r} does not take a step "
            f"({selector.step!r}) -- slice(lo, hi) only"
        )
    if coord._compact():
        spacing = dict.__getitem__(coord, "spacing")
        unit = spacing["unit"]
        size = coord._size
    else:
        materialised = coord["value"]
        values = materialised.as_subclass(Tensor)
        unit = materialised.units
        size = values.numel()
    start = (
        None
        if selector.start is None
        else _selector_value(selector.start, unit)
    )
    stop = (
        None if selector.stop is None else _selector_value(selector.stop, unit)
    )
    for bound in (start, stop):
        if bound is not None and math.isnan(bound):
            raise ValueError(
                f"sel: a range selector on {name!r} has a NaN bound"
            )
    if start is not None and stop is not None:
        lo, hi = (start, stop) if start <= stop else (stop, start)
    else:
        lo, hi = start, stop
    if coord._compact():
        return _compact_range_slice(coord, lo, hi, size)
    return _explicit_range_slice(values, lo, hi, name)


#: `interp` method names -> integer spline order (mirrors `fiery.interpol`).
_INTERP_ORDERS = {
    "nearest": 0,
    "zeroth": 0,
    "linear": 1,
    "first": 1,
    "quadratic": 2,
    "second": 2,
    "cubic": 3,
    "third": 3,
}


def _interp_order(method: tx.Any) -> int:
    """The integer spline order for an `interp` `method` (a name or an int)."""
    if isinstance(method, int) and not isinstance(method, bool):
        return method
    try:
        return _INTERP_ORDERS[method]
    except (KeyError, TypeError):
        raise ValueError(
            f"interp: unknown method {method!r}; use an int order or one of "
            f"{sorted(_INTERP_ORDERS)}"
        ) from None


def _query_values(target: tx.Any, unit: tx.Optional[str]) -> tx.Any:
    """
    A numeric `interp` query as a 1-D float tensor in the position `unit`, plus
    whether it **keeps** the axis (a list / 1-D tensor) or **drops** it (a
    scalar). A bare tensor is taken as already in the position unit (and its
    gradient rides through); everything else goes through `_selector_value`, so
    a unitful query (`"2s"`, `(2, "s")`, ...) is converted first.
    """
    if isinstance(target, Tensor):
        flat = target.reshape(-1)
        if not flat.is_floating_point():
            flat = flat.to(torch.get_default_dtype())
        return flat, target.ndim > 0
    is_many = isinstance(target, list)
    items = target if is_many else [target]
    values = [_selector_value(one, unit) for one in items]
    query = torch.tensor(values, dtype=torch.get_default_dtype())
    return query, is_many


def _irregular_frac(values: Tensor, query: Tensor, name: str) -> Tensor:
    """
    Fractional index for `query` against an **irregular** (non-uniform,
    strictly monotonic) 1-D coordinate `values`, via `torch.searchsorted` +
    a local linear inverse (issue #73): `searchsorted` brackets each query
    between two adjacent ticks `values[k] <= query <= values[k+1]` (ascending
    order; a descending coordinate is bracketed the same way by searching
    its reverse), then `k + (query - values[k]) / (values[k+1] - values[k])`
    is the exact fractional index -- it inverts the same piecewise-linear
    map the nearest/linear pull already samples between those two ticks, so
    the round trip is exact (unlike a higher order, whose spline basis is
    uniform in index space -- see #81). Differentiable w.r.t. both `query`
    and `values`: only the *search* (which bracket a query falls in) runs on
    detached copies, since an index has no useful gradient; the returned
    fraction is computed from the original tensors. `values` is guaranteed
    1-D here -- `_make_coordinate` rejects a non-1-D coordinate at
    construction (#97), so there's no need to re-check it per consumer.
    """
    n = values.numel()
    if n < 2:
        raise ValueError(
            f"interp: irregular coordinate {name!r} needs at least 2 points"
        )
    ticks = values.detach()  # the check is a predicate: no graph needed
    diffs = ticks[1:] - ticks[:-1]
    if bool((diffs > 0).all()):
        ascending, ordered = True, values
    elif bool((diffs < 0).all()):
        ascending, ordered = False, values.flip(0)
    else:
        # point at the first offending pair -- a tie (a repeated tick, easy to
        # hit by accident in float32) reads as "not monotonic" otherwise, with
        # nothing to say *where*.
        wanted = diffs > 0 if bool(diffs[0] > 0) else diffs < 0
        j = int(wanted.logical_not().long().argmax())
        raise ValueError(
            f"interp: irregular coordinate {name!r} must be strictly "
            f"monotonic (ascending or descending); ticks {j} and {j + 1} "
            f"are {float(ticks[j])} and {float(ticks[j + 1])}"
        )
    # a coordinate sliced with a step (`x[::2]`) is a strided view, which
    # `searchsorted` copies (and warns about) -- do it once, quietly.
    k = (
        torch.searchsorted(
            ordered.detach().contiguous(), query.detach(), right=False
        )
        - 1
    )
    k = k.clamp(0, n - 2)
    v0, v1 = ordered[k], ordered[k + 1]
    frac = k.to(query.dtype) + (query - v0) / (v1 - v0)
    return frac if ascending else (n - 1) - frac


def _interpol() -> tx.Any:
    """The optional `fiery.interpol` backend, or `None` if not installed."""
    try:
        from fiery import interpol
    except ImportError:
        return None
    return interpol


def _nearest_gather(
    moved: Tensor, frac: Tensor, length: int, bound: tx.Any
) -> Tensor:
    """
    Built-in nearest-neighbour pull along the **last** axis of `moved` (no
    backend). The fractional indices `frac` round to the closest tick; an
    out-of-range index is resolved by `bound` -- clamp for
    ``"replicate"``/``"nearest"``, wrap for ``"dft"``/``"wrap"``. Any other
    boundary needs the `fiery.interpol` backend.
    """
    idx = frac.round().long()
    if bound in ("replicate", "nearest", 1):
        idx = idx.clamp(0, length - 1)
    elif bound in ("dft", "wrap", 6):
        idx = idx.remainder(length)
    else:
        raise ImportError(
            f"interp method='nearest' with bound {bound!r} needs the "
            "fiery.interpol backend; install fiery-xtensor[interp]"
        )
    return moved.index_select(-1, idx)


def _interp_pull(
    raw: Tensor,
    axis: int,
    frac: Tensor,
    order: int,
    bound: tx.Any,
    extrapolate: tx.Any,
) -> Tensor:
    """
    Interpolate `raw` along `axis` at fractional indices `frac` (see `interp`).

    Order 0 (nearest) is done in-package -- a gather -- so it needs no backend;
    order >= 1 delegates to `fiery.interpol.grid_pull`, the optional
    `fiery-xtensor[interp]` dependency.
    """
    n = int(frac.shape[0])
    moved = torch.movedim(raw, axis, -1)  # (*rest, length)
    rest = moved.shape[:-1]
    length = int(moved.shape[-1])
    interpol = _interpol()
    if order == 0 and interpol is None:
        out = _nearest_gather(moved, frac, length, bound)
    else:
        if interpol is None:
            raise ImportError(
                "interp with order >= 1 needs the fiery.interpol backend; "
                "install fiery-xtensor[interp]"
            )
        flat = moved.reshape(-1, 1, length)
        if not flat.is_floating_point():
            flat = flat.to(torch.get_default_dtype())
        grid = frac.reshape(1, n, 1).to(flat).expand(flat.shape[0], n, 1)
        pulled = interpol.grid_pull(
            flat,
            grid,
            interpolation=order,
            bound=bound,
            extrapolate=extrapolate,
        )  # (batch, 1, n)
        out = pulled.reshape(*rest, n)
    return torch.movedim(out, -1, axis)


def _nd_nearest_pull(
    moved: Tensor, frac: Tensor, sizes: tuple, bound: tx.Any
) -> Tensor:
    """
    Built-in N-D nearest-neighbour pull (no backend), the multi-dim
    counterpart of `_nearest_gather` for a joint affine `interp` query
    (issue #82 phase 2): nearest-neighbour rounding is separable per axis
    (no cross-axis interpolation weight is ever needed), so each of
    `frac`'s `len(sizes)` columns is rounded and bounded independently,
    then the flat (row-major) index they jointly address is gathered in
    one shot -- `moved`'s last `len(sizes)` axes flattened to one first,
    matching how `torch.reshape` itself linearises them.
    """
    ndim = len(sizes)
    idx = frac.round().long()  # (n, ndim)
    flat_idx = None
    for k in range(ndim):
        col = idx[:, k]
        length = sizes[k]
        if bound in ("replicate", "nearest", 1):
            col = col.clamp(0, length - 1)
        elif bound in ("dft", "wrap", 6):
            col = col.remainder(length)
        else:
            raise ImportError(
                f"interp method='nearest' with bound {bound!r} needs the "
                "fiery.interpol backend; install fiery-xtensor[interp]"
            )
        flat_idx = col if flat_idx is None else flat_idx * length + col
    flat_moved = moved.reshape(*moved.shape[:-ndim], -1)
    return flat_moved.index_select(-1, flat_idx)


def _check_no_affine_curvilinear_dims_conflict(
    stored: dict, indexers: tx.Mapping[str, tx.Any]
) -> None:
    """
    Raise a clear error if the same *set* of dims is spanned by both a
    compact **affine** coordinate name and a general **curvilinear**
    coordinate name, both present in the same `interp` call (issue #82) --
    e.g. `y, x` carrying both a `p`/`q` affine map and a `lat`/`lon`
    curvilinear one, queried together. Each of `_affine_interp_group`/
    `_curvilinear_interp_group` only ever sees the coordinates of its own
    kind, so without this upfront check the two would silently race: one
    phase mutates the tensor (dropping `dims`) before the other ever looks
    for its own group, which would otherwise surface as a confusing "no
    axis" error rather than naming the actual conflict. Checked before
    either phase runs, so neither has mutated anything yet.

    Compares `dims` as an unordered `frozenset` with a non-empty-
    intersection test, not an exact-tuple equality test (review of #166,
    finding #7): an affine coordinate spanning `("y", "x")` and a
    curvilinear one spanning `("x", "y")` -- the same two axes, written in
    the other order -- previously slipped past an exact-tuple comparison
    entirely, as did a *partially* overlapping pair (e.g. `("y", "x")` vs.
    `("x", "z")`, sharing only `"x"`); both produce exactly the confusing
    "no axis" error this guard exists to prevent.
    """
    affine_dims = []
    curvilinear_dims = []
    for nm in indexers:
        entry = stored.get(nm)
        if entry is None:
            continue
        dims, coord = entry
        if len(dims) > 1 and isinstance(coord, Coordinate):
            group = affine_dims if coord._compact() else curvilinear_dims
            group.append(frozenset(dims))
    for a in affine_dims:
        for c in curvilinear_dims:
            overlap = a & c
            if overlap:
                raise ValueError(
                    f"interp: dim(s) {sorted(overlap)!r} are spanned by "
                    "both a compact affine coordinate and a general "
                    "curvilinear coordinate, both queried in the same "
                    "call -- pass one or the other"
                )


def _affine_interp_pull(
    tensor: "XTensor",
    dims: tuple,
    names_in_group: list,
    indexers: tx.Mapping[str, tx.Any],
    method: tx.Any,
    bound: tx.Any,
    extrapolate: tx.Any,
    name: tx.Optional[str],
) -> "XTensor":
    """
    Solve the closed-form affine inverse for a joint `interp` query (issue
    #82 phase 2): given a target world value for each of `len(dims)`
    coordinate names spanning the same `dims`, invert `A` **once** (shared
    across every query point, unlike `.sel`'s single-point solve) to a
    **fractional** index -- never rounded -- then genuinely pull the
    tensor's values at those N-D fractional positions via
    `fiery.interpol.grid_pull` (order >= 1, or order 0 with the backend
    installed) or a built-in separable nearest gather (order 0, no
    backend -- `_nd_nearest_pull`).

    A **scalar** query for every name in the group is a single point: the
    `len(dims)` spanned dims are dropped entirely (the existing `.sel`/
    `.interp` scalar-drops-the-axis convention). Any **list**/tensor query
    makes the whole group "many": every name's query broadcasts to a
    common length `N`, and the `len(dims)` spanned dims collapse into
    **one new axis** of `N` sampled points, inserted at the left-most
    spanned dim's position -- not an outer-product grid, since the dims
    are coupled (see `interp`'s docstring). The new axis is named `name`
    if given, else the shared name of any queried indexer that is itself
    a named 1-D `XTensor` (mirroring xarray's own vectorized-indexing
    convention, where the *indexer*'s own dim name becomes the result's
    new dimension) -- disagreeing indexer names with no `name=` override
    to resolve them raises. It carries every queried name's own sampled
    values as a riding coordinate (only when a name was resolved -- an
    unnamed axis can't be keyed).
    """
    order = _interp_order(method)
    bound = _get_option("interp_bound") if bound is None else bound
    extrapolate = (
        _get_option("interp_extrapolate")
        if extrapolate is None
        else extrapolate
    )
    stored = tensor.__dict__.get("_coords") or {}
    per_name = []
    lengths = set()
    for nm in names_in_group:
        _, coord = stored[nm]
        spacing = dict.__getitem__(coord, "spacing")
        origin = dict.get(coord, "origin")
        vec = spacing["value"]
        # solved in float64 regardless of the spacing's own/default dtype,
        # matching the closed-form conventions of both the 1-D interp path
        # and .sel's joint query (#82 phase 1 review finding #3).
        if isinstance(vec, Tensor):
            vec = vec.to(torch.float64)
        else:
            vec = torch.as_tensor(vec, dtype=torch.float64)
        base = float(origin["value"]) if origin is not None else 0.0
        query, is_many = _query_values(indexers[nm], spacing["unit"])
        query = query.to(torch.float64)
        lengths.add(query.numel())
        per_name.append((nm, vec, base, query, is_many, spacing["unit"]))
    lengths.discard(1)
    if len(lengths) > 1:
        raise ValueError(
            f"interp: a joint affine query over {dims!r} "
            f"({sorted(names_in_group)!r}) needs every coordinate's query "
            "to have the same length (a length-1 query broadcasts) -- got "
            f"lengths {sorted(lengths)!r}"
        )
    n = next(iter(lengths), 1)
    is_many_group = n > 1 or any(is_many for *_, is_many, _ in per_name)
    # if a query is itself a named 1-D XTensor, its own name is what the new
    # axis should be called -- mirroring xarray's own vectorized-indexing
    # convention (the shared dim name of the *indexer* arrays becomes the
    # result's new dimension, not a separate parameter). An explicit `name=`
    # still wins outright; two indexers disagreeing on a name (with no
    # `name=` override to resolve it) is ambiguous and raises.
    inferred_name = None
    conflicting = None
    for nm in names_in_group:
        target = indexers[nm]
        if (
            isinstance(target, XTensor)
            and target.ndim == 1
            and target.names[0] is not None
        ):
            tname = target.names[0]
            if inferred_name is None:
                inferred_name = tname
            elif inferred_name != tname:
                conflicting = tname
    if name is None and conflicting is not None:
        raise ValueError(
            f"interp: a joint affine query over {dims!r} "
            f"({sorted(names_in_group)!r}) was given named indexers that "
            f"disagree on the new axis's name ({inferred_name!r} vs. "
            f"{conflicting!r}) -- pass an explicit name= to resolve it"
        )
    name = name if name is not None else inferred_name
    broadcasted = []
    for nm, vec, base, query, _, unit in per_name:
        # `!= 1`, not `> 1`: an empty query (n == 0, #96's empty-axis case)
        # still needs a length-1 sibling broadcast *down* to empty, or
        # torch.stack sees mismatched [0]-vs-[1] rows and raises.
        if query.numel() == 1 and n != 1:
            query = query.expand(n)
        broadcasted.append((nm, vec, base, query, unit))
    matrix = torch.stack([vec for _, vec, _, _, _ in broadcasted])  # (k, k)
    rhs = torch.stack(
        [query - base for _, _, base, query, _ in broadcasted]
    )  # (k, n)
    try:
        frac = torch.inverse(matrix) @ rhs  # (k, n)
    except RuntimeError as exc:
        if "singular" not in str(exc).lower():
            raise
        raise ValueError(
            f"interp: the affine map over {dims!r} "
            f"({sorted(names_in_group)!r}) isn't invertible: {exc}"
        ) from None
    frac = frac.T.contiguous()  # (n, k)

    riding = [(nm, query, unit) for nm, _, _, query, unit in broadcasted]
    return _nd_interp_pull_and_wrap(
        tensor,
        dims,
        dims,
        frac,
        order,
        bound,
        extrapolate,
        name,
        is_many_group,
        riding,
    )


def _nd_interp_pull_and_wrap(
    tensor: "XTensor",
    dims: tuple,
    axes_order: tuple,
    frac: Tensor,
    order: int,
    bound: tx.Any,
    extrapolate: tx.Any,
    name: tx.Optional[str],
    is_many_group: bool,
    riding: list,
) -> "XTensor":
    """
    Pull `tensor`'s data at the N-D fractional index `frac` (`(n, k)`,
    columns matching `axes_order`) and wrap the result with the right
    names/coords -- the shared tail of a joint `interp` query, whichever
    coordinate form produced `frac`: the affine closed-form inverse
    (`_affine_interp_pull`) or the curvilinear Newton solve
    (`_curvilinear_interp_pull`) both hand off here for the actual
    `fiery.interpol.grid_pull` (or built-in nearest gather, `order == 0`
    with no backend) kernel call, and the same "a scalar query for every
    name drops the spanned dims; a list/tensor collapses them into one new
    axis" convention (see `interp`'s docstring).

    `dims` is the coordinate's own spanned dims (used only to drop the
    superseded coordinate(s)); `axes_order` is the order `frac`'s columns
    are in, which need not match `dims`'s own order for a curvilinear
    coordinate (bound to the host tensor's axis order, not its construction
    order -- see `Coordinate._bound_curvilinear`). `riding` is a `(coord
    name, broadcasted query values, unit)` list, reattached as a coordinate
    on the new axis when the query is "many" and `name` is resolved.
    """
    axes = [_resolve_axis(tensor.names, d) for d in axes_order]
    raw = tensor.as_subclass(Tensor)
    # `torch.movedim` takes a list of sources/destinations in one call, so
    # every spanned axis lands at the end, in `axes_order`, without the
    # index-shifting hazard of several sequential single-axis calls.
    moved = torch.movedim(raw, axes, list(range(-len(axes), 0)))
    rest = moved.shape[: -len(axes)]
    sizes = tuple(moved.shape[-len(axes) :])
    batch = 1
    for s in rest:
        batch *= s
    n = int(frac.shape[0])

    if n == 0:
        # an empty query -> an empty new axis, mirroring #96's empty-axis
        # convention for the ordinary 1-D interp path.
        pulled = torch.empty(*rest, 0, dtype=moved.dtype, device=moved.device)
    elif order == 0 and _interpol() is None:
        pulled = _nd_nearest_pull(moved, frac.to(moved.device), sizes, bound)
    else:
        interpol = _interpol()
        if interpol is None:
            raise ImportError(
                "interp with order >= 1 needs the fiery.interpol backend; "
                "install fiery-xtensor[interp]"
            )
        flat = moved.reshape(batch, 1, *sizes)
        if not flat.is_floating_point():
            flat = flat.to(torch.get_default_dtype())
        ndim = len(axes_order)
        # `grid_pull` treats `grid` as a *dense* output grid, one axis per
        # input spatial dim (`(batch, *outshape, dim)`, `len(outshape) ==
        # dim` -- mirroring `torch.nn.functional.grid_sample`'s N-D
        # convention, not a flat scattered-point-list API), so an arbitrary
        # point list needs `ndim - 1` synthetic singleton output axes ahead
        # of the real one (verified empirically against the installed
        # backend for both 2-D and 3-D `dims`).
        pad = (1,) * (ndim - 1)
        grid = (
            frac.reshape(1, *pad, n, ndim)
            .to(flat)
            .expand(batch, *pad, n, ndim)
        )
        pulled = interpol.grid_pull(
            flat,
            grid,
            interpolation=order,
            bound=bound,
            extrapolate=extrapolate,
        )  # (batch, 1, *pad, n)
        pulled = pulled.reshape(*rest, n)

    insert_at = min(axes)
    if is_many_group:
        result_raw = torch.movedim(pulled, -1, insert_at)
    else:
        result_raw = pulled.reshape(rest)

    remaining_names = []
    for i, nm0 in enumerate(tensor.names):
        if i in axes:
            if i == insert_at and is_many_group:
                remaining_names.append(name)
            continue
        remaining_names.append(nm0)

    out = _carry(tensor, result_raw)
    out.names = tuple(remaining_names)
    new_coords = _coords_dropping(tensor, *dims)
    if is_many_group and name is not None:
        for nm, query, unit in riding:
            values = query.to(torch.get_default_dtype())
            new_coords[nm] = (
                (name,),
                Coordinate(value=XTensor(values, units=unit)),
            )
    out._coords = new_coords
    return out


def _reslice_coordinate_values(
    values: tx.Any, slicer: _SmartSlicerT
) -> tx.Any:
    """
    Index a coordinate's materialised position array by `slicer`, explicitly
    carrying over its position unit (`_unit_of(values)`) rather than relying
    on it being propagated automatically.

    `_slice_coordinate` is reached both from inside a registered override
    (`narrow`/`select`/`split`/`flip`/`roll`/`index_select`, ...) and from
    plain `__getitem__`/`.sel`/`.isel` (not themselves registered overrides).
    The two paths differ in exactly the way that matters here: an override
    runs under `_no_dispatch()` (`_extended.py`), which suppresses the
    *generic* fallback (`_extended.py`'s `__torch_function__`, the "ops
    without a name-aware override" branch) that would otherwise copy
    `_data_units` from `values` onto `values[slicer]` for this un-overridden
    indexing op -- so left to itself, `values[slicer]` silently comes back
    unitless there, even though `values` itself carries a unit (issue #165).
    `__getitem__`/`.sel`/`.isel` never run under `_no_dispatch()`, so that
    generic fallback already carries the unit for them without this helper --
    they were never actually affected, despite reaching the same function.
    """
    sliced = values[slicer]
    if isinstance(values, XTensor) and isinstance(sliced, XTensor):
        sliced._data_units = _unit_of(values)
    return sliced


def _slice_coordinate(
    coord: Coordinate, slicer: _SmartSlicerT, size: int
) -> tx.Optional[Coordinate]:
    """
    Apply a 1-D `slicer` to a numeric `Coordinate` on an axis of `size`. A
    **basic slice** stays exact: a compact coordinate updates its affine
    (`spacing *= step`, `origin += start * spacing`); an explicit one slices
    its values. An **advanced** index materialises a compact coordinate to
    explicit first. Returns `None` for a slicer that cannot be applied (the
    coordinate then drops).
    """
    if isinstance(slicer, slice):
        start, stop, step = slicer.indices(size)
        if start == 0 and step == 1 and stop >= size:
            return coord  # a full slice leaves the coordinate untouched
        if coord._compact():
            spacing = dict.__getitem__(coord, "spacing")
            origin = dict.get(coord, "origin")
            base = origin["value"] if origin is not None else 0
            out = Coordinate()
            out["spacing"] = _units.Unitful(
                value=spacing["value"] * step, unit=spacing["unit"]
            )
            out["origin"] = _units.Unitful(
                value=base + start * spacing["value"], unit=spacing["unit"]
            )
            return out
        values = dict.__getitem__(coord, "value")
        return Coordinate(value=_reslice_coordinate_values(values, slicer))
    if arrayutils._is_boolean_index(slicer) or arrayutils._is_advanced_index(
        slicer
    ):
        if coord._compact():
            values = coord._bound(size)["value"]
        else:
            values = dict.__getitem__(coord, "value")
        return Coordinate(value=_reslice_coordinate_values(values, slicer))
    return None


def _slice_affine_coordinate(
    coord: Coordinate, dims: tuple, pieces: dict, sizes: dict
) -> tx.Optional[tuple]:
    """
    Apply one slicer per spanned dim to a compact coordinate that may span
    **several** dims (a non-dimension coordinate, `len(dims) >= 1`; the
    genuinely multi-dim case is Proposal 0005 step 3's affine coordinate --
    `spacing` a vector, one component per dim, `origin` a single shared
    scalar). Exact per-component, 0001's trick generalised:

    - a **basic slice** on a dim updates that dim's component exactly
      (`origin += start * component`, `component *= step`) and keeps the dim;
    - an **integer** index folds that dim out entirely (`origin += index *
      component`), dropping it from `dims`/`spacing` -- the coordinate
      survives with one fewer dim (possibly collapsing to an ordinary 1-D
      compact non-dimension coordinate);
    - anything else (boolean / advanced indexing) can't stay affine, so the
      *whole* coordinate is dropped.

    Returns `(new_dims, new_coord)`, or `None` to drop the coordinate
    (either because an unsupported indexer touched one of its dims, or
    because every dim it spanned was folded away by integer indices, leaving
    no axis for it to ride on).
    """
    if all(
        isinstance(pieces[dim], slice)
        and pieces[dim].indices(sizes[dim]) == (0, sizes[dim], 1)
        for dim in dims
    ):
        return dims, coord  # every dim is a full no-op slice; nothing to do
    spacing = dict.__getitem__(coord, "spacing")
    origin = dict.get(coord, "origin")
    unit = spacing["unit"]
    components = spacing["value"]
    # a coordinate already collapsed to a single dim stores a bare scalar
    # `spacing` (the ordinary 1-D compact form), not a length-1 vector -- only
    # index into it when there is more than one component to pick from.
    is_vector = len(dims) > 1
    base = origin["value"] if origin is not None else 0
    new_dims = []
    new_components = []
    for i, dim in enumerate(dims):
        piece = pieces[dim]
        size = sizes[dim]
        component = components[i] if is_vector else components
        if isinstance(piece, slice):
            start, _stop, step = piece.indices(size)
            base = base + start * component
            new_dims.append(dim)
            new_components.append(component * step)
        elif isinstance(piece, int):
            index = piece + size if piece < 0 else piece
            base = base + index * component
        else:
            return None  # boolean / advanced index: can't stay affine
    if not new_dims:
        return None  # every spanned dim was folded away; no axis left to ride
    new_coord = Coordinate()
    new_coord["spacing"] = _units.Unitful(
        value=new_components[0]
        if len(new_components) == 1
        else torch.stack(new_components),
        unit=unit,
    )
    new_coord["origin"] = _units.Unitful(value=base, unit=unit)
    return tuple(new_dims), new_coord


def _names_of(tensor: tx.Any) -> tuple:
    """
    Axis names of a tensor: its `names` if a `XTensor`, all-`None` for a
    plain tensor, and `()` for a non-tensor (e.g. a Python scalar operand).
    """
    if isinstance(tensor, XTensor):
        return tensor.names
    if isinstance(tensor, Tensor):
        return (None,) * tensor.ndim
    return ()


def _unit_of(x: tx.Any) -> tx.Optional[str]:
    """The data unit of `x`, or `None` (a plain tensor/scalar is unitless)."""
    return x.__dict__.get("_data_units") if isinstance(x, XTensor) else None


def _attach_unit(x: XTensor, operand: tx.Any, op: str) -> XTensor:
    """
    Combine a backend `Unit`/`Quantity` `operand` into `x` (Proposal 0003
    §2.4): its magnitude scales the data, its unit multiplies (`op="mul"`) or
    divides (`op="div"`) `x`'s data unit. A bare `Unit` has magnitude 1, so the
    data is untouched -- but through a fresh view, never `x` itself, so
    `_carry` cannot annotate the original in place.
    """
    magnitude, unit = _units.split_quantity(operand)
    if magnitude == 1.0:
        scaled = x.as_subclass(type(x))
    else:
        scaled = Tensor.mul(x, magnitude)
    current = _unit_of(x)
    combined = (
        _units.mul(current, unit) if op == "mul" else _units.div(current, unit)
    )
    return _carry(x, scaled, _data_units=combined)
