"""Tests for fiery.xtensor."""

import enum
import math

import pytest
import torch

from fiery.xtensor import (
    XTensor,
    as_xtensor,
    is_xtensor,
    set_options,
    xmatrix,
    xvector,
)
from fiery.xtensor._compat import torch_func as _torch_func
from fiery.xtensor._selection import _slice_labels
from fiery.xtensor._tensors import Coordinate

# Ops added in torch 1.8; the package registers them only when present, so on
# an older torch (the 1.7 floor) the corresponding tests are skipped.
_HAS_SWAPAXES = hasattr(torch, "swapaxes")  # swapaxes / swapdims
_HAS_MOVEAXIS = hasattr(torch, "moveaxis")
_HAS_BROADCAST_TO = hasattr(torch, "broadcast_to")

# torch 1.7's index_select rejects an int32 index outright ("Expected dtype
# int64 for index"); later torch accepts it (matching the IntTensor-or-
# LongTensor contract torch documents). Probed at runtime rather than gated
# on a version number, since it's the actual op behavior under test.
try:
    torch.zeros(1).index_select(0, torch.zeros(1, dtype=torch.int32))
    _INDEX_SELECT_ACCEPTS_INT32 = True
except RuntimeError:
    _INDEX_SELECT_ACCEPTS_INT32 = False

# ----------------------------------------------------------------------
# dispatch machinery: method form vs. functional form (issue #160)
# ----------------------------------------------------------------------


def test_method_form_runs_the_override_body_exactly_once(monkeypatch):
    # `x.sum(...)` used to run `_reduce_names` twice: once directly (Python
    # attribute lookup finds the registered method without ever going
    # through `__torch_function__`), then again when that call's own
    # `base(input, ...)` re-entered `__torch_function__` and found the same
    # override a second time. `torch.sum(x, ...)` only ever ran it once.
    import fiery.xtensor._reduce as _reduce

    calls = []
    orig = _reduce._reduce_names

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(_reduce, "_reduce_names", counting)

    x = XTensor(torch.arange(6.0).reshape(2, 3), names=("a", "b"))

    calls.clear()
    method_result = x.sum(dim="a")
    assert len(calls) == 1

    calls.clear()
    functional_result = torch.sum(x, 0)
    assert len(calls) == 1

    assert method_result.names == functional_result.names == ("b",)
    assert method_result.tolist() == functional_result.tolist()


def test_pointwise_method_functional_and_operator_forms_run_once(monkeypatch):
    import fiery.xtensor._pointwise as _pointwise

    calls = []
    orig = _pointwise._binary

    def counting(*a, **k):
        calls.append(1)
        return orig(*a, **k)

    monkeypatch.setattr(_pointwise, "_binary", counting)

    x = XTensor(torch.ones(2, 3), names=("a", "b"))
    y = XTensor(torch.ones(2, 3), names=("a", "b"))

    calls.clear()
    x.add(y)
    assert len(calls) == 1

    calls.clear()
    torch.add(x, y)
    assert len(calls) == 1

    calls.clear()
    x + y
    assert len(calls) == 1


def test_method_form_pointwise_op_on_partially_overlapping_names():
    # the double-execution bug (#160) had a user-visible symptom beyond
    # wasted work: re-running `_binary` on its own already name-aligned
    # output (rather than the original operands) could feed it a `None`
    # sitting after a named axis on the second pass, even though the
    # operands' own names never had that shape -- so every method-form
    # pointwise op on two operands with only partially-overlapping name
    # sets (needing the union-broadcast path in `_align_by_name`) used to
    # raise, while the functional and operator forms of the exact same op
    # (which only ever ran `_binary` once) both succeeded.
    a = XTensor(torch.ones(2, 3), names=("a", "b"))
    b = XTensor(torch.ones(3, 4), names=("b", "c"))
    expected_names = ("a", "b", "c")
    expected_shape = (2, 3, 4)

    functional = torch.add(a, b)
    operator_form = a + b
    method = a.add(b)  # used to raise ValueError before the fix

    for out in (functional, operator_form, method):
        assert out.names == expected_names
        assert out.shape == expected_shape
    assert torch.equal(method, functional)


# ----------------------------------------------------------------------
# dimensions (names)
# ----------------------------------------------------------------------


def test_named_tensor_getitem_with_new_axis_keeps_names():
    x = XTensor(torch.arange(6).reshape(2, 3), names=("row", "col"))
    y = x[:, None, 1:]
    assert isinstance(y, XTensor)
    assert y.shape == (2, 1, 2)
    assert y.names == ("row", None, "col")


def test_named_tensor_T_reverses_axis_order_and_names():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "height", "width"),
    )
    y = x.T
    assert y.shape == (4, 3, 2)
    assert y.names == ("width", "height", "batch")


def test_unsqueeze_and_squeeze_round_trip_axis_names():
    x = XTensor(torch.arange(6).reshape(2, 3), names=("row", "col"))
    y = x.unsqueeze(1)
    z = y.squeeze(1)
    assert y.names == ("row", None, "col")
    assert z.names == ("row", "col")


def test_squeeze_without_dim_removes_singleton_axis_names():
    x = XTensor(torch.ones(1, 3, 1), names=("left", "mid", "right"))
    y = x.squeeze()
    assert y.shape == (3,)
    assert y.names == ("mid",)


def test_view_keeps_matching_leading_name_and_marks_reshaped_axes_unnamed():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "height", "width"),
    )
    assert x.view(2, 12).names == ("batch", None)
    assert x.view(2, -1).names == ("batch", None)


def test_view_preserves_trailing_unchanged_axis_name():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "height", "width"),
    )
    y = x.view(6, 4)
    assert y.shape == (6, 4)
    assert y.names == (None, "width")


# ----------------------------------------------------------------------
# coordinates (labels keyed by dim name)
# ----------------------------------------------------------------------


def _labelled():
    return XTensor(
        torch.arange(12).reshape(3, 4),
        names=("row", "col"),
        coords={"col": ("w", "x", "y", "z")},
    )


def test_coords_are_keyed_by_dim_name():
    x = _labelled()
    assert x.coords == {"col": ("w", "x", "y", "z")}


def test_coords_constructor_requires_a_named_axis():
    # a coord keyed by a non-axis name is read as a non-dimension coordinate
    # (Proposal 0005), which must be a (dim, values) tuple -- `()` is not one
    with pytest.raises(ValueError, match="not an axis"):
        XTensor(torch.zeros(2, 3), names=("row", None), coords={"col": ()})


# -- non-dimension coordinates (Proposal 0005, first slice) -------------------


def test_non_dimension_coordinate_rides_along_a_dim():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={
            "t": ["a", "b", "c", "d"],  # dimension coord (the index)
            "season": ("t", ["w", "w", "sp", "sp"]),  # non-dim coord along t
        },
    )
    assert sorted(x.coords) == ["season", "t"]
    assert x.coords["season"] == ("w", "w", "sp", "sp")
    assert x.sel(t="b").item() == 1.0  # the index is selectable
    with pytest.raises(ValueError, match="not an index coordinate"):
        x.sel(season="sp")  # a non-dimension coordinate is not an index


def test_non_dimension_coordinate_propagates_and_drops():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"season": ("t", ["w", "w", "sp", "sp"])},
    )
    assert x.rename(t="time").coords["season"] == ("w", "w", "sp", "sp")
    assert "season" not in x.sum(dim="t").coords  # dim removed -> dropped
    assert (
        "season" not in x[:2].coords
    )  # dim resized -> dropped (conservative)


def test_non_dimension_coordinate_survives_slicing_an_unrelated_axis():
    x = XTensor(
        torch.arange(12.0).reshape(3, 4),
        names=("t", "u"),
        coords={"season": ("t", ["w", "w", "sp"])},
    )
    out = x[:, :2]  # "u" is resized, "t" (and its rider) is untouched
    assert out.coords["season"] == ("w", "w", "sp")


def test_non_dimension_numeric_coordinate():
    x = XTensor(
        torch.arange(3.0),
        names=("i",),
        coords={
            "wl": (
                "i",
                XTensor(torch.tensor([400.0, 500.0, 600.0]), units="nm"),
            )
        },
    )
    assert x.coords["wl"]["value"].tolist() == [400.0, 500.0, 600.0]
    assert x.coords["wl"]["value"].units == "nm"


def test_non_dimension_compact_coordinate_is_rejected():
    # a compact (spacing/origin) non-dimension coordinate isn't re-sliced
    # when its dim is (no slice-tracking yet) and would silently rebind to
    # the wrong affine after a resize -- rejected rather than allowed to
    # silently misbehave; an explicit tensor of values works (see above).
    with pytest.raises(NotImplementedError, match="isn't supported yet"):
        XTensor(
            torch.arange(4.0),
            names=("i",),
            coords={"wl": ("i", {"spacing": 10.0, "origin": 400.0})},
        )


def test_affine_coordinate_materialises_an_nd_grid():
    # Proposal 0005 step 3: a non-dimension coordinate spanning several dims,
    # compact spacing/origin generalised to a vector -- one component per dim.
    x = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={
            "lat": (
                ["y", "x"],
                {"spacing": ([1.0, 0.5], "deg"), "origin": 10.0},
            ),
            "lon": (["y", "x"], {"spacing": ([0.0, 2.0], "deg")}),
        },
    )
    lat = x.coords["lat"]["value"].as_subclass(torch.Tensor)
    lon = x.coords["lon"]["value"].as_subclass(torch.Tensor)
    expected_lat = (
        10.0 + torch.arange(3.0).view(3, 1) + 0.5 * torch.arange(4.0)
    )
    expected_lon = 2.0 * torch.arange(4.0).expand(3, 4)
    assert torch.allclose(lat, expected_lat)
    assert torch.allclose(lon, expected_lon)


def test_affine_coordinate_alone_is_an_under_determined_joint_query():
    # a lone affine coordinate can't be a "not an index" error any more --
    # #82 phase 1 makes a query over it a deliberate joint-affine feature,
    # just one this particular call is under-determined for (needs a value
    # for every coordinate spanning the same dims, not just one).
    x = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={"lat": (["y", "x"], {"spacing": ([1.0, 0.5], "deg")})},
    )
    with pytest.raises(ValueError, match="needs exactly 2 coordinate"):
        x.sel(lat=1.0)


def test_affine_coordinate_spacing_must_match_dims():
    with pytest.raises(ValueError, match="one component per dim"):
        XTensor(
            torch.zeros(3, 4),
            names=["y", "x"],
            coords={"lat": (["y", "x"], {"spacing": ([1.0], "deg")})},
        )


def test_affine_coordinate_bare_tuple_spacing_is_a_vector_not_value_unit():
    # a bare `(v0, v1)` is unambiguous (issue #93): its second element isn't
    # unit-like (not a string/None/backend Unit), so it's a 2-component
    # vector, not a `(value, unit)` pair.
    x = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={"lat": (["y", "x"], {"spacing": (1.0, 0.5)})},
    )
    spacing = x.coords["lat"]["spacing"]
    assert spacing["value"].tolist() == [1.0, 0.5]
    assert spacing["unit"] == ""


def test_affine_coordinate_tuple_spacing_with_a_unit_is_still_value_unit():
    # a 2-tuple whose second element *is* unit-like still parses as
    # `(value, unit)`, not a vector -- only a single component here, so it
    # must be rejected the same as any other wrong-length spacing.
    with pytest.raises(ValueError, match="one component per dim"):
        XTensor(
            torch.zeros(3, 4),
            names=["y", "x"],
            coords={"lat": (["y", "x"], {"spacing": (1.0, "mm")})},
        )


def test_coordinate_origin_must_be_a_scalar():
    with pytest.raises(ValueError, match="origin must be a scalar"):
        XTensor(
            torch.zeros(4),
            names=["t"],
            coords={"t": {"spacing": 1.0, "origin": torch.tensor([1.0, 2.0])}},
        )
    with pytest.raises(ValueError, match="origin must be a scalar"):
        XTensor(
            torch.zeros(3, 4),
            names=["y", "x"],
            coords={
                "lat": (
                    ["y", "x"],
                    {"spacing": ([1.0, 0.5], "deg"), "origin": [1.0, 2.0]},
                )
            },
        )


def test_compact_coordinate_origin_only_defaults_spacing_to_one():
    # an origin-only spec used to crash downstream with a bare
    # `KeyError: 'spacing'` (#95) -- symmetric to how an omitted `origin`
    # already defaults to 0 in `spacing`'s unit, an omitted `spacing`
    # defaults to 1 in `origin`'s unit.
    x = XTensor(torch.zeros(4), names=["t"], coords={"t": {"origin": 5.0}})
    assert x.coords["t"]["value"].tolist() == [5.0, 6.0, 7.0, 8.0]


def test_compact_coordinate_origin_only_default_spacing_shares_its_unit():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.zeros(4), names=["t"], coords={"t": {"origin": (5.0, "mm")}}
        )
        values = x.coords["t"]["value"]
        assert values.units == "millimeter"
        assert values.as_subclass(torch.Tensor).tolist() == [
            5.0,
            6.0,
            7.0,
            8.0,
        ]


def test_explicit_coordinate_must_be_1d():
    # a 2-D (or higher) coordinate tensor used to be silently accepted, only
    # for `.sel` to do a flattened `argmin` and return a bogus position
    # (#97); rejected clearly at construction instead.
    with pytest.raises(ValueError, match="must be 1-D"):
        XTensor(
            torch.arange(6.0).reshape(3, 2),
            names=["t", "u"],
            coords={"t": torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])},
        )


def test_bare_numeric_tuple_coordinate_is_auto_promoted():
    # a bare tuple/list of plain numbers used to be an uncomparable "label"
    # -- .sel could never match it, even the exact value (#107). It's now
    # auto-promoted through the same explicit-coordinate path a tensor spec
    # already takes.
    x = XTensor(
        torch.arange(6.0),
        names=("t",),
        coords={"t": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)},
    )
    assert isinstance(x.coords["t"], Coordinate)
    assert x.sel(t=1.0).item() == 2.0
    assert x.sel(t=1.7, mode="round").item() == 3.0
    # an int-only sequence keeps its int dtype (no gratuitous float upcast)
    y = XTensor(torch.arange(4.0), names=("t",), coords={"t": (0, 1, 2, 3)})
    assert (
        y.coords["t"]["value"].as_subclass(torch.Tensor).dtype == torch.int64
    )


def test_mixed_numeric_and_non_numeric_coordinate_raises():
    with pytest.raises(ValueError, match="mixes numeric values"):
        XTensor(torch.arange(3.0), names=["t"], coords={"t": (0, "a", 2)})


def test_intenum_coordinate_stays_a_label_not_a_numeric_coordinate():
    # an IntEnum member *is* an actual int (Python's own
    # class IntEnum(int, Enum)) -- it must not be swept into the numeric
    # auto-promotion, since that would discard the one thing that made
    # someone reach for an enum: a name (#107).
    season = enum.IntEnum("Season", ["WINTER", "SPRING", "SUMMER", "FALL"])
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={
            "t": (season.WINTER, season.SPRING, season.SUMMER, season.FALL)
        },
    )
    assert not isinstance(x.coords["t"], Coordinate)
    # both calling conventions resolve to the same position
    assert x.sel(t=season.SPRING).item() == 1.0
    assert x.sel(t="SPRING").item() == 1.0


def test_plain_enum_coordinate_is_selectable_by_member_or_name():
    Color = enum.Enum("Color", ["RED", "GREEN", "BLUE"])
    x = XTensor(
        torch.arange(3.0),
        names=("c",),
        coords={"c": (Color.RED, Color.GREEN, Color.BLUE)},
    )
    assert x.sel(c=Color.GREEN).item() == 1.0
    assert x.sel(c="GREEN").item() == 1.0


def test_bool_coordinate_is_selectable_by_value():
    # bool is technically an int subclass too, but a fixed two-value
    # category (like an Enum), not a position -- stays a label, and (unlike
    # before #107) is now actually selectable.
    x = XTensor(torch.arange(2.0), names=("t",), coords={"t": (True, False)})
    assert not isinstance(x.coords["t"], Coordinate)
    assert x.sel(t=True).item() == 0.0
    assert x.sel(t=False).item() == 1.0


def test_auto_promoted_numeric_coordinate_still_gets_the_length_check():
    # the #95/#97 "N labels for size M" validation must not be bypassed
    # just because the sequence happens to be all-numeric.
    with pytest.raises(ValueError, match="3 values.*size 6|6.*3 values"):
        XTensor(torch.arange(6.0), names=("t",), coords={"t": (0.0, 0.5, 1.0)})


def test_str_mixin_enum_coordinate_resolves_by_name_not_value():
    # a `class X(str, Enum)` member IS a str instance -- `_label_name` must
    # check `enum.Enum` before `str`, or this silently matches on the
    # member's *value* instead of its name (the same bug #107 fixed for
    # IntEnum, but on the str side).
    class Color(str, enum.Enum):
        RED = "r"
        BLUE = "b"

    x = XTensor(
        torch.arange(2.0), names=("c",), coords={"c": (Color.RED, Color.BLUE)}
    )
    assert x.sel(c="RED").item() == 0.0
    assert x.sel(c=Color.BLUE).item() == 1.0
    with pytest.raises(ValueError, match="no label"):
        x.sel(c="r")  # the *value*, not the name -- must not match


def test_composite_intflag_coordinate_is_selectable_by_member():
    # a composite Flag/IntFlag value (e.g. RED | BLUE) can have no single
    # matching member name (`.name` is `None` on Python <= 3.10) -- it must
    # still be selectable by passing the member back, even without a string
    # spelling for it.
    class Color(enum.IntFlag):
        RED = 1
        BLUE = 2

    composite = Color.RED | Color.BLUE
    x = XTensor(
        torch.arange(2.0),
        names=("c",),
        coords={"c": (Color.RED, composite)},
    )
    assert x.sel(c=composite).item() == 1.0
    assert x.sel(c=Color.RED).item() == 0.0


def test_non_dimension_numeric_coordinate_also_auto_promotes():
    # #107's bug was reachable through a non-dimension coordinate too (via
    # swap_dims promoting it to be the index) -- the auto-promotion must
    # apply there as well, not just to a dim's own coordinate.
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={
            "t": {"spacing": 1.0, "origin": 0.0},
            "sec": ("t", (0.0, 0.5, 1.0, 1.5)),
        },
    )
    assert isinstance(x.coords["sec"], Coordinate)
    y = x.swap_dims({"t": "sec"})
    assert y.sel(sec=1.0).item() == 2.0


def test_coordinate_origin_unit_defaults_to_spacings_when_unspecified():
    # a bare `origin` number (no unit given) previously silently defaulted
    # to a *different*, conflicting unit than `spacing`'s -- it should
    # simply inherit spacing's unit instead.
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.zeros(4),
            names=("t",),
            coords={"t": {"spacing": (1.0, "mm"), "origin": 5.0}},
        )
        values = x.coords["t"]["value"]
        assert values.units == "millimeter"
        assert values.as_subclass(torch.Tensor).tolist() == [
            5.0,
            6.0,
            7.0,
            8.0,
        ]


def test_coordinate_origin_converts_into_spacings_unit_when_compatible():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.zeros(4),
            names=("t",),
            coords={"t": {"spacing": (0.5, "mm"), "origin": (1.0, "cm")}},
        )
        values = x.coords["t"]["value"]
        assert values.units == "millimeter"
        assert values.as_subclass(torch.Tensor).tolist() == [
            10.0,
            10.5,
            11.0,
            11.5,
        ]
        # the same reconciliation applies to a multi-dim affine coordinate
        y = XTensor(
            torch.zeros(3, 4),
            names=("y", "x"),
            coords={
                "lat": (
                    ("y", "x"),
                    {"spacing": ([1.0, 0.5], "mm"), "origin": (1.0, "cm")},
                )
            },
        )
        expected = (
            10.0 + torch.arange(3.0).view(3, 1) + 0.5 * torch.arange(4.0)
        )
        assert torch.allclose(
            y.coords["lat"]["value"].as_subclass(torch.Tensor), expected
        )


def test_coordinate_origin_incompatible_unit_raises():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        with pytest.raises(ValueError, match="not compatible"):
            XTensor(
                torch.zeros(4),
                names=("t",),
                coords={"t": {"spacing": (0.5, "mm"), "origin": (1.0, "s")}},
            )


def test_coordinate_to_unit_carries_over_the_axis_size_binding():
    # `.to(unit)` on an already-bound coordinate (as returned by `.coords`)
    # must still materialise -- it previously dropped the binding and
    # raised `AttributeError` on the next `["value"]` access.
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.zeros(4),
            names=("t",),
            coords={"t": {"spacing": (0.5, "mm")}},
        )
        converted = x.coords["t"].to("cm")
        assert converted["value"].units == "centimeter"
        assert converted["value"].as_subclass(
            torch.Tensor
        ).tolist() == pytest.approx([0.0, 0.05, 0.1, 0.15])
        # same for the multi-dim affine `_bound_axes` binding
        y = XTensor(
            torch.zeros(3, 4),
            names=("y", "x"),
            coords={"lat": (("y", "x"), {"spacing": ([10.0, 5.0], "mm")})},
        )
        converted_lat = y.coords["lat"].to("cm")
        assert converted_lat["value"].units == "centimeter"
        assert torch.allclose(
            converted_lat["value"].as_subclass(torch.Tensor),
            torch.tensor(
                [
                    [0.0, 0.5, 1.0, 1.5],
                    [1.0, 1.5, 2.0, 2.5],
                    [2.0, 2.5, 3.0, 3.5],
                ]
            ),
        )


def test_multi_dim_explicit_coordinate_is_curvilinear():
    # a general curvilinear array of explicit values over several dims is
    # supported (issue #82 phase 2), not just the compact affine form.
    lat = torch.arange(12.0).reshape(3, 4)
    t = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat)},
    )
    values = t.coords["lat"]["value"].as_subclass(torch.Tensor)
    assert torch.equal(values, lat)


def test_multi_dim_explicit_coordinate_rejects_bad_shape():
    with pytest.raises(ValueError, match="spans dims"):
        XTensor(
            torch.zeros(3, 4),
            names=["y", "x"],
            coords={"lat": (["y", "x"], torch.zeros(3, 5))},
        )


def test_multi_dim_explicit_coordinate_rejects_non_tensor():
    with pytest.raises(ValueError, match="explicit tensor"):
        XTensor(
            torch.zeros(3, 4),
            names=["y", "x"],
            coords={"lat": (["y", "x"], 5.0)},
        )


def test_multi_dim_explicit_coordinate_drops_on_slice():
    lat = torch.arange(12.0).reshape(3, 4)
    t = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat)},
    )
    sliced = t[:, :2]
    assert "lat" not in sliced.coords


def test_multi_dim_explicit_coordinate_follows_permute():
    lat = torch.arange(12.0).reshape(3, 4)
    t = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat)},
    )
    permuted = t.permute(1, 0)
    expected = lat.permute(1, 0)
    assert torch.equal(
        permuted.coords["lat"]["value"].as_subclass(torch.Tensor), expected
    )


def _curvilinear_demo():
    y = torch.linspace(0, 1, 4).unsqueeze(1)
    x = torch.linspace(0, 1, 5).unsqueeze(0)
    lat = (10 + 5 * y + 0.1 * x**2).expand(4, 5).contiguous()
    lon = (100 + 2 * x + 0.05 * y**2).expand(4, 5).contiguous()
    data = torch.arange(20.0).reshape(4, 5)
    t = XTensor(
        data,
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    return t, lat, lon, data


def test_curvilinear_sel_nearest_neighbor():
    t, lat, lon, data = _curvilinear_demo()
    target_lat = float(lat[2, 3])
    target_lon = float(lon[2, 3])
    picked = t.sel(lat=target_lat, lon=target_lon)
    assert picked.item() == data[2, 3].item()


def test_curvilinear_sel_nearest_snaps_off_grid_value():
    t, lat, lon, data = _curvilinear_demo()
    target_lat = float(lat[2, 3])
    target_lon = float(lon[2, 3])
    picked = t.sel(
        lat=target_lat + 1e-3, lon=target_lon + 1e-3, method="nearest"
    )
    assert picked.item() == data[2, 3].item()


def test_curvilinear_sel_respects_permute():
    t, lat, lon, data = _curvilinear_demo()
    t = t.permute(1, 0)
    target_lat = float(lat[2, 3])
    target_lon = float(lon[2, 3])
    picked = t.sel(lat=target_lat, lon=target_lon)
    assert picked.item() == data[2, 3].item()


def test_curvilinear_sel_tolerance_raises_when_exceeded():
    t, lat, lon, data = _curvilinear_demo()
    target_lat = float(lat[2, 3])
    target_lon = float(lon[2, 3])
    with pytest.raises(ValueError, match="over tolerance"):
        t.sel(
            lat=target_lat + 5.0,
            lon=target_lon,
            method="nearest",
            tolerance=0.01,
        )


def test_curvilinear_sel_unbalanced_group_raises():
    lat = torch.arange(12.0).reshape(3, 4)
    lon = torch.arange(12.0).reshape(3, 4)
    t = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    with pytest.raises(ValueError, match="needs exactly 2"):
        t.sel(lat=5.0)


def test_curvilinear_sel_unsupported_mode_raises():
    lat = torch.arange(12.0).reshape(3, 4)
    lon = torch.arange(12.0).reshape(3, 4)
    t = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    with pytest.raises(NotImplementedError, match="curvilinear"):
        t.sel(lat=5.0, lon=5.0, mode="floor")


def test_curvilinear_sel_size_guard(monkeypatch):
    import fiery.xtensor._tensors as _tensors_mod

    monkeypatch.setattr(_tensors_mod, "_CURVILINEAR_SEL_MAX_BYTES", 10)
    lat = torch.zeros(4, 5)
    lon = torch.zeros(4, 5)
    t = XTensor(
        torch.zeros(4, 5),
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    with pytest.raises(ValueError, match="too large"):
        t.sel(lat=0.0, lon=0.0)


def test_curvilinear_sel_correct_on_large_float32_grid():
    # regression test (review finding #1): torch.cdist's default compute
    # mode switches to a matrix-multiply identity above 25 points, which
    # catastrophically cancels in float32 for realistic coordinate
    # magnitudes -- silently picking the wrong nearest neighbor. A 6x6 grid
    # crosses that threshold; realistic lat/lon magnitudes (~52 degrees)
    # trigger the cancellation.
    y = torch.linspace(0, 1, 6).unsqueeze(1)
    x = torch.linspace(0, 1, 6).unsqueeze(0)
    lat = (52.0 + 0.01 * y + 0.0001 * x**2).expand(6, 6).contiguous()
    lon = (4.0 + 0.01 * x + 0.0001 * y**2).expand(6, 6).contiguous()
    data = torch.arange(36.0).reshape(6, 6)
    t = XTensor(
        data,
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    for row in range(6):
        for col in range(6):
            target_lat = float(lat[row, col])
            target_lon = float(lon[row, col])
            picked = t.sel(lat=target_lat, lon=target_lon)
            assert picked.item() == data[row, col].item()


def test_curvilinear_coordinate_survives_to_unit():
    # regression test (review finding #2): Coordinate.to() must carry over
    # the curvilinear axis-order binding, or a unit conversion after a
    # permute silently reverts to the construction-order (transposed) shape.
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        lat_raw = torch.arange(12.0).reshape(3, 4)
        lat = as_xtensor(lat_raw, units="m")
        t = XTensor(
            torch.zeros(3, 4),
            names=["y", "x"],
            coords={"lat": (["y", "x"], lat)},
        ).permute(1, 0)
        bound = t.coords["lat"]
        expected = lat_raw.permute(1, 0) * 1000
        converted = bound.to("mm")["value"].as_subclass(torch.Tensor)
        assert torch.equal(converted, expected)


def test_curvilinear_sel_after_slice_raises_has_no_coordinates():
    # regression test (review finding #3): a curvilinear coordinate dropped
    # by a slice on one of its spanned dims must raise the usual "has no
    # coordinates" error, not a bare KeyError.
    lat = torch.arange(12.0).reshape(3, 4)
    lon = torch.arange(12.0).reshape(3, 4)
    t = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    sliced = t[:, :2]
    with pytest.raises(ValueError, match="has no coordinates"):
        sliced.sel(lat=1.0, lon=1.0)


def test_curvilinear_sel_ignores_nan_grid_point():
    # regression test (review finding #4): a NaN grid point (a masked/fill
    # swath cell) must not win nearest-neighbor argmin by NaN-propagation,
    # and must not defeat the tolerance check either.
    t, lat, lon, data = _curvilinear_demo()
    target_lat = float(lat[2, 3])
    target_lon = float(lon[2, 3])
    lat = lat.clone()
    lon = lon.clone()
    lat[2, 3] = float("nan")
    lon[2, 3] = float("nan")
    t = XTensor(
        data,
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    # the NaN cell is exactly the query target, but it must be excluded --
    # some *other* (non-NaN) cell, farther away, is chosen instead.
    picked = t.sel(lat=target_lat, lon=target_lon, method="nearest")
    assert picked.item() != data[2, 3].item()
    with pytest.raises(ValueError, match="over tolerance"):
        t.sel(
            lat=target_lat,
            lon=target_lon,
            method="nearest",
            tolerance=1e-9,
        )


def test_curvilinear_sel_works_on_integer_dtype_grid():
    # regression test (review finding #5): an integer-dtype curvilinear
    # grid must not raise a raw torch dtype error.
    lat = torch.arange(12).reshape(3, 4)
    lon = torch.arange(12).reshape(3, 4)
    data = torch.arange(12.0).reshape(3, 4)
    t = XTensor(
        data,
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    picked = t.sel(lat=5, lon=5)
    assert picked.item() == data[1, 1].item()


def test_curvilinear_sel_rejects_multi_point_query():
    # regression test (review finding #6): the deliberately-unsupported
    # vectorized/slice query forms must raise a clear error pointing at the
    # single-point-only restriction, not an internal TypeError/ValueError.
    lat = torch.arange(12.0).reshape(3, 4)
    lon = torch.arange(12.0).reshape(3, 4)
    t = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    with pytest.raises(TypeError, match="single point"):
        t.sel(lat=[1.0, 2.0], lon=[3.0, 4.0])
    with pytest.raises(TypeError, match="single point"):
        t.sel(lat=slice(1.0, 2.0), lon=3.0)


def test_curvilinear_sel_empty_grid_raises():
    lat = torch.zeros(0, 4)
    lon = torch.zeros(0, 4)
    t = XTensor(
        torch.zeros(0, 4),
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    with pytest.raises(ValueError, match="empty grid"):
        t.sel(lat=1.0, lon=1.0)


# ----------------------------------------------------------------------
# general curvilinear .interp -- Newton inversion (issue #82)
# ----------------------------------------------------------------------


def _bilinear_2d(grid: torch.Tensor, fi: float, fj: float) -> float:
    """Hand-rolled bilinear sample of a 2-D tensor, used as ground truth."""
    h, w = grid.shape
    i0, j0 = int(fi), int(fj)
    i1, j1 = min(i0 + 1, h - 1), min(j0 + 1, w - 1)
    di, dj = fi - i0, fj - j0
    return (
        grid[i0, j0].item() * (1 - di) * (1 - dj)
        + grid[i0, j1].item() * (1 - di) * dj
        + grid[i1, j0].item() * di * (1 - dj)
        + grid[i1, j1].item() * di * dj
    )


def test_curvilinear_interp_exact_grid_point_matches_data():
    pytest.importorskip("fiery.interpol")
    t, lat, lon, data = _curvilinear_demo()
    out = t.interp(lat=float(lat[2, 3]), lon=float(lon[2, 3]), method="linear")
    assert out.item() == data[2, 3].item()


def test_curvilinear_interp_nearest_matches_sel():
    t, lat, lon, data = _curvilinear_demo()
    out = t.interp(
        lat=float(lat[2, 3]) + 1e-3,
        lon=float(lon[2, 3]) + 1e-3,
        method="nearest",
    )
    assert out.item() == data[2, 3].item()


def test_curvilinear_interp_matches_a_hand_computed_bilinear_reference():
    # ground truth: query the *exact* bilinear interpolate of the stored
    # (genuinely nonlinear) lat/lon grids at a fractional index, so Newton
    # must recover that same fractional index back out, and the pulled
    # data must match a hand-rolled bilinear reference over the data.
    pytest.importorskip("fiery.interpol")
    import random

    t, lat, lon, data = _curvilinear_demo()
    rng = random.Random(0)
    for _ in range(20):
        fi = rng.uniform(0, 2.999)
        fj = rng.uniform(0, 3.999)
        target_lat = _bilinear_2d(lat, fi, fj)
        target_lon = _bilinear_2d(lon, fi, fj)
        got = t.interp(lat=target_lat, lon=target_lon, method="linear").item()
        expected = _bilinear_2d(data, fi, fj)
        # investigated (review of #166, finding #9): the ~1e-4 floor here
        # is not float32 fixture noise from `_curvilinear_demo`'s data --
        # rerunning with float64 data/coordinates measures the same ~7e-6
        # worst case, so the floor traces to the optional
        # `fiery.interpol.grid_pull` backend itself (out of this module's
        # control), not to anything the curvilinear solve does. Left as-is
        # rather than tightened to a value that would flake on backend
        # precision it doesn't own.
        assert abs(got - expected) < 1e-4, (fi, fj, got, expected)


def test_curvilinear_interp_matches_the_closed_form_affine_path():
    # cross-check: an affine map stored as an *explicit* curvilinear array
    # (no formula available to the general solver) must still agree with
    # the already-tested closed-form affine `.interp` path, since a
    # bilinear pull over an affine map has no approximation error.
    pytest.importorskip("fiery.interpol")
    import random

    h, w = 5, 6
    field = torch.rand(h, w, dtype=torch.float64)
    affine = XTensor(
        field,
        names=("y", "x"),
        coords={
            "lat": (
                ("y", "x"),
                {"spacing": ([1.3, 0.4], ""), "origin": (10.0, "")},
            ),
            "lon": (
                ("y", "x"),
                {"spacing": ([-0.2, 2.1], ""), "origin": (5.0, "")},
            ),
        },
    )
    yv = torch.arange(h, dtype=torch.float64).unsqueeze(1)
    xv = torch.arange(w, dtype=torch.float64).unsqueeze(0)
    lat_arr = (10.0 + 1.3 * yv + 0.4 * xv).expand(h, w).contiguous()
    lon_arr = (5.0 - 0.2 * yv + 2.1 * xv).expand(h, w).contiguous()
    curv = XTensor(
        field,
        names=("y", "x"),
        coords={"lat": (("y", "x"), lat_arr), "lon": (("y", "x"), lon_arr)},
    )
    rng = random.Random(1)
    for _ in range(20):
        fi = rng.uniform(0, h - 1)
        fj = rng.uniform(0, w - 1)
        target_lat = 10.0 + 1.3 * fi + 0.4 * fj
        target_lon = 5.0 - 0.2 * fi + 2.1 * fj
        expected = affine.interp(
            lat=target_lat, lon=target_lon, method="linear"
        ).item()
        got = curv.interp(
            lat=target_lat, lon=target_lon, method="linear"
        ).item()
        # tightened from 1e-5 (review of #166, finding #9): measured worst
        # discrepancy across hundreds of points is ~1.3e-7, so 1e-5 was
        # ~100x looser than reality and wouldn't have caught a real
        # regression; 1e-6 keeps comfortable margin without being loose.
        assert abs(got - expected) < 1e-6, (fi, fj, got, expected)


def test_curvilinear_interp_many_query_produces_one_new_named_axis():
    pytest.importorskip("fiery.interpol")
    t, lat, lon, data = _curvilinear_demo()
    out = t.interp(
        lat=[float(lat[0, 0]), float(lat[3, 4])],
        lon=[float(lon[0, 0]), float(lon[3, 4])],
        name="pts",
    )
    assert out.names == ("pts",)
    assert out.shape == (2,)
    assert out.tolist() == [data[0, 0].item(), data[3, 4].item()]


def test_curvilinear_interp_out_of_range_raises():
    pytest.importorskip("fiery.interpol")
    t, lat, lon, data = _curvilinear_demo()
    with pytest.raises(ValueError, match="outside the grid"):
        t.interp(lat=1000.0, lon=1000.0, method="linear")


def test_curvilinear_interp_boundary_seeded_query_succeeds():
    # regression test (review of #166, finding #1): `_bilinear_sample_2d`
    # used to clamp its query point *before* differencing, so the Newton
    # solve's finite-difference Jacobian was silently halved whenever an
    # iterate sat exactly at index 0 or size-1 -- which the nearest-
    # neighbor seed does constantly, since it is always an exact integer
    # index by construction. On this anisotropic 11x11 grid the corrupted
    # Jacobian used to send the solve's very first step off the edge,
    # collapse the (still-clamped) Jacobian to 0 there, and raise
    # "outside the grid's coordinate range" for a target that is squarely
    # interior (true index ~(0.4299, 5.2996)).
    pytest.importorskip("fiery.interpol")
    i = torch.arange(11.0).unsqueeze(1)
    j = torch.arange(11.0).unsqueeze(0)
    lat = (100.0 * (i + 0.3 * j)).expand(11, 11).contiguous()
    lon = (j - 0.2 * i).expand(11, 11).contiguous()
    data = (10 * i + j).expand(11, 11).contiguous()
    t = XTensor(
        data,
        names=("y", "x"),
        coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    )
    out = t.interp(
        lat=201.98156282543255, lon=5.213648134218069, method="linear"
    )
    assert out.item() == pytest.approx(9.598889754567772, abs=1e-4)


def test_curvilinear_interp_singular_check_is_scale_aware():
    # regression test (review of #166, finding #2): `|det(J)|` has units of
    # world^2 / index^2, so comparing it to a bare absolute constant only
    # makes sense for O(1)-magnitude coordinates. The same well-conditioned
    # geometry, stored as fine-resolution (~5 m and ~1 m cell) lat/lon
    # degrees, used to be misdiagnosed as "not locally invertible" purely
    # because of its units -- with no change to the actual geometry.
    pytest.importorskip("fiery.interpol")

    def make(cell_deg):
        h = w = 9
        y = torch.arange(h, dtype=torch.float64).unsqueeze(1)
        x = torch.arange(w, dtype=torch.float64).unsqueeze(0)
        lat = (10.0 + cell_deg * (y + 0.05 * x)).expand(h, w).contiguous()
        lon = (20.0 + cell_deg * (x - 0.05 * y)).expand(h, w).contiguous()
        data = (10 * y + x).expand(h, w).contiguous()
        return (
            XTensor(
                data,
                names=("y", "x"),
                coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
            ),
            lat,
            lon,
        )

    for cell_deg in (4.5e-5, 9e-6):  # ~5 m and ~1 m cells, in degrees
        t, lat, lon = make(cell_deg)
        target_lat = float(lat[4, 4]) + cell_deg * 0.3
        target_lon = float(lon[4, 4]) + cell_deg * 0.2
        out = t.interp(lat=target_lat, lon=target_lon, method="linear")
        assert math.isfinite(out.item())


def test_curvilinear_interp_converges_at_large_coordinate_magnitude():
    # regression test (review of #166, finding #3): the convergence check
    # used to compare the residual to a bare absolute world-unit constant,
    # unreachable in float64 past a coordinate magnitude of a few million
    # (UTM/ECEF-scale). Same nonlinear geometry, scaled up to a
    # metres-in-large-datum magnitude, must still converge.
    pytest.importorskip("fiery.interpol")
    h = w = 9
    y = torch.arange(h, dtype=torch.float64).unsqueeze(1)
    x = torch.arange(w, dtype=torch.float64).unsqueeze(0)
    scale = 1e10
    lat = (scale * (y + 0.3 * torch.sin(x))).expand(h, w).contiguous()
    lon = (scale * (x + 0.3 * torch.cos(y))).expand(h, w).contiguous()
    data = (10 * y + x).expand(h, w).contiguous()
    t = XTensor(
        data,
        names=("y", "x"),
        coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    )
    fi, fj = 3.4, 4.6
    target_lat = _bilinear_2d(lat, fi, fj)
    target_lon = _bilinear_2d(lon, fi, fj)
    got = t.interp(lat=target_lat, lon=target_lon, method="linear").item()
    expected = _bilinear_2d(data, fi, fj)
    assert got == pytest.approx(expected, rel=1e-6)


def test_curvilinear_interp_batched_query_matches_single_point_queries():
    # regression test (review of #166, finding #4): the Newton loop kept
    # singularity-testing *already-converged* points on every subsequent
    # iteration driven by a slower batch-mate. A point that seeds exactly
    # onto a fold's apex converges in one step with a genuinely singular
    # local Jacobian there -- fine on its own, but re-checking it on every
    # later iteration used to make the whole batched call raise, purely
    # because of who else was in the batch.
    pytest.importorskip("fiery.interpol")
    i = torch.arange(9.0).unsqueeze(1)
    j = torch.arange(9.0).unsqueeze(0)
    lat = ((i - 4.0) ** 2 / 8.0).expand(9, 9).contiguous()  # apex at i=4
    lon = j.expand(9, 9).contiguous()
    data = (10 * i + j).expand(9, 9).contiguous()
    t = XTensor(
        data,
        names=("y", "x"),
        coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    )
    out_a = t.interp(lat=[0.0], lon=[3.0], name="p")
    out_b = t.interp(lat=[0.5], lon=[2.37], name="p")
    out_ab = t.interp(lat=[0.0, 0.5], lon=[3.0, 2.37], name="p")
    assert out_ab[0].item() == pytest.approx(out_a[0].item())
    assert out_ab[1].item() == pytest.approx(out_b[0].item())


def test_curvilinear_interp_near_fold_resolves_to_a_valid_preimage():
    # a folded map (lat depends on |y - 2|, so two different y's -- 1.7 and
    # 2.3 -- share the same lat=0.3) is only locally non-invertible *right
    # at* the fold line itself (y == 2 exactly, where the Jacobian is
    # genuinely rank-deficient): away from it, each branch is individually
    # well-conditioned, so the solver must resolve to *one* of the two
    # legitimate preimages -- never silently returning a value that matches
    # neither (review of #166, finding #5: the "raises at a fold" guarantee
    # is local to the fold line, not the whole folded region).
    pytest.importorskip("fiery.interpol")
    h, w = 5, 5
    data = torch.arange(25.0).reshape(h, w)
    y = torch.arange(h, dtype=torch.float64).unsqueeze(1)
    x = torch.arange(w, dtype=torch.float64).unsqueeze(0)
    lat = (y - 2.0).abs().expand(h, w).contiguous()
    lon = x.expand(h, w).contiguous()
    t = XTensor(
        data,
        names=("y", "x"),
        coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    )
    got = t.interp(lat=0.3, lon=2.3, method="linear").item()
    # the two legitimate preimages, y=1.7 or y=2.3 (both x=2.3), give
    # data = 5*y + x -- either is an acceptable, genuinely-valid answer.
    branch_low = 5 * 1.7 + 2.3
    branch_high = 5 * 2.3 + 2.3
    assert got == pytest.approx(branch_low) or got == pytest.approx(
        branch_high
    ), (got, branch_low, branch_high)


def test_curvilinear_interp_rank_deficient_map_raises_singular():
    # a map where one spanned axis (x) contributes nothing to *either*
    # world coordinate is genuinely, everywhere non-invertible (the
    # Jacobian's x-column is exactly zero) -- unlike a fold, this isn't a
    # numerical near-miss, so it must raise regardless of query point.
    pytest.importorskip("fiery.interpol")
    h, w = 5, 5
    data = torch.arange(25.0).reshape(h, w)
    y = torch.arange(h, dtype=torch.float64).unsqueeze(1)
    lat = y.expand(h, w).contiguous()
    lon = (2.0 * y).expand(h, w).contiguous()
    t = XTensor(
        data,
        names=("y", "x"),
        coords={"lat": (("y", "x"), lat), "lon": (("y", "x"), lon)},
    )
    with pytest.raises(ValueError, match="isn't locally invertible"):
        t.interp(lat=1.5, lon=3.0, method="linear")


def test_curvilinear_interp_gradients_flow_through_data_not_query():
    # documented limitation: the Newton solve is detached (an iterative,
    # data-dependent root find has no well-defined gradient), so only the
    # tensor's own *data* values stay differentiable -- same as every other
    # interp path -- while the query itself carries no gradient. Checks
    # both halves of the claim (review of #166, finding #9): the *query*
    # tensors must come back with `.grad is None`, not just be unchecked,
    # and the *data* gradient must match the exact bilinear weights (not
    # just "sums to something positive", which is a tautology -- interp
    # weights always sum to 1 regardless of whether they're correct).
    pytest.importorskip("fiery.interpol")
    t, lat, lon, data = _curvilinear_demo()
    values = data.clone().requires_grad_(True)
    tg = XTensor(
        values,
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    fi, fj = 1.4, 2.6  # an interior, non-exact-node fractional index
    target_lat = _bilinear_2d(lat, fi, fj)
    target_lon = _bilinear_2d(lon, fi, fj)
    query_lat = torch.tensor(target_lat, requires_grad=True)
    query_lon = torch.tensor(target_lon, requires_grad=True)
    out = tg.interp(lat=query_lat, lon=query_lon, method="linear")
    out.backward()

    # query point: no gradient at all (detached Newton solve).
    assert query_lat.grad is None
    assert query_lon.grad is None

    # data: gradient must equal the exact bilinear corner weights, not
    # merely be non-negative/summing to 1.
    assert values.grad is not None
    assert not torch.isnan(values.grad).any()
    i0, j0 = int(fi), int(fj)
    i1, j1 = i0 + 1, j0 + 1
    di, dj = fi - i0, fj - j0
    expected = torch.zeros_like(values.grad)
    expected[i0, j0] = (1 - di) * (1 - dj)
    expected[i0, j1] = (1 - di) * dj
    expected[i1, j0] = di * (1 - dj)
    expected[i1, j1] = di * dj
    torch.testing.assert_close(values.grad, expected, atol=1e-5, rtol=1e-4)
    assert values.grad.sum().item() == pytest.approx(1.0)


def test_curvilinear_interp_unbalanced_group_raises():
    lat = torch.arange(12.0).reshape(3, 4)
    lon = torch.arange(12.0).reshape(3, 4)
    t = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={"lat": (["y", "x"], lat), "lon": (["y", "x"], lon)},
    )
    with pytest.raises(ValueError, match="needs exactly 2"):
        t.interp(lat=5.0)


def test_curvilinear_interp_mismatched_lengths_raises():
    t, lat, lon, data = _curvilinear_demo()
    with pytest.raises(ValueError, match="same length"):
        t.interp(lat=[11.0, 12.0], lon=[24.0, 20.0, 22.0])


def test_curvilinear_interp_multiple_groups_in_one_call_raises():
    field = torch.arange(16.0).reshape(2, 2, 2, 2)
    lat = torch.arange(4.0).reshape(2, 2)
    lon = torch.arange(4.0).reshape(2, 2) + 10.0
    p = torch.arange(4.0).reshape(2, 2) + 100.0
    q = torch.arange(4.0).reshape(2, 2) + 200.0
    t = XTensor(
        field,
        names=("y", "x", "z", "w"),
        coords={
            "lat": (("y", "x"), lat),
            "lon": (("y", "x"), lon),
            "p": (("z", "w"), p),
            "q": (("z", "w"), q),
        },
    )
    with pytest.raises(NotImplementedError, match="more than one"):
        t.interp(lat=0.0, lon=10.0, p=100.0, q=200.0)


def test_curvilinear_interp_beyond_2d_raises():
    field = torch.zeros(2, 2, 2)
    a = torch.arange(8.0).reshape(2, 2, 2)
    b = torch.arange(8.0).reshape(2, 2, 2) + 100.0
    c = torch.arange(8.0).reshape(2, 2, 2) + 200.0
    t = XTensor(
        field,
        names=("y", "x", "z"),
        coords={
            "a": (("y", "x", "z"), a),
            "b": (("y", "x", "z"), b),
            "c": (("y", "x", "z"), c),
        },
    )
    with pytest.raises(NotImplementedError, match="2-D"):
        t.interp(a=1.0, b=101.0, c=201.0)


def test_curvilinear_interp_higher_order_raises():
    pytest.importorskip("fiery.interpol")
    t, lat, lon, data = _curvilinear_demo()
    with pytest.raises(NotImplementedError, match="nearest.*linear"):
        t.interp(
            lat=float(lat[2, 3]), lon=float(lon[2, 3]), method="quadratic"
        )


def test_curvilinear_interp_and_affine_group_conflict_raises():
    # a dim resolved by *both* a joint affine and a joint curvilinear query
    # in the same call is ambiguous -- pick one or the other (mirrors the
    # equivalent `.sel` check).
    pytest.importorskip("fiery.interpol")
    field = torch.arange(24.0).reshape(4, 6)
    lat = torch.arange(24.0).reshape(4, 6)
    lon = torch.arange(24.0).reshape(4, 6) + 100.0
    t = XTensor(
        field,
        names=("y", "x"),
        coords={
            "lat": (("y", "x"), lat),
            "lon": (("y", "x"), lon),
            "p": (
                ("y", "x"),
                {"spacing": ([1.0, 0.0], ""), "origin": (0.0, "")},
            ),
            "q": (
                ("y", "x"),
                {"spacing": ([0.0, 1.0], ""), "origin": (0.0, "")},
            ),
        },
    )
    with pytest.raises(ValueError, match="spanned by both"):
        t.interp(lat=5.0, lon=105.0, p=1.0, q=1.0)


def test_curvilinear_interp_and_affine_group_conflict_raises_reversed_dims():
    # regression test (review of #166, finding #7): the same two axes,
    # spanned in the *other* order by the curvilinear coordinate, must
    # still be caught -- an exact-tuple `dims` comparison let `("y", "x")`
    # (affine) vs. `("x", "y")` (curvilinear) slip through, producing a
    # confusing "no axis named ..." error instead of naming the conflict.
    pytest.importorskip("fiery.interpol")
    field = torch.arange(24.0).reshape(4, 6)
    arr = torch.arange(24.0).reshape(6, 4)
    t = XTensor(
        field,
        names=("y", "x"),
        coords={
            "p": (
                ("y", "x"),
                {"spacing": ([1.0, 0.0], ""), "origin": (0.0, "")},
            ),
            "q": (
                ("y", "x"),
                {"spacing": ([0.0, 1.0], ""), "origin": (0.0, "")},
            ),
            "lat": (("x", "y"), arr),
            "lon": (("x", "y"), arr + 100.0),
        },
    )
    with pytest.raises(ValueError, match="spanned by both"):
        t.interp(p=1.0, q=2.0, lat=11.5, lon=102.5)


def test_curvilinear_interp_and_affine_group_conflict_raises_partial_overlap():
    # regression test (review of #166, finding #7): a curvilinear coordinate
    # sharing only *some* of the affine coordinate's dims (not the exact
    # same tuple) must also be caught -- the exact-tuple comparison missed
    # this too.
    pytest.importorskip("fiery.interpol")
    field = torch.arange(48.0).reshape(4, 6, 2)
    arr = torch.arange(12.0).reshape(6, 2)
    t = XTensor(
        field,
        names=("y", "x", "z"),
        coords={
            "p": (
                ("y", "x"),
                {"spacing": ([1.0, 0.0], ""), "origin": (0.0, "")},
            ),
            "q": (
                ("y", "x"),
                {"spacing": ([0.0, 1.0], ""), "origin": (0.0, "")},
            ),
            "lat": (("x", "z"), arr),
            "lon": (("x", "z"), arr + 100.0),
        },
    )
    with pytest.raises(ValueError, match="spanned by both"):
        t.interp(p=1.0, q=2.0, lat=11.5, lon=102.5)


def test_affine_coordinate_repeated_dim_is_rejected():
    with pytest.raises(ValueError, match="repeats a dim"):
        XTensor(
            torch.zeros(3, 4),
            names=["y", "x"],
            coords={"lat": (["y", "y"], {"spacing": ([1.0, 1.0], "deg")})},
        )


def test_affine_coordinate_basic_slice_reslices_exactly():
    x = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={
            "lat": (
                ["y", "x"],
                {"spacing": ([1.0, 0.5], "deg"), "origin": 10.0},
            )
        },
    )
    full = x.coords["lat"]["value"].as_subclass(torch.Tensor)
    cropped = x[1:3, 1:4]
    assert torch.allclose(
        cropped.coords["lat"]["value"].as_subclass(torch.Tensor),
        full[1:3, 1:4],
    )
    strided = x[::2, ::2]
    assert torch.allclose(
        strided.coords["lat"]["value"].as_subclass(torch.Tensor),
        full[::2, ::2],
    )


def test_affine_coordinate_integer_index_folds_a_dim_out():
    x = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={
            "lat": (
                ["y", "x"],
                {"spacing": ([1.0, 0.5], "deg"), "origin": 10.0},
            )
        },
    )
    full = x.coords["lat"]["value"].as_subclass(torch.Tensor)
    row = x[1, :]
    assert row.names == ("x",)
    assert torch.allclose(
        row.coords["lat"]["value"].as_subclass(torch.Tensor), full[1, :]
    )
    # folding every spanned dim away leaves no axis for it to ride on
    assert "lat" not in x[1, 2].coords


def test_affine_coordinate_collapsed_to_one_dim_can_be_resliced_further():
    # a coordinate collapsed by folding out every dim but one stores a bare
    # scalar `spacing` (the ordinary 1-D compact form), not a length-1
    # vector -- reslicing it again must not try to index into that scalar.
    x = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={
            "lat": (
                ["y", "x"],
                {"spacing": ([1.0, 0.5], "deg"), "origin": 10.0},
            )
        },
    )
    full = x.coords["lat"]["value"].as_subclass(torch.Tensor)
    row = x[1, :]
    resliced = row[1:3]
    assert torch.allclose(
        resliced.coords["lat"]["value"].as_subclass(torch.Tensor),
        full[1, 1:3],
    )


def test_affine_coordinate_partial_fold_and_slice_over_three_dims():
    x = XTensor(
        torch.zeros(2, 3, 4),
        names=["z", "y", "x"],
        coords={
            "vol": (
                ["z", "y", "x"],
                {"spacing": ([10.0, 1.0, 0.5], "mm"), "origin": 0.0},
            )
        },
    )
    full = x.coords["vol"]["value"].as_subclass(torch.Tensor)
    sliced = x[1, 1:3, ::2]
    assert sliced.names == ("y", "x")
    assert torch.allclose(
        sliced.coords["vol"]["value"].as_subclass(torch.Tensor),
        full[1, 1:3, ::2],
    )


def test_affine_coordinate_advanced_indexing_drops_it():
    x = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={"lat": (["y", "x"], {"spacing": ([1.0, 0.5], "deg")})},
    )
    assert "lat" not in x[[0, 2], :].coords
    assert "lat" not in x[:, torch.tensor([0, 1])].coords


def test_affine_coordinate_survives_rename():
    x = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={
            "lat": (
                ["y", "x"],
                {"spacing": ([1.0, 0.5], "deg"), "origin": 10.0},
            )
        },
    )
    full = x.coords["lat"]["value"].as_subclass(torch.Tensor)
    renamed = x.rename(y="row", x="col")
    assert torch.allclose(
        renamed.coords["lat"]["value"].as_subclass(torch.Tensor), full
    )


def test_affine_coordinate_rename_onto_one_of_its_dims_is_rejected():
    # renaming an axis onto a multi-dim coordinate's key would leave an entry
    # whose key *is* a dim but which is not that dim's index, breaking the
    # `dims == (name,)` <-> dimension-coordinate invariant (`.sel` and
    # `__getitem__`'s dimension-coordinate pass would then treat the vector
    # `spacing` as a 1-D one and corrupt it / raise).
    def make():
        return XTensor(
            torch.zeros(4, 4),
            names=["y", "x"],
            coords={
                "lat": (
                    ["y", "x"],
                    {"spacing": ([1.0, 0.5], "deg"), "origin": 10.0},
                )
            },
        )

    with pytest.raises(ValueError, match="multi-dim coordinate's name"):
        make().rename(y="lat")
    with pytest.raises(ValueError, match="multi-dim coordinate's name"):
        make().rename(x="lat")
    # a *single*-dim non-dimension coordinate still becomes that dim's index
    single = XTensor(
        torch.zeros(4, 4),
        names=["y", "x"],
        coords={"lab": ("y", torch.arange(4.0))},
    )
    assert single.rename(y="lab").sel(lab=2.0).names == ("x",)


def test_affine_coordinate_grid_follows_the_tensor_axis_order():
    # `["value"]` is a bare array with no dims of its own, so it must be laid
    # out in the *tensor's* axis order -- otherwise transposing silently
    # misaligns the coordinate with the data (undetectably so when square).
    x = XTensor(
        torch.zeros(4, 4),
        names=["y", "x"],
        coords={
            "lat": (
                ["y", "x"],
                {"spacing": ([1.0, 0.5], "deg"), "origin": 10.0},
            )
        },
    )
    full = x.coords["lat"]["value"].as_subclass(torch.Tensor)
    for moved in (x.permute(1, 0), x.transpose("y", "x"), x.movedim(0, 1)):
        assert moved.names == ("x", "y")
        assert torch.allclose(
            moved.coords["lat"]["value"].as_subclass(torch.Tensor), full.T
        )
    # ... and it keeps following the axes through a later slice
    sliced = x.permute(1, 0)[1:4, 1:3]
    assert torch.allclose(
        sliced.coords["lat"]["value"].as_subclass(torch.Tensor),
        full.T[1:4, 1:3],
    )


def test_affine_coordinate_dims_may_be_given_out_of_axis_order():
    # `dims` names the spacing components; the materialised grid still lines
    # up with the tensor's own axes.
    x = XTensor(
        torch.zeros(3, 5),
        names=["y", "x"],
        coords={
            "lat": (
                ["x", "y"],
                {"spacing": ([0.5, 1.0], "deg"), "origin": 10.0},
            )
        },
    )
    values = x.coords["lat"]["value"].as_subclass(torch.Tensor)
    expected = 10.0 + torch.arange(3.0).view(3, 1) + 0.5 * torch.arange(5.0)
    assert values.shape == x.shape
    assert torch.allclose(values, expected)
    assert torch.allclose(
        x[1:3, ::2].coords["lat"]["value"].as_subclass(torch.Tensor),
        expected[1:3, ::2],
    )


def test_affine_coordinate_squeeze_folds_a_size_one_dim_exactly():
    # a squeezed dim is always size 1, so folding it (like an integer index
    # would) is exact -- not merely a conservative drop.
    x = XTensor(
        torch.zeros(1, 4),
        names=["y", "x"],
        coords={
            "lat": (
                ["y", "x"],
                {"spacing": ([1.0, 0.5], "deg"), "origin": 10.0},
            )
        },
    )
    full = x.coords["lat"]["value"].as_subclass(torch.Tensor)
    for squeezed in (x.squeeze(), x.squeeze(dim="y")):
        assert squeezed.names == ("x",)
        assert torch.allclose(
            squeezed.coords["lat"]["value"].as_subclass(torch.Tensor),
            full[0, :],
        )
    # every spanned dim squeezed away -> no axis left, drops
    y = XTensor(
        torch.zeros(1, 1),
        names=["y", "x"],
        coords={
            "lat": (
                ["y", "x"],
                {"spacing": ([1.0, 0.5], "deg"), "origin": 10.0},
            )
        },
    )
    assert "lat" not in y.squeeze().coords


def test_affine_coordinate_full_noop_slice_is_a_fast_path():
    # `x[:, :]` shouldn't rebuild `spacing`/`origin` -- verify it stays
    # correct (the implementation returns the coordinate object unchanged).
    x = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={
            "lat": (
                ["y", "x"],
                {"spacing": ([1.0, 0.5], "deg"), "origin": 10.0},
            )
        },
    )
    full = x.coords["lat"]["value"].as_subclass(torch.Tensor)
    noop = x[:, :]
    assert torch.allclose(
        noop.coords["lat"]["value"].as_subclass(torch.Tensor), full
    )


def test_affine_coordinate_gradients_flow_through_spacing_and_origin():
    sy = torch.tensor(1.0, requires_grad=True)
    sx = torch.tensor(0.5, requires_grad=True)
    origin = torch.tensor(10.0, requires_grad=True)
    x = XTensor(
        torch.zeros(3, 4),
        names=["y", "x"],
        coords={
            "lat": (
                ["y", "x"],
                {"spacing": ([sy, sx], "deg"), "origin": origin},
            )
        },
    )
    values = x.coords["lat"]["value"].as_subclass(torch.Tensor)
    values.sum().backward()
    assert sy.grad is not None
    assert sx.grad is not None
    assert origin.grad is not None


# ----------------------------------------------------------------------
# joint affine .sel (issue #82 phase 1)
# ----------------------------------------------------------------------


def _lat_lon_tensor():
    # lat[i,j] = 10 + 1*i + 0*j ; lon[i,j] = 20 + 0*i + 2*j
    field = torch.arange(12.0).reshape(3, 4)
    return XTensor(
        field,
        names=("y", "x"),
        coords={
            "lat": (
                ("y", "x"),
                {"spacing": ([1.0, 0.0], "deg"), "origin": (10.0, "deg")},
            ),
            "lon": (
                ("y", "x"),
                {"spacing": ([0.0, 2.0], "deg"), "origin": (20.0, "deg")},
            ),
        },
    )


def test_affine_sel_joint_query_picks_the_right_position():
    x = _lat_lon_tensor()
    out = x.sel(lat=11.0, lon=24.0)
    assert out.item() == x.as_subclass(torch.Tensor)[1, 2].item()


def test_affine_sel_joint_query_rounds_to_the_nearest_position():
    x = _lat_lon_tensor()
    # a bare joint query is exact-by-default (tolerance=0), matching the 1-D
    # path's contract -- an inexact target needs an explicit mode to snap.
    out = x.sel(mode="round", lat=11.4, lon=23.6)  # nearest is still (1, 2)
    assert out.item() == x.as_subclass(torch.Tensor)[1, 2].item()


def test_affine_sel_bare_joint_query_is_exact_by_default():
    x = _lat_lon_tensor()
    with pytest.raises(ValueError, match="over tolerance"):
        x.sel(lat=11.4, lon=23.6)  # no mode -- tolerance=0, and 11.4 is off


def test_affine_sel_joint_query_respects_an_explicit_tolerance():
    x = _lat_lon_tensor()
    x.sel(mode="round", tolerance=0.5, lat=11.4, lon=23.6)  # within 0.5
    with pytest.raises(ValueError, match="over tolerance"):
        x.sel(mode="round", tolerance=0.1, lat=11.4, lon=23.6)  # not within


def test_affine_sel_joint_query_mixes_with_an_ordinary_indexer():
    field = torch.arange(24.0).reshape(2, 3, 4)
    x = XTensor(
        field,
        names=("t", "y", "x"),
        coords={
            "t": {"spacing": 1.0},
            "lat": (
                ("y", "x"),
                {"spacing": ([1.0, 0.0], "deg"), "origin": (10.0, "deg")},
            ),
            "lon": (
                ("y", "x"),
                {"spacing": ([0.0, 2.0], "deg"), "origin": (20.0, "deg")},
            ),
        },
    )
    out = x.sel(t=1.0, lat=11.0, lon=24.0)
    assert out.item() == field[1, 1, 2].item()


def test_affine_sel_dim_set_both_jointly_and_directly_raises():
    # a dim resolved by the joint solve must not be silently overwritten by
    # an ordinary indexer on that same dim in the same call -- previously
    # this discarded the joint result with no error.
    field = torch.arange(12.0).reshape(3, 4)
    x = XTensor(
        field,
        names=("y", "x"),
        coords={
            "lat": (
                ("y", "x"),
                {"spacing": ([1.0, 0.0], "deg"), "origin": (10.0, "deg")},
            ),
            "lon": (
                ("y", "x"),
                {"spacing": ([0.0, 2.0], "deg"), "origin": (20.0, "deg")},
            ),
            "y": {"spacing": 1.0},
        },
    )
    with pytest.raises(ValueError, match="set both by a joint affine"):
        x.sel(lat=11.0, lon=24.0, y=2.0)


def test_affine_sel_three_way_joint_query():
    field = torch.arange(60.0).reshape(3, 4, 5)
    x = XTensor(
        field,
        names=("a", "b", "c"),
        coords={
            "p": (("a", "b", "c"), {"spacing": ([1.0, 0.0, 0.0], "")}),
            "q": (("a", "b", "c"), {"spacing": ([0.0, 1.0, 0.0], "")}),
            "r": (("a", "b", "c"), {"spacing": ([0.0, 0.0, 1.0], "")}),
        },
    )
    out = x.sel(p=2.0, q=1.0, r=3.0)
    assert out.item() == field[2, 1, 3].item()


def test_affine_sel_under_determined_query_raises():
    x = _lat_lon_tensor()
    with pytest.raises(ValueError, match="needs exactly 2 coordinate"):
        x.sel(lat=11.0)


def test_affine_sel_rejects_a_non_round_mode():
    x = _lat_lon_tensor()
    with pytest.raises(NotImplementedError, match="isn't supported"):
        x.sel(lat=11.0, lon=24.0, mode="floor")


def test_affine_sel_out_of_range_query_raises():
    x = _lat_lon_tensor()
    with pytest.raises(ValueError, match="out of range"):
        x.sel(lat=999.0, lon=24.0)


def test_affine_sel_singular_map_raises():
    field = torch.arange(12.0).reshape(3, 4)
    x = XTensor(
        field,
        names=("y", "x"),
        coords={
            "a": (("y", "x"), {"spacing": ([1.0, 1.0], "")}),
            "b": (("y", "x"), {"spacing": ([2.0, 2.0], "")}),
        },
    )
    with pytest.raises(ValueError, match="isn't invertible"):
        x.sel(a=1.0, b=2.0)


def test_affine_sel_solves_in_float64_precision():
    # a spacing difference near float32 epsilon (1e-7) must not be silently
    # downcast away -- solving in float32 would make the system near-
    # singular and resolve to the wrong index (review finding #3).
    field = torch.arange(16.0).reshape(4, 4)
    x = XTensor(
        field,
        names=("y", "x"),
        coords={
            "p": (("y", "x"), {"spacing": ([1.0, 1.0], "")}),
            "q": (("y", "x"), {"spacing": ([1.0, 1.0 + 1e-7], "")}),
        },
    )
    i, j = 1, 2
    p_target = i * 1.0 + j * 1.0
    q_target = i * 1.0 + j * (1.0 + 1e-7)
    out = x.sel(p=p_target, q=q_target)
    assert out.item() == field[i, j].item()


def test_affine_sel_ordinary_non_dimension_coordinate_still_raises():
    # a 1-D non-dimension coordinate is unaffected -- still needs swap_dims
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={
            "t": {"spacing": 1.0},
            "season": ("t", ("w", "w", "sp", "sp")),
        },
    )
    with pytest.raises(ValueError, match="not an index coordinate"):
        x.sel(season="w")


def test_affine_sel_gradients_do_not_crash_with_a_learnable_spacing():
    sy = torch.tensor(1.0, requires_grad=True)
    sx = torch.tensor(0.0, requires_grad=True)
    field = torch.arange(12.0).reshape(3, 4)
    x = XTensor(
        field,
        names=("y", "x"),
        coords={
            "lat": (("y", "x"), {"spacing": ([sy, sx], "deg")}),
            "lon": (("y", "x"), {"spacing": ([0.0, 2.0], "deg")}),
        },
    )
    out = x.sel(lat=1.0, lon=2.0)
    assert out.item() == field[1, 1].item()


def test_affine_sel_matches_a_brute_force_reference_exhaustively():
    # independent randomized comparison: solve the affine system via a
    # brute-force materialise-and-argmin reference instead of the
    # closed-form inverse under test.
    import random

    rng = random.Random(0)
    for _ in range(200):
        h, w = rng.randint(2, 6), rng.randint(2, 6)
        field = torch.arange(float(h * w)).reshape(h, w)
        s1 = [round(rng.uniform(-3, 3), 2) for _ in range(2)]
        s2 = [round(rng.uniform(-3, 3), 2) for _ in range(2)]
        o1, o2 = round(rng.uniform(-5, 5), 2), round(rng.uniform(-5, 5), 2)
        matrix = torch.tensor([s1, s2])
        if abs(torch.det(matrix).item()) < 1e-3:
            continue  # skip near-singular draws
        x = XTensor(
            field,
            names=("y", "x"),
            coords={
                "p": (("y", "x"), {"spacing": (s1, ""), "origin": (o1, "")}),
                "q": (("y", "x"), {"spacing": (s2, ""), "origin": (o2, "")}),
            },
        )
        i = rng.randint(0, h - 1)
        j = rng.randint(0, w - 1)
        p_val = o1 + s1[0] * i + s1[1] * j
        q_val = o2 + s2[0] * i + s2[1] * j
        try:
            got = x.sel(p=p_val, q=q_val)
        except ValueError:
            continue  # an out-of-range round-trip on a near-degenerate draw
        assert got.item() == field[i, j].item(), (s1, s2, o1, o2, i, j)


# ----------------------------------------------------------------------
# joint affine .interp (issue #82 phase 2)
# ----------------------------------------------------------------------


def test_affine_interp_scalar_query_picks_the_exact_value():
    pytest.importorskip("fiery.interpol")
    x = _lat_lon_tensor()
    field = x.as_subclass(torch.Tensor)
    out = x.interp(lat=11.0, lon=24.0)  # exactly (1, 2)
    assert out.ndim == 0
    assert out.item() == field[1, 2].item()


def test_affine_interp_scalar_query_interpolates_between_ticks():
    pytest.importorskip("fiery.interpol")
    x = _lat_lon_tensor()
    field = x.as_subclass(torch.Tensor)
    out = x.interp(lat=11.5, lon=24.0)  # halfway between row 1 and row 2
    expected = (field[1, 2].item() + field[2, 2].item()) / 2
    assert abs(out.item() - expected) < 1e-6


def test_affine_interp_many_query_produces_one_new_named_axis():
    pytest.importorskip("fiery.interpol")
    x = _lat_lon_tensor()
    field = x.as_subclass(torch.Tensor)
    out = x.interp(lat=[11.0, 12.0], lon=[24.0, 20.0], name="pts")
    assert out.names == ("pts",)
    assert out.shape == (2,)
    assert out[0].item() == field[1, 2].item()
    assert out[1].item() == field[2, 0].item()
    assert out.coords["lat"]["value"].tolist() == [11.0, 12.0]
    assert out.coords["lon"]["value"].tolist() == [24.0, 20.0]


def test_affine_interp_new_axis_defaults_to_unnamed():
    pytest.importorskip("fiery.interpol")
    x = _lat_lon_tensor()
    out = x.interp(lat=[11.0, 12.0], lon=[24.0, 20.0])  # no name= given
    assert out.names == (None,)


def test_affine_interp_infers_the_new_axis_name_from_a_named_indexer():
    # mirrors xarray's own vectorized-indexing convention: the *indexer*
    # array's own dim name becomes the result's new dimension, no separate
    # name= needed.
    pytest.importorskip("fiery.interpol")
    x = _lat_lon_tensor()
    field = x.as_subclass(torch.Tensor)
    lat_q = XTensor(torch.tensor([11.0, 12.0]), names=("pts",))
    out = x.interp(lat=lat_q, lon=[24.0, 20.0])
    assert out.names == ("pts",)
    assert out[0].item() == field[1, 2].item()
    assert out[1].item() == field[2, 0].item()


def test_affine_interp_agreeing_named_indexers_are_fine():
    pytest.importorskip("fiery.interpol")
    x = _lat_lon_tensor()
    lat_q = XTensor(torch.tensor([11.0, 12.0]), names=("pts",))
    lon_q = XTensor(torch.tensor([24.0, 20.0]), names=("pts",))
    out = x.interp(lat=lat_q, lon=lon_q)
    assert out.names == ("pts",)


def test_affine_interp_disagreeing_named_indexers_raise():
    pytest.importorskip("fiery.interpol")
    x = _lat_lon_tensor()
    lat_q = XTensor(torch.tensor([11.0, 12.0]), names=("pts",))
    lon_q = XTensor(torch.tensor([24.0, 20.0]), names=("other",))
    with pytest.raises(ValueError, match="disagree on the new axis's name"):
        x.interp(lat=lat_q, lon=lon_q)


def test_affine_interp_explicit_name_overrides_and_resolves_conflicts():
    pytest.importorskip("fiery.interpol")
    x = _lat_lon_tensor()
    lat_q = XTensor(torch.tensor([11.0, 12.0]), names=("pts",))
    lon_q = XTensor(torch.tensor([24.0, 20.0]), names=("other",))
    # an explicit name= wins outright, even resolving a naming conflict
    out = x.interp(lat=lat_q, lon=lon_q, name="resolved")
    assert out.names == ("resolved",)
    out2 = x.interp(lat=lat_q, lon=[24.0, 20.0], name="explicit")
    assert out2.names == ("explicit",)


def test_affine_interp_rejects_a_non_string_name():
    # name= binds to this parameter before a same-named indexer ever
    # reaches **indexers_kwargs -- so a dim literally called "name" queried
    # as interp(name=3.0) would otherwise silently do nothing at all
    # (indexers ends up empty). A loud TypeError beats that silent no-op.
    x = _lat_lon_tensor()
    with pytest.raises(TypeError, match="name= must be a str or None"):
        x.interp(lat=11.0, lon=24.0, name=3.0)


def test_affine_interp_new_axis_lands_at_the_left_most_spanned_position():
    pytest.importorskip("fiery.interpol")
    field = torch.arange(24.0).reshape(2, 3, 4)
    x = XTensor(
        field,
        names=("t", "y", "x"),  # y, x are not the tensor's leading axes
        coords={
            "t": {"spacing": 1.0},
            "lat": (
                ("y", "x"),
                {"spacing": ([1.0, 0.0], "deg"), "origin": (10.0, "deg")},
            ),
            "lon": (
                ("y", "x"),
                {"spacing": ([0.0, 2.0], "deg"), "origin": (20.0, "deg")},
            ),
        },
    )
    out = x.interp(lat=[11.0, 12.0], lon=[24.0, 20.0], name="pts")
    assert out.names == ("t", "pts")
    assert out.shape == (2, 2)
    assert out[:, 0].tolist() == field[:, 1, 2].tolist()
    assert out[:, 1].tolist() == field[:, 2, 0].tolist()


def test_affine_interp_mixes_with_an_ordinary_indexer():
    pytest.importorskip("fiery.interpol")
    field = torch.arange(24.0).reshape(2, 3, 4)
    x = XTensor(
        field,
        names=("t", "y", "x"),
        coords={
            "t": {"spacing": 1.0},
            "lat": (
                ("y", "x"),
                {"spacing": ([1.0, 0.0], "deg"), "origin": (10.0, "deg")},
            ),
            "lon": (
                ("y", "x"),
                {"spacing": ([0.0, 2.0], "deg"), "origin": (20.0, "deg")},
            ),
        },
    )
    out = x.interp(t=0.5, lat=11.0, lon=24.0)
    expected = (field[0, 1, 2].item() + field[1, 1, 2].item()) / 2
    assert out.item() == expected


def test_affine_interp_broadcasts_a_length_one_query_against_a_list():
    pytest.importorskip("fiery.interpol")
    x = _lat_lon_tensor()
    field = x.as_subclass(torch.Tensor)
    out = x.interp(lat=11.0, lon=[24.0, 20.0], name="pts")
    assert out.shape == (2,)
    assert out[0].item() == field[1, 2].item()
    assert out[1].item() == field[1, 0].item()


def test_affine_interp_mismatched_lengths_raises():
    x = _lat_lon_tensor()
    with pytest.raises(ValueError, match="same length"):
        x.interp(lat=[11.0, 12.0], lon=[24.0, 20.0, 22.0])


def test_affine_interp_under_determined_query_raises():
    x = _lat_lon_tensor()
    with pytest.raises(ValueError, match="needs exactly 2 coordinate"):
        x.interp(lat=11.0)


def test_affine_interp_singular_map_raises():
    field = torch.arange(12.0).reshape(3, 4)
    x = XTensor(
        field,
        names=("y", "x"),
        coords={
            "a": (("y", "x"), {"spacing": ([1.0, 1.0], "")}),
            "b": (("y", "x"), {"spacing": ([2.0, 2.0], "")}),
        },
    )
    with pytest.raises(ValueError, match="isn't invertible"):
        x.interp(a=1.0, b=2.0)


def test_affine_interp_multiple_groups_in_one_call_raises():
    field = torch.arange(16.0).reshape(2, 2, 2, 2)
    x = XTensor(
        field,
        names=("y", "x", "z", "w"),
        coords={
            "lat": (("y", "x"), {"spacing": ([1.0, 0.0], "")}),
            "lon": (("y", "x"), {"spacing": ([0.0, 1.0], "")}),
            "p": (("z", "w"), {"spacing": ([1.0, 0.0], "")}),
            "q": (("z", "w"), {"spacing": ([0.0, 1.0], "")}),
        },
    )
    with pytest.raises(NotImplementedError, match="more than one"):
        x.interp(lat=0.0, lon=1.0, p=0.0, q=1.0)


def test_affine_interp_empty_query_returns_an_empty_axis():
    pytest.importorskip("fiery.interpol")
    x = _lat_lon_tensor()
    out = x.interp(lat=[], lon=[], name="pts")
    assert out.shape == (0,)
    assert out.names == ("pts",)


def test_affine_interp_empty_query_broadcasts_against_a_scalar_sibling():
    # a length-1 (or scalar) sibling has to broadcast *down* to empty, not
    # just up -- previously only "n > 1" expanded, so an empty query paired
    # with a scalar/length-1 one crashed inside torch.stack instead of
    # producing a well-formed empty axis (review finding on PR #124).
    pytest.importorskip("fiery.interpol")
    x = _lat_lon_tensor()
    out = x.interp(lat=[], lon=2.0, name="pts")
    assert out.shape == (0,)
    assert out.names == ("pts",)
    out2 = x.interp(lat=[], lon=[1.0], name="pts")
    assert out2.shape == (0,)


def test_affine_interp_nearest_works_without_the_backend(monkeypatch):
    from fiery.xtensor import _tensors

    monkeypatch.setattr(_tensors, "_interpol", lambda: None)
    x = _lat_lon_tensor()
    field = x.as_subclass(torch.Tensor)
    out = x.interp(lat=11.4, lon=23.6, method="nearest")  # nearest -> (1, 2)
    assert out.item() == field[1, 2].item()


def test_affine_interp_higher_order_needs_the_backend_too(monkeypatch):
    from fiery.xtensor import _tensors

    monkeypatch.setattr(_tensors, "_interpol", lambda: None)
    x = _lat_lon_tensor()
    with pytest.raises(ImportError, match="fiery-xtensor\\[interp\\]"):
        x.interp(lat=11.0, lon=24.0, method="linear")


def test_affine_interp_gradients_flow_through_a_learnable_spacing():
    pytest.importorskip("fiery.interpol")
    spacing = torch.tensor([1.0, 0.0], requires_grad=True)
    x = XTensor(
        torch.arange(12.0).reshape(3, 4),
        names=("y", "x"),
        coords={
            "lat": (
                ("y", "x"),
                {"spacing": (spacing, "deg"), "origin": (10.0, "deg")},
            ),
            "lon": (
                ("y", "x"),
                {"spacing": ([0.0, 2.0], "deg"), "origin": (20.0, "deg")},
            ),
        },
    )
    x.interp(lat=11.0, lon=24.0).backward()
    assert spacing.grad is not None
    assert not torch.isnan(spacing.grad).any()


def test_affine_interp_matches_a_brute_force_bilinear_reference():
    pytest.importorskip("fiery.interpol")
    # independent randomized comparison: compute the bilinear value by hand
    # (manual 2x2 neighbourhood + weights) instead of via the closed-form
    # inverse + grid_pull path under test.
    import random

    rng = random.Random(0)
    for _ in range(100):
        h, w = rng.randint(3, 6), rng.randint(3, 6)
        field = torch.rand(h, w, dtype=torch.float64)
        x = XTensor(
            field,
            names=("y", "x"),
            coords={
                "lat": (("y", "x"), {"spacing": ([1.0, 0.0], "")}),
                "lon": (("y", "x"), {"spacing": ([0.0, 1.0], "")}),
            },
        )
        fi = rng.uniform(0, h - 1)
        fj = rng.uniform(0, w - 1)
        i0, j0 = int(fi), int(fj)
        i1, j1 = min(i0 + 1, h - 1), min(j0 + 1, w - 1)
        di, dj = fi - i0, fj - j0
        expected = (
            field[i0, j0].item() * (1 - di) * (1 - dj)
            + field[i0, j1].item() * (1 - di) * dj
            + field[i1, j0].item() * di * (1 - dj)
            + field[i1, j1].item() * di * dj
        )
        got = x.interp(lat=fi, lon=fj).item()
        assert abs(got - expected) < 1e-5, (h, w, fi, fj)


def test_non_dimension_coordinate_length_is_checked():
    with pytest.raises(ValueError, match="has 2 values for dim"):
        XTensor(
            torch.arange(4.0), names=("t",), coords={"s": ("t", ["a", "b"])}
        )


def test_non_dimension_coordinate_drops_when_its_dim_is_reordered():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"season": ("t", ["w", "w", "sp", "sp"])},
    )
    # a rider's positions no longer correspond once its dim is
    # sorted/flipped/rolled/gathered/index_selected -- conservatively dropped,
    # same as the dimension coordinate itself would be
    assert "season" not in x.flip("t").coords
    assert "season" not in x.roll(1, "t").coords
    assert "season" not in x.sort("t").values.coords
    idx = torch.tensor([0, 1, 2, 3])
    assert "season" not in x.gather("t", idx).coords
    assert "season" not in x.index_select("t", idx[:2]).coords


def test_rename_raises_on_a_coordinate_name_collision():
    # renaming an axis onto an existing non-dimension coordinate's name would
    # otherwise silently drop one of the two entries (dict key collision)
    x = XTensor(
        torch.arange(6.0).reshape(2, 3),
        names=("t", "u"),
        coords={"season": ("t", ["w", "sp"]), "u": ["x", "y", "z"]},
    )
    with pytest.raises(ValueError, match="coordinate name collision"):
        x.rename(u="season")
    # a non-colliding rename is unaffected
    renamed = x.rename(u="month")
    assert renamed.coords["season"] == ("w", "sp")
    assert renamed.coords["month"] == ("x", "y", "z")


def test_coords_constructor_checks_label_count():
    with pytest.raises(ValueError, match="labels for size"):
        XTensor(
            torch.zeros(2, 3), names=("row", "col"), coords={"col": ("a", "b")}
        )


def test_coords_ellipsis_fills_unlabelled_positions():
    x = XTensor(
        torch.zeros(2, 4),
        names=("row", "col"),
        coords={"col": ("a", ..., "z")},
    )
    assert x.coords["col"] == ("a", None, None, "z")


# -- compact numeric coordinates (Proposal 0001 phase 1) ----------------------


def test_compact_numeric_coordinate_stores_and_materialises():
    x = XTensor(
        torch.zeros(2, 4),
        names=("y", "x"),
        coords={"x": {"spacing": (0.5, "mm"), "origin": (-1.0, "mm")}},
    )
    cx = x.coords["x"]
    assert cx["spacing"]["value"] == 0.5
    assert cx["spacing"].unit == "mm"  # attribute sugar for a safe key
    # `["value"]` is a derived key: origin + i*spacing
    assert cx["value"].tolist() == [-1.0, -0.5, 0.0, 0.5]
    assert cx["value"].units == "mm"  # the POSITION unit


def test_numeric_and_categorical_coords_coexist():
    x = XTensor(
        torch.zeros(3, 4),
        names=("c", "x"),
        coords={"c": ["r", "g", "b"], "x": {"spacing": (0.5, "mm")}},
    )
    assert x.coords["c"] == ("r", "g", "b")  # labels
    assert x.coords["x"]["value"].tolist() == [0.0, 0.5, 1.0, 1.5]


def test_numeric_coord_propagates_and_drops_with_its_axis():
    x = XTensor(
        torch.zeros(2, 4),
        names=("y", "x"),
        coords={"x": {"spacing": (0.5, "mm")}},
    )
    assert x.T.coords["x"]["spacing"]["value"] == 0.5  # transpose keeps it
    reduced = x.sum(dim="y")
    assert reduced.coords["x"]["spacing"]["value"] == 0.5  # reduce other axis
    assert "x" not in x.sum(dim="x").coords  # reducing its axis drops it
    assert x.rename(x="u").coords["u"]["spacing"]["value"] == 0.5  # rename


def test_numeric_coord_attribute_sugar_does_not_shadow_dict_api():
    x = XTensor(
        torch.zeros(4), names=("x",), coords={"x": {"spacing": (2, "m")}}
    )
    sp = x.coords["x"]["spacing"]
    assert callable(sp.values)  # `values` stays the dict method...
    assert sp["value"] == 2  # ...the key is item-access only
    assert sp.unit == "m"  # safe key reachable as an attribute


def test_learnable_spacing_keeps_its_gradient():
    pytest.importorskip("pint")
    leaf = torch.tensor(0.5, requires_grad=True)
    with set_options(unit_backend="pint"):
        step = XTensor(leaf, units="mm")
        x = XTensor(
            torch.zeros(5), names=("t",), coords={"t": {"spacing": step}}
        )
        x.coords["t"]["value"].sum().backward()
        assert leaf.grad.item() == 10.0  # d/dstep sum(i*step) = 0+1+2+3+4


def test_as_xtensor_from_a_bare_scalar():
    out = as_xtensor(2.0, units="mm")
    assert isinstance(out, XTensor)
    assert out.item() == 2.0
    assert out.units == "mm"
    assert not out.requires_grad


def test_as_xtensor_does_not_force_a_float_dtype():
    # must not change the input's natural dtype (review comment on #112):
    # an int value stays int64, matching what Unitful's old do-nothing
    # storage already let downstream arithmetic produce.
    assert as_xtensor(2, units="").dtype == torch.int64
    assert as_xtensor(2.0, units="").dtype == torch.get_default_dtype()


def test_as_xtensor_preserves_the_graph_of_an_existing_tensor():
    # the torch.tensor(existing_tensor) footgun: it always copies, silently
    # returning requires_grad=False even when the input required grad. This
    # must go through the as_subclass-based (torch.as_tensor-like) path
    # instead, which never detaches.
    leaf = torch.tensor(2.0, requires_grad=True)
    out = as_xtensor(leaf, units="mm")
    assert out.requires_grad
    assert out.data_ptr() == leaf.data_ptr()  # same storage, no copy at all
    (out.as_subclass(torch.Tensor) * 3).sum().backward()
    assert leaf.grad.item() == 3.0


def test_as_xtensor_preserves_the_graph_across_a_dtype_conversion():
    # a genuine dtype conversion still has to happen (float32 -> float64),
    # so it can't be the *same* tensor -- but it must stay a differentiable
    # op, not a detaching copy.
    leaf = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
    out = as_xtensor(leaf.to(torch.float64), units="mm")
    assert out.requires_grad
    assert out.dtype == torch.float64
    (out.as_subclass(torch.Tensor) * 5).sum().backward()
    assert leaf.grad.item() == 5.0


def test_as_xtensor_from_a_plain_non_xtensor_tensor():
    leaf = torch.tensor(3.0, requires_grad=True)
    out = as_xtensor(leaf, units="s")
    assert isinstance(out, XTensor)
    assert out.units == "s"
    assert out.requires_grad
    (out.as_subclass(torch.Tensor) * 2).sum().backward()
    assert leaf.grad.item() == 2.0


def test_as_xtensor_with_no_overrides_is_a_true_passthrough():
    # value is already an XTensor and nothing is overridden -- the exact
    # same object comes back (metadata, graph, and identity all untouched),
    # matching torch.as_tensor's own "no conversion requested" contract.
    x = XTensor(
        torch.arange(4.0, requires_grad=True),
        names=("t",),
        coords={"t": {"spacing": 1.0}},
        units="mm",
    )
    out = as_xtensor(x)
    assert out is x


def test_as_xtensor_preserves_unit_names_coords_by_default():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 1.0}},
        units="mm",
    )
    out = as_xtensor(x, units="s")  # override only unit
    assert out.units == "s"
    assert out.names == ("t",)  # preserved
    assert out.coords["t"]["value"].tolist() == [
        1.0,
        3.0,
        5.0,
        7.0,
    ]  # preserved


def test_as_xtensor_renaming_goes_stale_the_same_way_a_direct_rename_does():
    # overriding `names` while a coordinate is keyed by the *old* name isn't
    # a rename-and-follow -- it goes stale (filtered out by `.coords`'s own
    # getter), exactly like reassigning `.names` directly on an existing
    # `XTensor` already does; pass `coords=` explicitly to set up a
    # coordinate under the new name instead.
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    out = as_xtensor(x, names=("u",))
    assert out.coords == {}
    # matches: a plain rename on the original already drops it this way too
    x.names = ("u",)
    assert x.coords == {}


def test_as_xtensor_override_replaces_wholesale_not_merge():
    x = XTensor(torch.arange(3.0), names=("c",), coords={"c": ("r", "g", "b")})
    out = as_xtensor(x, coords={"c": ("x", "y", "z")})
    assert out.coords["c"] == ("x", "y", "z")  # replaced, not merged/appended


def test_as_xtensor_never_mutates_the_original_when_overriding():
    x = XTensor(torch.arange(3.0), names=("c",), units="mm")
    out = as_xtensor(x, units="s")
    assert out.units == "s"
    assert x.units == "mm"  # the original is untouched
    assert out is not x


def test_as_xtensor_preserves_axis_descriptors_when_overriding_the_unit():
    x = XTensor(
        torch.arange(2.0),
        axes=[{"name": "x", "type": "space", "unit": "mm"}],
    )
    out = as_xtensor(x, units="s")  # override unrelated metadata
    assert out.axes[0]["type"] == "space"  # descriptor still rides through


def test_as_xtensor_explicit_none_clears_the_unit():
    x = XTensor(torch.arange(3.0), units="mm")
    out = as_xtensor(x, units=None)
    assert out.units is None


def test_as_xtensor_dtype_override_converts_and_preserves_metadata():
    # torch.as_tensor(an_xtensor, dtype=...) silently degrades to a plain
    # Tensor when it actually has to convert -- as_xtensor must not.
    x = XTensor(
        torch.arange(4, dtype=torch.int64),
        names=("t",),
        coords={"t": {"spacing": 1}},
        units="mm",
    )
    plain = torch.as_tensor(x, dtype=torch.float64)
    assert not isinstance(plain, XTensor)  # the footgun this avoids

    out = as_xtensor(x, dtype=torch.float64)
    assert isinstance(out, XTensor)
    assert out.dtype == torch.float64
    assert out.names == ("t",)
    assert out.units == "mm"
    # the coordinate itself isn't converted -- only the data's own dtype is
    # -- so its dtype is untouched (int64), not just numerically equal.
    assert out.coords["t"]["value"].dtype == torch.int64
    assert out.coords["t"]["value"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_as_xtensor_device_override_actually_converts():
    # a genuine (GPU-free) device conversion, not just the cpu->cpu no-op
    # the other device test covers -- "meta" exercises the real .to() path,
    # where available (old torch doesn't recognise it as a device type).
    try:
        torch.device("meta")
    except RuntimeError:
        pytest.skip("this torch build has no 'meta' device")
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    out = as_xtensor(x, device="meta")
    assert isinstance(out, XTensor)
    assert out.device.type == "meta"
    assert out.names == ("t",)


def test_as_xtensor_dtype_none_is_a_true_passthrough():
    # dtype=None (the default) means "leave as is", same as torch.as_tensor
    # -- combined with no metadata override, still a strict identity return.
    x = XTensor(torch.arange(4.0), names=("t",), units="mm")
    assert as_xtensor(x, dtype=None, device=None) is x


def test_as_xtensor_dtype_override_is_graph_safe():
    leaf = torch.tensor(2.0, dtype=torch.float32, requires_grad=True)
    x = XTensor(leaf, names=())
    out = as_xtensor(x, dtype=torch.float64)
    assert out.requires_grad
    assert out.dtype == torch.float64
    (out.as_subclass(torch.Tensor) * 5).sum().backward()
    assert leaf.grad.item() == 5.0


def test_as_xtensor_dtype_override_from_a_bare_number():
    out = as_xtensor(2, dtype=torch.float64, units="mm")
    assert out.dtype == torch.float64
    assert out.units == "mm"


def test_as_xtensor_no_op_dtype_override_keeps_identity():
    # matching the current dtype is a no-op -- as_xtensor skips calling
    # .to() at all in this case (rather than relying on .to()'s own
    # no-op-returns-self behaviour, which isn't consistent across the
    # torch versions this library supports), so this is a true passthrough.
    x = XTensor(torch.arange(3.0), names=("c",))
    assert as_xtensor(x, dtype=x.dtype) is x


def test_as_xtensor_device_override_is_a_noop_on_the_same_device():
    x = XTensor(torch.arange(3.0), names=("c",))
    assert as_xtensor(x, device="cpu") is x


def test_as_xtensor_dtype_override_composes_with_metadata_override():
    x = XTensor(torch.arange(3, dtype=torch.int64), names=("c",), units="mm")
    out = as_xtensor(x, dtype=torch.float64, units="s")
    assert out.dtype == torch.float64
    assert out.units == "s"
    assert out.names == ("c",)  # untouched metadata still rides through


def test_is_xtensor_true_for_an_xtensor():
    assert is_xtensor(XTensor(torch.arange(3.0), names=("c",)))


def test_is_xtensor_false_for_a_plain_tensor_or_number():
    assert not is_xtensor(torch.arange(3.0))
    assert not is_xtensor(2.0)
    assert not is_xtensor("not a tensor")


def test_spacing_unitful_converts():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.zeros(4),
            names=("x",),
            coords={"x": {"spacing": (2.0, "mm")}},
        )
        converted = x.coords["x"]["spacing"].to("um")
        assert converted["value"] == 2000.0
        assert converted["unit"] == "micrometer"


# -- explicit numeric coords + slicing + conversion (0001 phase 2) ------------


def test_explicit_numeric_coordinate_stores_positions():
    t = XTensor(torch.tensor([0.0, 0.5, 2.0, 4.0]), units="s")
    x = XTensor(torch.arange(4.0), names=("t",), coords={"t": t})
    assert x.coords["t"]["value"].tolist() == [0.0, 0.5, 2.0, 4.0]
    assert x.coords["t"]["value"].units == "s"  # position unit


def test_explicit_dict_form_is_equivalent_to_a_bare_tensor():
    # {"value": ...} is sugar for the same explicit coordinate a bare
    # tensor/list produces -- not a distinct third coordinate kind.
    bare = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": XTensor(torch.tensor([0.0, 0.5, 2.0, 4.0]), units="s")},
    )
    dict_form = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": {"value": XTensor([0.0, 0.5, 2.0, 4.0], units="s")}},
    )
    assert isinstance(dict_form.coords["t"], Coordinate)
    assert (
        dict_form.coords["t"]["value"].tolist()
        == bare.coords["t"]["value"].tolist()
    )
    assert dict_form.coords["t"]["value"].units == "s"
    assert dict_form.sel(t=0.5).item() == 1.0


def test_bare_list_of_numbers_is_equivalent_to_the_explicit_dict_form():
    from_list = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": (0.0, 0.5, 2.0, 4.0)}
    )
    from_dict = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": {"value": [0.0, 0.5, 2.0, 4.0]}},
    )
    assert isinstance(from_dict.coords["t"], Coordinate)
    assert (
        from_list.coords["t"]["value"].tolist()
        == from_dict.coords["t"]["value"].tolist()
    )
    assert from_dict.coords["t"]["value"].units == ""
    assert (
        from_dict.coords["t"]["value"].dtype
        == from_list.coords["t"]["value"].dtype
    )


def test_explicit_dict_form_works_for_a_curvilinear_multi_dim_coordinate():
    lat = torch.tensor([[10.0, 11.0], [12.0, 13.0]])
    x = XTensor(
        torch.arange(4.0).reshape(2, 2),
        names=("y", "x"),
        coords={
            "lat": (("y", "x"), {"value": XTensor(lat, units="deg")}),
        },
    )
    assert isinstance(x.coords["lat"], Coordinate)
    assert x.coords["lat"]["value"].tolist() == lat.tolist()
    assert x.coords["lat"]["value"].units == "deg"


def test_coordinate_value_key_is_attribute_accessible():
    # renamed from "values" specifically so it doesn't collide with dict's
    # own `.values()` method -- `.value` is a real, working shortcut now,
    # for an explicit coordinate.
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": (0.0, 0.5, 2.0, 4.0)}
    )
    assert x.coords["t"].value.tolist() == [0.0, 0.5, 2.0, 4.0]


def test_coordinate_value_attribute_still_raises_for_a_compact_coordinate():
    # a known, pre-existing MagicDict limitation: "value" is a *derived* key
    # for a compact coordinate (materialised only via bracket access), so it
    # isn't seen by __getattr__'s "is this key actually present" check.
    # Bracket access (["value"]) works either way.
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    assert x.coords["t"]["value"].tolist() == [0.0, 1.0, 2.0, 3.0]
    with pytest.raises(AttributeError):
        _ = x.coords["t"].value


def test_mixed_compact_and_explicit_coord_spec_raises():
    with pytest.raises(ValueError, match="cannot mix"):
        XTensor(
            torch.arange(4.0),
            names=("t",),
            coords={"t": {"spacing": 1.0, "value": [0.0, 1.0, 2.0, 3.0]}},
        )


def test_mixed_compact_and_explicit_multi_dim_coord_spec_raises():
    with pytest.raises(ValueError, match="cannot mix"):
        XTensor(
            torch.arange(4.0).reshape(2, 2),
            names=("y", "x"),
            coords={
                "lat": (
                    ("y", "x"),
                    {"spacing": [1.0, 2.0], "value": torch.zeros(2, 2)},
                )
            },
        )


def test_explicit_dict_form_length_mismatch_raises():
    with pytest.raises(ValueError, match="has 2 values for size 4"):
        XTensor(
            torch.arange(4.0),
            names=("t",),
            coords={"t": {"value": [0.0, 1.0]}},
        )


def test_bare_tensor_length_mismatch_raises():
    # a bare tensor used to silently drop the coordinate on a length
    # mismatch instead of raising, unlike the label-promotion path.
    with pytest.raises(ValueError, match="has 2 values for size 4"):
        XTensor(
            torch.arange(4.0),
            names=("t",),
            coords={"t": torch.tensor([0.0, 1.0])},
        )


def test_compact_coord_slices_affinely():
    x = XTensor(
        torch.arange(8.0),
        names=("x",),
        coords={"x": {"spacing": (0.5, "mm"), "origin": (0.0, "mm")}},
    )
    # an offset slice shifts origin; a strided slice scales spacing
    off = x[2:].coords["x"]["value"].tolist()
    assert off == [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]
    assert x[::2].coords["x"]["value"].tolist() == [0.0, 1.0, 2.0, 3.0]
    assert x[1:6:2].coords["x"]["value"].tolist() == [0.5, 1.5, 2.5]
    assert "x" not in x[3].coords  # integer index drops the axis + coord


def test_explicit_coord_slices_including_advanced():
    t = XTensor(torch.tensor([0.0, 0.5, 2.0, 4.0]), units="s")
    x = XTensor(torch.arange(4.0), names=("t",), coords={"t": t})
    assert x[1:3].coords["t"]["value"].tolist() == [0.5, 2.0]
    assert x[[0, 2]].coords["t"]["value"].tolist() == [0.0, 2.0]


def test_compact_advanced_index_materialises_to_explicit():
    x = XTensor(
        torch.arange(8.0),
        names=("x",),
        coords={"x": {"spacing": (0.5, "mm")}},
    )
    picked = x[[1, 3]].coords["x"]
    assert not picked._compact()  # became an explicit coordinate
    assert picked["value"].tolist() == [0.5, 1.5]


def test_coordinate_converts_its_position_unit():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        compact = XTensor(
            torch.arange(4.0),
            names=("x",),
            coords={"x": {"spacing": (2.0, "mm")}},
        )
        assert compact.coords["x"].to("um")["spacing"]["value"] == 2000.0
        explicit = XTensor(
            torch.arange(3.0),
            names=("t",),
            coords={"t": XTensor(torch.tensor([1.0, 2.0, 3.0]), units="mm")},
        )
        got = explicit.coords["t"].to("um")["value"].tolist()
        assert got == [1000.0, 2000.0, 3000.0]


def test_sel_by_numeric_coordinate_value():
    # values 0, 0.5, 1.0, ... 3.5 along t
    x = XTensor(
        torch.arange(8.0),
        names=("t",),
        coords={"t": {"spacing": (0.5, "s"), "origin": (0.0, "s")}},
    )
    assert x.sel(t=1.5).item() == 3.0  # exact value -> index 3
    assert x.sel(t=[0.5, 2.0]).tolist() == [1.0, 4.0]  # a list keeps the axis
    with pytest.raises(ValueError, match="over tolerance"):
        x.sel(t=1.7)  # bare sel is exact (tolerance 0)
    # a mode implies an unbounded snap -> nearest tick is 1.5 (index 3)
    assert x.sel(t=1.7, mode="round").item() == 3.0
    assert x.sel(t=1.7, method="nearest").item() == 3.0  # xarray alias
    with pytest.raises(ValueError, match="over tolerance"):
        x.sel(t=1.7, mode="round", tolerance=0.1)


def test_sel_accepts_indexers_as_a_dict():
    x = XTensor(
        torch.arange(5.0), names=("t",), coords={"t": {"spacing": 2.0}}
    )
    assert x.sel({"t": 2.0}).item() == 1.0


def test_sel_dict_escape_hatch_reaches_a_dim_named_like_a_keyword():
    # a dim literally named "mode" can never be spelled as a keyword
    # argument -- sel(mode=...) always binds sel's own parameter, never
    # the indexers -- so the dict form is the only way to reach it.
    x = XTensor(
        torch.arange(5.0), names=("mode",), coords={"mode": {"spacing": 2.0}}
    )
    assert x.sel({"mode": 2.0}).item() == 1.0


def test_sel_indexers_dict_and_kwargs_together_raises():
    x = XTensor(
        torch.arange(5.0), names=("t",), coords={"t": {"spacing": 2.0}}
    )
    with pytest.raises(ValueError, match="dict OR as keyword arguments"):
        x.sel({"t": 2.0}, t=2.0)


def test_sel_too_many_positional_indexers_raises():
    x = XTensor(
        torch.arange(5.0), names=("t",), coords={"t": {"spacing": 2.0}}
    )
    with pytest.raises(TypeError, match="at most one positional argument"):
        x.sel({"t": 2.0}, {"t": 3.0})


def test_sel_indexers_is_captured_positionally_not_by_keyword():
    # the escape hatch mustn't introduce the exact collision it fixes: a
    # dim literally named "indexers" still has to work as an ordinary
    # keyword argument, since the positional mapping is captured via
    # *args, never a named `indexers=` parameter.
    x = XTensor(
        torch.arange(5.0),
        names=("indexers",),
        coords={"indexers": {"spacing": 2.0}},
    )
    assert x.sel(indexers=2.0).item() == 1.0


def test_sel_dict_form_reaches_the_joint_affine_path():
    x = _lat_lon_tensor()
    field = x.as_subclass(torch.Tensor)
    out = x.sel({"lat": 11.0, "lon": 24.0})
    assert out.item() == field[1, 2].item()


def test_sel_attribute_style_label_access_on_a_colliding_dim_name():
    # x.<label> resolves through .sel internally -- must still work for a
    # dim literally named "mode"/"tolerance"/"method".
    x = XTensor(
        torch.arange(3.0), names=("mode",), coords={"mode": ["a", "b", "c"]}
    )
    assert x.a.item() == 0.0


def test_sel_modes_round_floor_ceil_prev_next():
    # ascending ticks 0,2,4,6,8 ; data 0,10,20,30,40
    x = XTensor(
        torch.arange(5.0) * 10,
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    assert x.sel(t=5.0, mode="floor").item() == 20.0  # value 4
    assert x.sel(t=5.0, mode="ceil").item() == 30.0  # value 6
    # ascending: prev == floor, next == ceil
    assert x.sel(t=5.0, mode="prev").item() == 20.0
    assert x.sel(t=5.0, mode="next").item() == 30.0


def test_sel_modes_split_value_vs_tickorder_on_descending():
    # descending ticks 8,6,4,2,0 ; data 0,10,20,30,40
    d = XTensor(
        torch.arange(5.0) * 10,
        names=("t",),
        coords={"t": XTensor(torch.tensor([8.0, 6.0, 4.0, 2.0, 0.0]))},
    )
    # value-space floor/ceil are orientation-robust
    assert d.sel(t=5.0, mode="floor").item() == 20.0  # value 4
    assert d.sel(t=5.0, mode="ceil").item() == 10.0  # value 6
    # tick-order prev/next SWAP vs floor/ceil on a descending coordinate
    assert d.sel(t=5.0, mode="prev").item() == 10.0  # == ceil here
    assert d.sel(t=5.0, mode="next").item() == 20.0  # == floor here


def test_sel_compact_descending_modes_split_value_vs_tickorder():
    # same as the explicit-coordinate version above, but compact (negative
    # spacing) -- exercises the closed-form path's ascending/descending swap.
    d = XTensor(
        torch.arange(5.0) * 10,
        names=("t",),
        coords={"t": {"spacing": -2.0, "origin": 8.0}},  # ticks 8,6,4,2,0
    )
    assert d.sel(t=5.0, mode="floor").item() == 20.0  # value 4
    assert d.sel(t=5.0, mode="ceil").item() == 10.0  # value 6
    assert d.sel(t=5.0, mode="prev").item() == 10.0  # == ceil here
    assert d.sel(t=5.0, mode="next").item() == 20.0  # == floor here


def test_sel_compact_never_materialises(monkeypatch):
    # the whole point of #110: resolving *which* position(s) a scalar
    # selector targets on a compact coordinate must be O(1), never touching
    # Coordinate._materialise / a search over the full array. (A *list*
    # selector is a separate matter -- carrying the resulting several
    # positions through __getitem__ as an advanced index always materialises
    # the coordinate for the surviving positions, compact or not; that's
    # _slice_coordinate's pre-existing behaviour, not part of this path.)
    def boom(self):
        raise AssertionError("materialise() called -- O(1) fast path bypassed")

    monkeypatch.setattr(Coordinate, "_materialise", boom)
    x = XTensor(
        torch.arange(1_000_000.0),
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.0}},
    )
    assert x.sel(t=500_000.5, mode="round").item() == 500000.0
    assert x.sel(t=500_000.5, mode="floor").item() == 500000.0
    assert x.sel(t=500_000.5, mode="ceil").item() == 500001.0


def test_sel_compact_floor_ceil_clamp_beyond_the_whole_coordinate():
    # a target beyond every tick still has a floor (the last tick) / ceil
    # (the first tick) match -- it's not "no tick", unlike a target beyond
    # the coordinate on the *unsatisfiable* side, which is.
    x = XTensor(
        torch.arange(5.0) * 10,  # ticks 0,1,2,3,4
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.0}},
    )
    assert x.sel(t=100.0, mode="floor").item() == 40.0  # clamps to the last
    with pytest.raises(ValueError, match="no ceil tick"):
        x.sel(t=100.0, mode="ceil")  # nothing is >= 100
    assert x.sel(t=-100.0, mode="ceil").item() == 0.0  # clamps to the first
    with pytest.raises(ValueError, match="no floor tick"):
        x.sel(t=-100.0, mode="floor")  # nothing is <= -100


def test_sel_compact_exact_tick_is_robust_to_division_rounding():
    # (target - origin) / spacing can land a hair off an exact integer due
    # to floating-point division noise even when the target IS exactly on a
    # tick -- must still resolve exactly, the same way the direct
    # origin + i*spacing comparison the search path uses would.
    x = XTensor(
        torch.arange(8.0),
        names=("t",),
        coords={"t": {"spacing": 0.3, "origin": -1.0}},
    )
    assert x.sel(t=-0.7, mode="ceil").item() == 1.0  # exact tick (index 1)
    assert x.sel(t=-0.7, mode="floor").item() == 1.0
    assert x.sel(t=-0.7).item() == 1.0  # bare (exact, tolerance=0) sel too


def test_sel_compact_exact_tick_survives_a_large_origin_spacing_ratio():
    # `(target - origin) / spacing` is a *cancellation* error that scales
    # with |origin/spacing|, not a fixed few ULPs -- an epsilon-tolerance
    # guard sized for the latter (as a first version of this fix was) goes
    # wrong at a realistic ratio like an epoch-seconds axis with
    # millisecond spacing (~1.7e12). The target here IS exactly tick 5.
    x = XTensor(
        torch.arange(20.0),
        names=("t",),
        coords={"t": {"spacing": 0.001, "origin": 1.7e9}},
    )
    target = 1700000000.005
    assert x.sel(t=target, mode="ceil").item() == 5.0
    assert x.sel(t=target, mode="floor").item() == 5.0
    assert x.sel(t=target).item() == 5.0


def test_sel_compact_round_tie_break_is_not_biased_by_a_rounding_guard():
    # a target a hair above an exact midpoint must round to the *farther*
    # tick, not get pulled to the nearer-by-a-hair one by an overly
    # aggressive epsilon guard on the tie-break itself.
    x = XTensor(
        torch.arange(10.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    assert x.sel(t=3.5 + 1e-10, mode="round").item() == 4.0
    y = XTensor(
        torch.arange(8.0),
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.7}},
    )
    assert y.sel(t=2.2, mode="round").item() == 2.0


def test_sel_compact_degenerate_spacing_next_and_prev_are_not_inverted():
    # every tick sits at `origin` (spacing == 0); the search-based path
    # treats this as ascending (`(diffs >= 0).all()`), so prev -> floor,
    # next -> ceil -- both directions must agree with that, not swap.
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": {"spacing": 0.0, "origin": 5.0}},
    )
    assert x.sel(t=-1.0, mode="next").item() == 0.0  # ceil: 5 >= -1
    with pytest.raises(ValueError, match="no next tick"):
        x.sel(t=11.0, mode="next")  # ceil: 5 >= 11 is false
    assert x.sel(t=11.0, mode="prev").item() == 0.0  # floor: 5 <= 11
    with pytest.raises(ValueError, match="no prev tick"):
        x.sel(t=-1.0, mode="prev")  # floor: 5 <= -1 is false


def test_sel_compact_size_one_prev_next_matches_the_explicit_convention():
    # a single-tick coordinate has no direction of its own -- match
    # `_numeric_select`'s explicit-coordinate default (ascending) rather
    # than trusting a declared negative spacing that has nothing to order.
    x = XTensor(
        torch.arange(1.0),
        names=("t",),
        coords={"t": {"spacing": -7.0, "origin": 0.0}},
    )
    assert x.sel(t=3.5, mode="prev").item() == 0.0  # ascending: floor(3.5)=0
    with pytest.raises(ValueError, match="no prev tick"):
        x.sel(t=-3.5, mode="prev")


def test_sel_compact_infinite_target_resolves_like_the_search_path():
    x = XTensor(
        torch.arange(10.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    assert x.sel(t=float("inf"), mode="floor").item() == 9.0  # clamps to last
    assert x.sel(t=float("-inf"), mode="ceil").item() == 0.0  # clamps to first
    with pytest.raises(ValueError, match="no floor tick"):
        x.sel(t=float("-inf"), mode="floor")
    with pytest.raises(ValueError, match="no ceil tick"):
        x.sel(t=float("inf"), mode="ceil")


def test_sel_compact_nan_target_raises_a_clear_error():
    x = XTensor(
        torch.arange(10.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    with pytest.raises(ValueError, match="not a number"):
        x.sel(t=float("nan"), mode="round")


def test_sel_compact_empty_coordinate_raises_cleanly():
    x = XTensor(
        torch.zeros(0),
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.0}},
    )
    with pytest.raises(ValueError, match="no round tick"):
        x.sel(t=1.0)


def test_sel_compact_matches_the_search_path_exhaustively():
    # independent randomized comparison against `_pick_sel_index` (the
    # search-based reference) across many spacing/origin/target/mode
    # combinations, both directions -- the closed-form path must be exact,
    # not just "close" (this is the property that broke on PR #115's first
    # version, see the two tests above for the specific counterexamples).
    import random

    from fiery.xtensor._selection import (
        _closed_form_sel_index,
        _ClosedFormMiss,
        _pick_sel_index,
    )

    rng = random.Random(0)
    modes = ["round", "floor", "ceil", "prev", "next"]
    for _ in range(3000):
        size = rng.randint(1, 12)
        step = rng.choice([1, -1]) * round(rng.uniform(0.01, 5.0), 4)
        base = round(rng.uniform(-20, 20), 4)
        mode = rng.choice(modes)
        choice = rng.random()
        if choice < 0.3:
            k = rng.randint(0, size - 1)
            target = base + k * step
        elif choice < 0.7:
            frac = rng.uniform(-0.5, size - 0.5)
            target = base + frac * step
        else:
            target = base + step * rng.uniform(-5, size + 5)
        ascending = True if size <= 1 else step > 0
        values = torch.tensor(
            [base + i * step for i in range(size)], dtype=torch.float64
        )
        expected = _pick_sel_index(values, target, mode, ascending)
        try:
            got = _closed_form_sel_index(
                base, step, target, mode, ascending, size
            )
        except _ClosedFormMiss:
            continue  # the rare fallback path; see the tests below
        assert got == expected, (size, step, base, mode, target, expected, got)


def test_sel_compact_closed_form_miss_falls_back_correctly():
    # a ratio extreme enough to actually exhaust the walk's step budget --
    # confirms _numeric_select_compact's _ClosedFormMiss fallback (not just
    # _closed_form_sel_index in isolation) produces the right answer, on a
    # target inside the coordinate's range where every mode needs a walk.
    x = XTensor(
        torch.arange(300.0),
        names=("t",),
        coords={"t": {"spacing": 1e-9, "origin": 1.7e9}},
    )
    target = 1.7e9  # the very first tick -- deep inside the walk's territory
    for mode in ("round", "floor", "ceil", "prev", "next"):
        assert x.sel(t=target, mode=mode).item() == 0.0


def test_sel_compact_closed_form_miss_fallback_is_not_precision_starved():
    # the fallback must materialise at its OWN float64 precision, not go
    # through Coordinate["value"] (which computes in the tensor's default,
    # float32, dtype) and upcast afterwards -- upcasting after the fact
    # cannot recover precision already lost, and in this exact regime that
    # silently turned a real tick into "no tick exists".
    x = XTensor(
        torch.arange(300.0),
        names=("t",),
        coords={"t": {"spacing": 1e-9, "origin": 1.7e9}},
    )
    target = 1.7e9 + 120e-9  # the 120th tick, well inside the range
    assert x.sel(t=target, mode="ceil").item() == 120.0
    assert x.sel(t=target, mode="round").item() == 120.0


def test_sel_compact_closed_form_miss_forced_matches_the_search_path(
    monkeypatch,
):
    # force the fallback on an *ordinary* coordinate (via a monkeypatched
    # zero step budget) and confirm it still matches the search-based
    # reference for every mode -- the fallback path itself is new code, not
    # just a call-through to already-tested logic.
    import fiery.xtensor._selection as tensors_mod

    monkeypatch.setattr(tensors_mod, "_CLOSED_FORM_MAX_STEPS", 0)
    x = XTensor(
        torch.arange(10.0),
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.0}},
    )
    for mode in ("round", "floor", "ceil", "prev", "next"):
        for target in (3.4, 3.5, 3.6, -1.0, 15.0):
            try:
                got = x.sel(t=target, mode=mode).item()
            except ValueError:
                got = None
            values = torch.arange(10.0)
            ascending = True
            expected_idx = tensors_mod._pick_sel_index(
                values, target, mode, ascending
            )
            expected = (
                None if expected_idx is None else values[expected_idx].item()
            )
            assert got == expected, (mode, target, got, expected)


def test_sel_mode_and_method_are_exclusive_and_validated():
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.0}},
    )
    with pytest.raises(ValueError, match="either 'mode' or 'method'"):
        x.sel(t=1.0, mode="round", method="nearest")
    with pytest.raises(ValueError, match="unknown mode"):
        x.sel(t=1.0, mode="bogus")


def test_sel_numeric_is_unit_aware():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.arange(4.0),  # positions 0,1,2,3 mm
            names=("x",),
            coords={"x": {"spacing": (1.0, "mm")}},
        )
        assert x.sel(x="2000um", method="nearest").item() == 2.0  # converted
        assert x.sel(x=(2, "mm")).item() == 2.0  # a (value, unit) tuple


def test_sel_string_selector_splits_value_and_unit_without_a_backend():
    # a unitful string like "0.5s" must not silently discard its magnitude
    # when no unit backend is active -- it used to parse to (1, "0.5s").
    x = XTensor(
        torch.tensor([10.0, 20.0, 30.0, 40.0]),
        names=("t",),
        coords={"t": XTensor([0.0, 0.5, 1.0, 1.5], units="s")},
    )
    assert x.sel(t="0s").item() == 10.0
    assert x.sel(t="0.5s").item() == 20.0
    assert x.sel(t="1s").item() == 30.0


def test_compact_spacing_string_splits_value_and_unit_without_a_backend():
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": "2mm"}}
    )
    spacing = x.coords["t"]["spacing"]
    assert spacing["value"] == 2.0
    assert spacing["unit"] == "mm"


def test_bare_unit_string_without_a_leading_number_still_defaults_to_one():
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": "mm"}}
    )
    spacing = x.coords["t"]["spacing"]
    assert spacing["value"] == 1
    assert spacing["unit"] == "mm"


def test_sel_explicit_numeric_coordinate():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": XTensor(torch.tensor([0.0, 0.5, 2.0, 4.0]), units="s")},
    )
    assert x.sel(t=2.0).item() == 2.0
    assert x.sel(t=1.0, method="nearest").item() == 1.0  # nearest tick is 0.5


# ----------------------------------------------------------------------
# .sel value-range selection (issue #109)
# ----------------------------------------------------------------------


def test_sel_range_on_a_compact_ascending_coordinate():
    # ticks 0, 0.5, 1.0, 1.5, 2.0, 2.5 ; data 0..5
    x = XTensor(
        torch.arange(6.0),
        names=("t",),
        coords={"t": {"spacing": 0.5, "origin": 0.0}},
    )
    assert x.sel(t=slice(1.0, 2.5)).tolist() == [2.0, 3.0, 4.0]  # [1.0, 2.5)
    assert x.sel(t=slice(None, 1.0)).tolist() == [0.0, 1.0]  # value < 1.0
    assert x.sel(t=slice(1.5, None)).tolist() == [
        3.0,
        4.0,
        5.0,
    ]  # value >= 1.5
    assert x.sel(t=slice(None, None)).tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
    ]


def test_sel_range_on_a_compact_descending_coordinate():
    # ticks 8, 6, 4, 2, 0 ; data 0..4
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": -2.0, "origin": 8.0}},
    )
    assert x.sel(t=slice(0, 8)).tolist() == [1.0, 2.0, 3.0, 4.0]  # value < 8
    assert x.sel(t=slice(8, 0)).tolist() == [
        1.0,
        2.0,
        3.0,
        4.0,
    ]  # order-independent
    assert x.sel(t=slice(4, None)).tolist() == [0.0, 1.0, 2.0]  # value >= 4
    assert x.sel(t=slice(None, 4)).tolist() == [3.0, 4.0]  # value < 4


def test_sel_range_on_an_explicit_ascending_coordinate():
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": XTensor([0.0, 0.5, 2.0, 4.0, 10.0], units="s")},
    )
    assert x.sel(t=slice("1s", "5s")).tolist() == [
        2.0,
        3.0,
    ]  # [1, 5) -> 2., 4.
    assert x.sel(t=slice(None, "2s")).tolist() == [0.0, 1.0]
    assert x.sel(t=slice("4s", None)).tolist() == [3.0, 4.0]


def test_sel_range_on_an_explicit_descending_coordinate():
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": XTensor([10.0, 4.0, 2.0, 0.5, 0.0], units="s")},
    )
    assert x.sel(t=slice("1s", "5s")).tolist() == [1.0, 2.0]  # values 4., 2.


def test_sel_range_empty_result_is_a_well_formed_empty_axis():
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.0}},
    )
    assert x.sel(t=slice(10, 20)).tolist() == []  # entirely out of range
    assert x.sel(t=slice(2, 2)).tolist() == []  # a degenerate (empty) range

    y = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": XTensor([0.0, 1.0, 2.0, 3.0])},
    )
    assert y.sel(t=slice(10, 20)).tolist() == []


def test_sel_range_rejects_a_step():
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.0}},
    )
    with pytest.raises(ValueError, match="does not take a step"):
        x.sel(t=slice(0, 4, 2))


def test_sel_range_needs_a_monotonic_explicit_coordinate():
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": XTensor([0.0, 2.0, 1.0, 4.0, 3.0])},
    )
    with pytest.raises(ValueError, match="monotonic"):
        x.sel(t=slice(0, 3))


def test_sel_range_allows_a_repeated_tick_on_an_explicit_coordinate():
    # only-non-decreasing (a repeated tick) is not a *reversal*, and the
    # matching set is still contiguous -- same monotonicity requirement
    # `_numeric_select`'s mode="prev"/"next" already uses (>=0/<=0), not a
    # *strict* requirement `.interp` needs (division by v1 - v0).
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": XTensor([0.0, 1.0, 1.0, 2.0])},
    )
    assert x.sel(t=slice(0.5, 2.0)).tolist() == [1.0, 2.0]


def test_sel_range_on_an_int_coordinate_keeps_fractional_bounds():
    # a bound must not be silently truncated to the coordinate's own
    # (integer) dtype when the needle is built for `searchsorted` -- 10.5
    # becoming 10 would wrongly include tick 10.
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": XTensor(torch.tensor([0, 10, 20, 30, 40]))},
    )
    assert x.sel(t=slice(10.5, 30.5)).tolist() == [2.0, 3.0]
    assert x.sel(t=slice(9.9, 30.1)).tolist() == [1.0, 2.0, 3.0]


def test_sel_range_on_a_large_int_coordinate_does_not_lose_precision():
    # promoting an integer coordinate for the search must use float64, not
    # the tensor default (float32) -- an int64 epoch-timestamp coordinate
    # holds values well past float32's 2**24 exact-integer limit, where
    # float32 would collapse distinct ticks together.
    epoch = torch.tensor(
        [1700000000, 1700000001, 1700000002, 1700000003], dtype=torch.int64
    )
    x = XTensor(torch.arange(4.0), names=("t",), coords={"t": XTensor(epoch)})
    assert x.sel(t=slice(1700000001, 1700000003)).tolist() == [1.0, 2.0]
    assert x.sel(t=slice(1700000002, None)).tolist() == [2.0, 3.0]


def test_sel_range_handles_infinite_bounds_on_both_coordinate_kinds():
    # slice(-inf, inf) is an idiomatic slice(None, None); a compact
    # coordinate must resolve it exactly like an explicit one, not crash.
    c = XTensor(
        torch.arange(10.0), names=("t",), coords={"t": {"spacing": 0.1}}
    )
    assert c.sel(t=slice(1.0, float("inf"))).tolist() == []
    assert c.sel(t=slice(float("-inf"), 0.3)).tolist() == [0.0, 1.0, 2.0]

    e = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": XTensor([0.0, 1.0, 2.0, 3.0])},
    )
    assert e.sel(t=slice(float("-inf"), float("inf"))).tolist() == [
        0.0,
        1.0,
        2.0,
        3.0,
    ]


def test_sel_range_nan_bound_raises_on_both_coordinate_kinds():
    c = XTensor(
        torch.arange(10.0), names=("t",), coords={"t": {"spacing": 0.1}}
    )
    with pytest.raises(ValueError, match="NaN"):
        c.sel(t=slice(float("nan"), 1.0))
    e = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": XTensor([0.0, 1.0, 2.0, 3.0])},
    )
    with pytest.raises(ValueError, match="NaN"):
        e.sel(t=slice(float("nan"), 1.0))


def test_sel_range_compact_never_materialises(monkeypatch):
    # #109's range selection shares #110's O(1) property for a compact
    # coordinate -- it must not materialise the whole grid just to find
    # unit/size.
    def boom(self):
        raise AssertionError("materialise() called -- O(1) fast path bypassed")

    monkeypatch.setattr(Coordinate, "_materialise", boom)
    x = XTensor(
        torch.arange(1_000_000.0),
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.0}},
    )
    assert x.sel(t=slice(500_000.0, 500_002.0)).tolist() == [
        500000.0,
        500001.0,
    ]


def test_sel_range_survives_a_large_origin_spacing_ratio():
    # the same cancellation-error hazard fixed for point-selection (#110)
    # applies to range bounds too, since they share the same closed-form
    # primitives -- an epoch-seconds axis with millisecond spacing.
    x = XTensor(
        torch.arange(20.0),
        names=("t",),
        coords={"t": {"spacing": 0.001, "origin": 1.7e9}},
    )
    result = x.sel(t=slice(1700000000.005, 1700000000.010))
    assert result.tolist() == [5.0, 6.0, 7.0, 8.0, 9.0]


def test_sel_range_is_half_open_at_exact_tick_boundaries():
    # ticks 0,1,2,3,4 ; slice(1, 3) should include 1 but exclude 3
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.0}},
    )
    assert x.sel(t=slice(1, 3)).tolist() == [1.0, 2.0]


def test_sel_range_matches_a_boolean_mask_reference_exhaustively():
    # exhaustive-ish check across compact/explicit, ascending/descending,
    # against an independent boolean-mask reference implementation.
    import random

    rng = random.Random(0)
    for _ in range(500):
        kind = rng.choice(
            ["compact_asc", "compact_desc", "explicit_asc", "explicit_desc"]
        )
        n = rng.randint(2, 8)
        if kind == "compact_asc":
            step = round(rng.uniform(0.1, 3.0), 3)
            origin = round(rng.uniform(-5, 5), 3)
            values = torch.tensor([origin + i * step for i in range(n)])
            coords = {"t": {"spacing": step, "origin": origin}}
        elif kind == "compact_desc":
            step = -round(rng.uniform(0.1, 3.0), 3)
            origin = round(rng.uniform(-5, 5), 3)
            values = torch.tensor([origin + i * step for i in range(n)])
            coords = {"t": {"spacing": step, "origin": origin}}
        elif kind == "explicit_asc":
            vals = sorted({round(rng.uniform(-10, 10), 3) for _ in range(n)})
            values = torch.tensor(vals)
            coords = {"t": XTensor(values)}
        else:
            vals = sorted(
                {round(rng.uniform(-10, 10), 3) for _ in range(n)},
                reverse=True,
            )
            values = torch.tensor(vals)
            coords = {"t": XTensor(values)}
        size = values.numel()
        data = torch.arange(size, dtype=torch.float32)
        x = XTensor(data, names=("t",), coords=coords)

        lo = rng.choice([None, round(rng.uniform(-12, 12), 3)])
        hi = rng.choice([None, round(rng.uniform(-12, 12), 3)])
        selector = rng.choice([slice(lo, hi), slice(hi, lo)])

        got = x.sel(t=selector).tolist()

        start, stop = selector.start, selector.stop
        if start is not None and stop is not None:
            elo, ehi = (start, stop) if start <= stop else (stop, start)
        else:
            elo, ehi = start, stop
        mask = torch.ones(size, dtype=torch.bool)
        if elo is not None:
            mask &= values >= elo
        if ehi is not None:
            mask &= values < ehi
        expected = data[mask].tolist()
        assert got == expected, (kind, values.tolist(), lo, hi, selector)


def test_interp_nearest_is_builtin_without_the_backend(monkeypatch):
    # order-0 (nearest) needs no fiery.interpol; force the backend absent.
    from fiery.xtensor import _tensors

    monkeypatch.setattr(_tensors, "_interpol", lambda: None)
    x = XTensor(  # ticks 0,2,4,6,8
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    got = x.interp(t=[3.0, 5.0], method="nearest")
    assert got.tolist() == [2.0, 2.0]  # round(1.5)=2, round(2.5)=2 (half-even)
    assert got.coords["t"]["value"].as_subclass(torch.Tensor).tolist() == [
        3.0,
        5.0,
    ]
    # out-of-range clamps (replicate, the default) ...
    assert x.interp(t=[-4.0, 20.0], method="nearest").tolist() == [0.0, 4.0]
    # ... or wraps with bound="wrap"
    assert x.interp(t=10.0, method="nearest", bound="wrap").item() == 0.0


def test_interp_accepts_indexers_as_a_dict():
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.arange(5.0), names=("t",), coords={"t": {"spacing": 2.0}}
    )
    assert x.interp({"t": 3.0}).item() == 1.5


def test_interp_dict_escape_hatch_reaches_a_dim_named_like_a_keyword():
    # a dim literally named "method" can never be spelled as a keyword
    # argument -- interp(method=...) always binds interp's own parameter,
    # never the indexers -- so the dict form is the only way to reach it.
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.arange(5.0),
        names=("method",),
        coords={"method": {"spacing": 2.0}},
    )
    assert x.interp({"method": 3.0}).item() == 1.5


def test_interp_indexers_dict_and_kwargs_together_raises():
    x = XTensor(
        torch.arange(5.0), names=("t",), coords={"t": {"spacing": 2.0}}
    )
    with pytest.raises(ValueError, match="dict OR as keyword arguments"):
        x.interp({"t": 3.0}, t=3.0)


def test_interp_too_many_positional_indexers_raises():
    x = XTensor(
        torch.arange(5.0), names=("t",), coords={"t": {"spacing": 2.0}}
    )
    with pytest.raises(TypeError, match="at most one positional argument"):
        x.interp({"t": 2.0}, {"t": 3.0})


def test_interp_indexers_is_captured_positionally_not_by_keyword():
    # the escape hatch mustn't introduce the exact collision it fixes: a
    # dim literally named "indexers" still has to work as an ordinary
    # keyword argument, since the positional mapping is captured via
    # *args, never a named `indexers=` parameter.
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.arange(5.0),
        names=("indexers",),
        coords={"indexers": {"spacing": 2.0}},
    )
    assert x.interp(indexers=2.0).item() == 1.0


def test_interp_higher_order_needs_the_backend(monkeypatch):
    from fiery.xtensor import _tensors

    monkeypatch.setattr(_tensors, "_interpol", lambda: None)
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    with pytest.raises(ImportError, match="fiery-xtensor\\[interp\\]"):
        x.interp(t=[1.0], method="linear")


def test_interp_linear_computes_new_values():
    pytest.importorskip("fiery.interpol")
    x = XTensor(  # value == index (ticks 0,2,4,6,8)
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    got = x.interp(t=[1.0, 3.0, 5.0])  # halfway ticks -> half-index values
    assert got.tolist() == [0.5, 1.5, 2.5]
    assert got.names == ("t",)
    assert got.coords["t"]["value"].as_subclass(torch.Tensor).tolist() == [
        1.0,
        3.0,
        5.0,
    ]


def test_interp_scalar_drops_the_axis():
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    got = x.interp(t=3.0)  # a scalar query drops the axis, like sel
    assert got.ndim == 0
    assert got.item() == 1.5


def test_interp_keeps_other_axes_and_names():
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.arange(10.0).reshape(2, 5),
        names=("b", "t"),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    got = x.interp(t=[1.0, 3.0])
    assert got.shape == (2, 2)
    assert got.names == ("b", "t")
    assert got.tolist() == [[0.5, 1.5], [5.5, 6.5]]


def test_interp_empty_query_returns_an_empty_axis():
    # used to crash with an opaque internal reshape error (#96); should
    # behave like any other empty selection (e.g. `x[[]]`).
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    got = x.interp(t=[])
    assert got.shape == (0,)
    assert got.names == ("t",)
    assert got.coords["t"]["value"].tolist() == []

    got_tensor_query = x.interp(t=torch.tensor([]))
    assert got_tensor_query.shape == (0,)

    # another axis is untouched
    y = XTensor(
        torch.arange(8.0).reshape(2, 4),
        names=("b", "t"),
        coords={"t": {"spacing": 1.0}},
    )
    got_multi = y.interp(t=[])
    assert got_multi.shape == (2, 0)
    assert got_multi.names == ("b", "t")

    # an irregular coordinate too, both without and with the backend
    z = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 4.0, 9.0])},
    )
    assert z.interp(t=[], method="nearest").shape == (0,)
    pytest.importorskip("fiery.interpol")
    assert z.interp(t=[], method="linear").shape == (0,)


def test_interp_is_unit_aware():
    pytest.importorskip("fiery.interpol")
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.arange(5.0),
            names=("t",),
            coords={"t": {"spacing": (2.0, "s"), "origin": (0.0, "s")}},
        )
        assert x.interp(t="3000ms").item() == 1.5  # 3000 ms -> 3 s -> 1.5


def test_interp_query_gradients_flow():
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    q = torch.tensor([3.0], requires_grad=True)
    x.interp(t=q).sum().backward()
    assert q.grad.tolist() == [0.5]  # d(value)/d(query) = 1 / spacing


def test_interp_bound_option_and_override():
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 0.0}},
    )
    with set_options(interp_bound="wrap"):
        assert x.interp(t=10.0, method="nearest").item() == 0.0  # wraps
        # a per-call bound overrides the global option
        got = x.interp(t=10.0, method="nearest", bound="replicate")
        assert got.item() == 4.0  # clamps


def test_interp_irregular_nearest_is_builtin_without_the_backend(monkeypatch):
    # order-0 (nearest) needs no fiery.interpol, same as the regular case.
    from fiery.xtensor import _tensors

    monkeypatch.setattr(_tensors, "_interpol", lambda: None)
    x = XTensor(  # value == index**2, ticks 0, 1, 4, 9 (irregular)
        torch.tensor([0.0, 1.0, 4.0, 9.0]),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 4.0, 9.0])},
    )
    got = x.interp(t=[2.5, 8.0], method="nearest")
    assert got.tolist() == [4.0, 9.0]  # nearest tick to 2.5 is 4, to 8 is 9
    assert got.coords["t"]["value"].as_subclass(torch.Tensor).tolist() == [
        2.5,
        8.0,
    ]


def test_interp_irregular_linear_computes_new_values():
    pytest.importorskip("fiery.interpol")
    # data 0, 10, 20, 30 at irregular ticks 0, 1, 4, 9 -- deliberately *not*
    # the identity map, so a bracket-and-interpolate bug can't hide behind
    # "the query happened to equal the answer".
    x = XTensor(
        torch.tensor([0.0, 10.0, 20.0, 30.0]),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 4.0, 9.0])},
    )
    # 0.5 is halfway between ticks 0 and 1 (positions 0 and 1) -> data 0..10
    # 2.5 is 1.5/3 of the way between ticks 1 and 4 (positions 1 and 2)
    got = x.interp(t=[0.5, 2.5])
    assert got.tolist() == pytest.approx([5.0, 15.0])
    assert got.coords["t"]["value"].as_subclass(torch.Tensor).tolist() == [
        0.5,
        2.5,
    ]
    # exact match on a tick is exact, not just "close"
    assert x.interp(t=4.0).item() == pytest.approx(20.0)
    # out-of-range clamps to the edge value (the "replicate" default bound,
    # same as the regular-coordinate case) -- the fractional index itself
    # still extrapolates past the end segment's slope (frac -1 and 3.2 for
    # these queries), it's `bound` that then clamps it into range.
    got_extrap = x.interp(t=[-1.0, 10.0])
    assert got_extrap.tolist() == pytest.approx([0.0, 30.0])


def test_interp_irregular_descending_coordinate():
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.tensor([0.0, 10.0, 20.0, 30.0]),
        names=("t",),
        coords={"t": torch.tensor([9.0, 4.0, 1.0, 0.0])},  # descending
    )
    # 2.5 is 1.5/3 of the way between ticks 4 (index 1) and 1 (index 2)
    assert x.interp(t=2.5).item() == pytest.approx(15.0)
    assert x.interp(t=9.0).item() == pytest.approx(0.0)
    assert x.interp(t=0.0).item() == pytest.approx(30.0)


def test_interp_irregular_coordinate_must_be_monotonic():
    x = XTensor(
        torch.zeros(4),
        names=("t",),
        coords={"t": torch.tensor([0.0, 2.0, 1.0, 3.0])},
    )
    with pytest.raises(ValueError, match="strictly monotonic"):
        x.interp(t=1.5)


def test_interp_irregular_higher_order_is_not_planned():
    # a uniform-index-space spline basis isn't a true non-uniform spline in
    # value space -- structurally unsupported, not a pending TODO; see #81.
    x = XTensor(
        torch.tensor([0.0, 1.0, 4.0, 9.0]),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 4.0, 9.0])},
    )
    with pytest.raises(NotImplementedError, match="#81"):
        x.interp(t=2.5, method="cubic")


def test_interp_irregular_gradients_flow_through_query_and_values():
    pytest.importorskip("fiery.interpol")
    values = torch.tensor([0.0, 1.0, 4.0, 9.0], requires_grad=True)
    x = XTensor(
        torch.tensor([0.0, 1.0, 4.0, 9.0]), names=("t",), coords={"t": values}
    )
    q = torch.tensor(2.5, requires_grad=True)
    x.interp(t=q).backward()
    assert q.grad is not None and q.grad.item() != 0.0
    assert values.grad is not None
    # only the bracketing ticks (indices 1 and 2, ticks 1 and 4) get a
    # nonzero gradient -- the query never touches ticks 0 or 3
    assert values.grad[0].item() == 0.0
    assert values.grad[3].item() == 0.0
    assert values.grad[1].item() != 0.0
    assert values.grad[2].item() != 0.0


def test_interp_irregular_gradcheck_against_finite_differences():
    # "gradients flow" is not the same as "gradients are right": check the
    # analytic gradient of the whole value -> frac -> pull chain numerically.
    pytest.importorskip("fiery.interpol")
    data = torch.tensor([3.0, -1.0, 7.0, 2.0, 11.0], dtype=torch.float64)

    def pull(vals, query):
        x = XTensor(data, names=("t",), coords={"t": vals})
        return x.interp(t=query).as_subclass(torch.Tensor)

    ticks = torch.tensor(
        [0.0, 1.0, 4.0, 9.0, 25.0], dtype=torch.float64, requires_grad=True
    )
    # in-range brackets only (a query sitting exactly on a tick is a kink of
    # the piecewise-linear map, where a one-sided derivative is expected)
    q = torch.tensor([0.5, 2.5, 12.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(pull, (ticks, q), eps=1e-6, atol=1e-6)
    # ... and the same for a descending coordinate (the flipped search path)
    down = torch.tensor(
        [25.0, 9.0, 4.0, 1.0, 0.0], dtype=torch.float64, requires_grad=True
    )
    assert torch.autograd.gradcheck(pull, (down, q), eps=1e-6, atol=1e-6)


def test_interp_irregular_matches_the_regular_path_when_ticks_are_uniform():
    # an explicit coordinate that *happens* to be uniform must agree with the
    # compact one describing the same ticks -- bit for bit, for every bound.
    pytest.importorskip("fiery.interpol")
    data = torch.tensor([3.0, -1.0, 7.0, 2.0, 11.0])
    query = [-3.0, 0.25, 1.7, 3.9, 4.5, 11.0]  # ticks are 1, 3, 5, 7, 9
    explicit = XTensor(
        data, names=("t",), coords={"t": torch.arange(5.0) * 2.0 + 1.0}
    )
    compact = XTensor(
        data, names=("t",), coords={"t": {"spacing": 2.0, "origin": 1.0}}
    )
    for method in ("nearest", "linear"):
        got = explicit.interp(t=query, method=method)
        want = compact.interp(t=query, method=method)
        assert got.tolist() == want.tolist()
    for bound in ("replicate", "zero", "dft"):
        got = explicit.interp(t=query, bound=bound)
        want = compact.interp(t=query, bound=bound)
        assert got.tolist() == want.tolist()


def test_interp_irregular_matches_piecewise_linear_over_a_batch():
    # one query per bracket plus both out-of-range ends, all in one call --
    # the brackets differ across the batch, so a non-vectorised (or
    # first-bracket-for-everyone) inversion would show up here.
    pytest.importorskip("fiery.interpol")
    ticks = [0.0, 1.0, 4.0, 9.0, 25.0]
    data = [3.0, -1.0, 7.0, 2.0, 11.0]
    x = XTensor(
        torch.tensor(data), names=("t",), coords={"t": torch.tensor(ticks)}
    )
    query = [0.5, 2.5, 6.5, 17.0]
    expected = [1.0, 3.0, 4.5, 6.5]  # hand-computed segment midpoints
    assert x.interp(t=query).tolist() == pytest.approx(expected)
    # every tick reproduces its own value exactly (not just "close")
    assert x.interp(t=ticks).tolist() == data
    # both ends clamp under the default "replicate" bound
    assert x.interp(t=[-100.0, 100.0]).tolist() == pytest.approx([3.0, 11.0])


def test_interp_irregular_descending_out_of_range_both_ends():
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.tensor([11.0, 2.0, 7.0, -1.0, 3.0]),
        names=("t",),
        coords={"t": torch.tensor([25.0, 9.0, 4.0, 1.0, 0.0])},
    )
    # past the *high* end is index 0, past the *low* end is index -1
    got = x.interp(t=[-100.0, 0.5, 17.0, 100.0])
    assert got.tolist() == pytest.approx([3.0, 1.0, 6.5, 11.0])


def test_interp_irregular_two_point_coordinate():
    pytest.importorskip("fiery.interpol")
    x = XTensor(  # the smallest coordinate an inversion can bracket
        torch.tensor([1.0, 5.0]),
        names=("t",),
        coords={"t": torch.tensor([0.0, 4.0])},
    )
    got = x.interp(t=[-1.0, 0.0, 2.0, 4.0, 6.0])
    assert got.tolist() == pytest.approx([1.0, 1.0, 3.0, 5.0, 5.0])


def test_interp_irregular_is_unit_aware():
    pytest.importorskip("fiery.interpol")
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.tensor([0.0, 10.0, 20.0, 30.0]),
            names=("t",),
            coords={
                "t": XTensor(torch.tensor([0.0, 1.0, 4.0, 9.0]), units="s")
            },
        )
        assert x.interp(t="2500ms").item() == pytest.approx(15.0)
        got = x.interp(t=["500ms", "2500ms"])
        assert got.tolist() == pytest.approx([5.0, 15.0])
        # the new coordinate carries the *position* unit, not the query's
        coord = got.coords["t"]["value"]
        assert coord.units == "second"
        assert coord.as_subclass(torch.Tensor).tolist() == [0.5, 2.5]


def test_interp_irregular_higher_order_needs_the_backend(monkeypatch):
    # the backend guard is the same one the regular path uses: order >= 1
    # still needs fiery.interpol on an irregular coordinate.
    from fiery.xtensor import _tensors

    monkeypatch.setattr(_tensors, "_interpol", lambda: None)
    x = XTensor(
        torch.tensor([0.0, 1.0, 4.0, 9.0]),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 4.0, 9.0])},
    )
    with pytest.raises(ImportError, match="fiery-xtensor\\[interp\\]"):
        x.interp(t=[2.5], method="linear")


def test_interp_irregular_repeated_ticks_name_the_offending_pair():
    # a repeated tick is a zero-width cell: there is no inverse, and it is
    # easy to hit by accident (a float32 cumsum of many small steps), so the
    # error says *where* rather than just "not monotonic".
    x = XTensor(
        torch.zeros(4),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 1.0, 2.0])},
    )
    with pytest.raises(ValueError, match="ticks 1 and 2 are 1.0 and 1.0"):
        x.interp(t=1.5)


def test_interp_irregular_needs_at_least_two_ticks():
    x = XTensor(
        torch.tensor([5.0]), names=("t",), coords={"t": torch.tensor([2.0])}
    )
    with pytest.raises(ValueError, match="at least 2 points"):
        x.interp(t=2.0)


def test_interp_irregular_needs_a_1d_coordinate():
    # a 2-D (or higher) coordinate tensor is rejected at construction (#97,
    # see test_explicit_coordinate_must_be_1d) rather than being silently
    # storable and only surfacing as an opaque searchsorted shape error
    # inside `.interp` itself.
    with pytest.raises(ValueError, match="must be 1-D"):
        XTensor(
            torch.arange(6.0).reshape(3, 2),
            names=("t", "u"),
            coords={"t": torch.tensor([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])},
        )


def test_interp_irregular_on_a_sliced_coordinate():
    # `x[::2]` leaves the coordinate a strided view; the inversion must still
    # bracket against the *sliced* ticks.
    pytest.importorskip("fiery.interpol")
    x = XTensor(
        torch.tensor([3.0, -1.0, 7.0, 2.0, 11.0]),
        names=("t",),
        coords={"t": torch.tensor([0.0, 1.0, 4.0, 9.0, 25.0])},
    )
    sliced = x[::2]  # ticks 0, 4, 25 / data 3, 7, 11
    assert sliced.coords["t"]["value"].as_subclass(torch.Tensor).tolist() == [
        0.0,
        4.0,
        25.0,
    ]
    assert sliced.interp(t=2.0).item() == pytest.approx(5.0)


def test_interp_needs_a_numeric_coordinate():
    x = _labelled()
    with pytest.raises(ValueError, match="no numeric coordinate"):
        x.interp(col=1.0)


def test_sel_selects_a_labelled_position_and_drops_the_axis():
    x = _labelled()
    y = x.sel(col="y")
    assert y.shape == (3,)
    assert y.names == ("row",)
    assert torch.equal(y, x.as_subclass(torch.Tensor)[:, 2])


def test_sel_with_a_list_of_labels_keeps_the_axis():
    x = _labelled()
    y = x.sel(col=["w", "y"])
    assert y.shape == (3, 2)
    assert y.coords == {"col": ("w", "y")}


def test_sel_unknown_label_or_dim_raises():
    x = _labelled()
    with pytest.raises(ValueError, match="no label 'nope'"):
        x.sel(col="nope")
    with pytest.raises(ValueError, match="no coordinates"):
        x.sel(row="anything")


def test_isel_selects_by_integer_position_along_a_named_dim():
    x = _labelled()
    assert x.isel(col=slice(1, 3)).coords == {"col": ("x", "y")}
    assert x.isel(row=0).names == ("col",)


def test_attribute_access_selects_a_single_label():
    x = _labelled()
    out = x.x  # label "x" on dim "col"
    assert out.shape == (3,)
    assert torch.equal(out, x.as_subclass(torch.Tensor)[:, 1])


def test_attribute_access_can_be_chained_across_dims():
    x = XTensor(
        torch.arange(6).reshape(2, 3),
        names=("row", "col"),
        coords={"row": ("r0", "r1"), "col": ("c0", "c1", "c2")},
    )
    out = x.r1.c2
    assert out.ndim == 0
    assert out.item() == 5


def test_attribute_access_unknown_label_raises():
    with pytest.raises(AttributeError):
        _ = _labelled().nope


def test_getitem_slices_the_labels_of_kept_axes():
    x = _labelled()
    y = x[:, 1:3]
    assert y.coords == {"col": ("x", "y")}
    # an integer index drops the axis and its labels
    assert x[:, 1].coords == {}


def test_getitem_resolves_a_positional_label_against_its_axis():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("b", "row", "col"),
        coords={"row": ("r0", "r1", "r2"), "col": ("w", "x", "y", "z")},
    )
    # a bare label indexes the axis it sits on (like an int there): drops it
    out = x[:, "r1", "y"]
    assert out.names == ("b",)
    assert torch.equal(out, x[:, 1, 2])


def test_getitem_labels_address_trailing_axes_through_ellipsis():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("b", "row", "col"),
        coords={"row": ("r0", "r1", "r2"), "col": ("w", "x", "y", "z")},
    )
    # ellipsis fills the leading axes; the labels address the last two
    assert torch.equal(x[..., "r2", "z"], x[:, 2, 3])
    assert x[..., "r2", "z"].names == ("b",)
    # a label mixes with ints and slices
    assert torch.equal(x[0, "r1", :], x[0, 1, :])


def test_getitem_list_of_labels_is_an_advanced_index_keeping_the_axis():
    x = _labelled()  # names ("row", "col"), coords col=("w","x","y","z")
    out = x[:, ["w", "y"]]
    assert out.names == ("row", "col")
    assert out.coords == {"col": ("w", "y")}


def test_getitem_top_level_label_tuple_is_one_label_per_axis():
    x = XTensor(
        torch.arange(6).reshape(2, 3),
        names=("row", "col"),
        coords={"row": ("r0", "r1"), "col": ("c0", "c1", "c2")},
    )
    # x["r1", "c2"] -> label per axis, not a single advanced index
    assert x["r1", "c2"].item() == 5


def test_getitem_label_on_an_unlabelled_or_missing_axis_raises():
    x = _labelled()  # only "col" is labelled
    with pytest.raises(KeyError, match="no coordinates"):
        _ = x["w"]  # axis 0 ("row") has no coordinates
    with pytest.raises(KeyError, match="no label"):
        _ = x[:, "nope"]


# ----------------------------------------------------------------------
# structured coordinates (dict-valued labels + field queries)
# ----------------------------------------------------------------------


def _channels():
    return XTensor(
        torch.arange(6).reshape(3, 2).float(),
        names=("c", "x"),
        coords={
            "c": [
                {"name": "DAPI", "type": "nucleus"},
                {"name": "GFP", "type": "signal"},
                {"name": "RFP", "type": "signal"},
            ]
        },
    )


def test_structured_labels_select_by_name_everywhere():
    img = _channels()
    expected = img.as_subclass(torch.Tensor)[1]
    assert torch.equal(img.sel(c="GFP"), expected)  # keyword
    assert torch.equal(img.GFP, expected)  # attribute
    assert torch.equal(img["GFP"], expected)  # positional string
    assert img.sel(c="GFP").names == ("x",)  # a single name drops the axis


def test_structured_query_selects_contiguous_matches_as_a_slice():
    img = _channels()
    sig = img[{"type": "signal"}]  # positions 1,2 -> a slice, keeps the axis
    assert sig.names == ("c", "x")
    assert sig.shape == (2, 2)
    assert [label["name"] for label in sig.coords["c"]] == ["GFP", "RFP"]
    # the sel keyword spelling is equivalent
    assert torch.equal(img.sel(c={"type": "signal"}), sig)


def test_structured_query_selects_non_contiguous_matches_as_a_list():
    img = XTensor(
        torch.arange(6).reshape(3, 2).float(),
        names=("c", "x"),
        coords={
            "c": [
                {"name": "A", "type": "signal"},
                {"name": "B", "type": "nucleus"},
                {"name": "C", "type": "signal"},
            ]
        },
    )
    out = img[{"type": "signal"}]  # positions 0,2 -> advanced index
    assert [label["name"] for label in out.coords["c"]] == ["A", "C"]
    assert out.shape == (2, 2)


def test_structured_query_composes_positionally_and_can_be_empty():
    vol = XTensor(
        torch.zeros(4, 3, 2),
        names=("z", "c", "x"),
        coords={
            "c": [
                {"name": "A", "type": "s"},
                {"name": "B", "type": "s"},
                {"name": "C", "type": "t"},
            ]
        },
    )
    # a query addresses the axis it sits on, mixing with an int index
    assert vol[0, {"type": "s"}].names == ("c", "x")
    assert vol[0, {"type": "s"}].shape == (2, 2)
    # a query matching nothing gives a size-0 axis (no error)
    assert vol[:, {"type": "nope"}].shape == (4, 0, 2)


def test_permute_carries_coordinates_unchanged():
    x = _labelled()
    assert x.T.coords == {"col": ("w", "x", "y", "z")}
    assert x.transpose(0, 1).coords == {"col": ("w", "x", "y", "z")}


def test_coords_getter_hides_stale_metadata():
    # A shape-changing op without coordinate handling must not report labels
    # of the wrong length; the getter guards this.
    x = _labelled()
    y = x.clone()
    y._coords = {"col": ("w", "x")}  # wrong length for size-4 axis
    assert y.coords == {}


# ----------------------------------------------------------------------
# renaming
# ----------------------------------------------------------------------


def test_rename_out_of_place_sets_and_clears_names():
    x = XTensor(torch.zeros(2, 3), names=("row", "col"))
    y = x.rename("a", "b")
    assert y.names == ("a", "b")
    assert x.names == ("row", "col")  # out-of-place: x unchanged
    assert y.rename(None).names == (None, None)
    plain = torch.Tensor
    assert torch.equal(y.as_subclass(plain), x.as_subclass(plain))


def test_rename_by_map():
    x = XTensor(torch.zeros(2, 3), names=("row", "col"))
    assert x.rename(col="C").names == ("row", "C")
    with pytest.raises(ValueError, match="no axis named"):
        x.rename(nope="X")


def test_rename_in_place_returns_self():
    x = XTensor(torch.zeros(2, 3), names=("row", "col"))
    out = x.rename_("a", "b")
    assert out is x
    assert x.names == ("a", "b")


def test_rename_moves_coordinates_to_the_new_dim_name():
    x = _labelled()
    # positional rename sets the whole name tuple; the labelled axis keeps its
    # name here, so the coordinates stay under "col"
    y = x.rename("batch", "col")
    assert y.coords == {"col": ("w", "x", "y", "z")}
    # renaming the labelled axis moves its coordinates to the new name
    z = x.rename(col="feature")
    assert z.coords == {"feature": ("w", "x", "y", "z")}


def test_swap_dims_promotes_a_label_and_demotes_the_old_index():
    # "time" is a bare numeric tuple -- auto-promoted to a numeric
    # coordinate (#107), not a plain label tuple.
    x = XTensor(
        torch.arange(6.0),
        names=("time",),
        coords={
            "time": (0.0, 0.5, 1.0, 1.5, 2.0, 2.5),
            "label": ("time", ("a", "b", "c", "d", "e", "f")),
        },
    )
    y = x.swap_dims({"time": "label"})
    assert y.names == ("label",)
    assert set(y.coords) == {"time", "label"}
    assert y.coords["time"]["value"].as_subclass(torch.Tensor).tolist() == [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
        2.5,
    ]
    assert y.coords["label"] == ("a", "b", "c", "d", "e", "f")
    assert y.sel(label="c").item() == 2.0
    with pytest.raises(ValueError, match="not an index"):
        y.sel(time=1.0)


def test_swap_dims_by_keyword():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={
            "t": (10.0, 20.0, 30.0, 40.0),
            "label": ("t", ("a", "b", "c", "d")),
        },
    )
    assert x.swap_dims(t="label").names == ("label",)


def test_swap_dims_demotes_a_compact_coordinate_and_it_still_reslices():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={
            "t": {"spacing": 1.0, "origin": 0.0},
            "label": ("t", ("a", "b", "c", "d")),
        },
    )
    y = x.swap_dims({"t": "label"})
    assert y.names == ("label",)
    # the demoted compact coordinate keeps its old key and rides the renamed
    # axis, and still re-slices exactly (same machinery as a multi-dim affine
    # coordinate, generalised to a single dim).
    values = y[:2].coords["t"]["value"].as_subclass(torch.Tensor)
    assert values.tolist() == [0.0, 1.0]


def test_swap_dims_axis_descriptor_follows_the_renamed_axis():
    x = XTensor(
        torch.arange(4.0),
        axes=[{"name": "t", "type": "time", "unit": "s"}],
        coords={
            "t": (0.0, 1.0, 2.0, 3.0),
            "label": ("t", ("a", "b", "c", "d")),
        },
    )
    y = x.swap_dims({"t": "label"})
    assert y.axes == ({"name": "label", "type": "time", "unit": "s"},)


def test_swap_dims_rejects_a_nonexistent_dim():
    x = XTensor(torch.zeros(3), names=("t",))
    with pytest.raises(ValueError, match="no axis named"):
        x.swap_dims({"nope": "label"})


def test_swap_dims_rejects_a_target_that_is_not_a_coordinate():
    x = XTensor(torch.zeros(3), names=("t",))
    with pytest.raises(ValueError, match="must be an existing"):
        x.swap_dims({"t": "label"})


def test_swap_dims_rejects_a_multi_dim_coordinate_as_target():
    x = XTensor(
        torch.zeros(3, 4),
        names=("y", "x"),
        coords={"lat": (("y", "x"), {"spacing": ([1.0, 0.5], "mm")})},
    )
    with pytest.raises(ValueError, match="must be an existing"):
        x.swap_dims({"y": "lat"})


def test_swap_dims_rejects_a_new_name_colliding_with_an_axis():
    x = XTensor(
        torch.zeros(3, 4),
        names=("y", "x"),
        coords={"y2": ("y", ("a", "b", "c"))},
    )
    with pytest.raises(ValueError, match="already an axis name"):
        x.swap_dims({"y": "x"})


def test_swap_dims_in_place_returns_self():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={
            "t": (0.0, 1.0, 2.0, 3.0),
            "label": ("t", ("a", "b", "c", "d")),
        },
    )
    out = x.swap_dims_({"t": "label"})
    assert out is x
    assert x.names == ("label",)


def test_swap_dims_empty_mapping_is_a_noop():
    x = XTensor(torch.zeros(3), names=("t",))
    assert x.swap_dims({}) is x
    assert x.swap_dims() is x


# ----------------------------------------------------------------------
# builtin named-tensor API, re-implemented self-managed
# ----------------------------------------------------------------------


def test_refine_names_only_names_unnamed_axes():
    x = XTensor(torch.zeros(2, 3, 4), names=(None, "b", None))
    assert x.refine_names("a", "b", "c").names == ("a", "b", "c")
    assert x.refine_names(None, "b", None).names == (None, "b", None)


def test_refine_names_ellipsis_keeps_spanned_names():
    x = XTensor(torch.zeros(2, 3, 4, 5), names=(None, "b", "c", None))
    assert x.refine_names("a", ..., "d").names == ("a", "b", "c", "d")


def test_names_ellipsis_fills_the_middle_with_unnamed_axes():
    # a single `...` in `names=` stands for a run of unnamed (None) axes
    t = torch.zeros(2, 3, 4, 5)
    assert XTensor(t, names=("b", ..., "x")).names == ("b", None, None, "x")
    assert XTensor(t, names=(..., "x")).names == (None, None, None, "x")
    assert XTensor(t, names=("b", ...)).names == ("b", None, None, None)
    assert XTensor(t, names=(...,)).names == (None,) * 4


def test_names_setter_accepts_ellipsis():
    x = XTensor(torch.zeros(2, 3, 4))
    x.names = ("b", ..., "w")
    assert x.names == ("b", None, "w")


def test_names_ellipsis_with_a_descriptor_dict():
    x = XTensor(
        torch.zeros(2, 3, 4),
        axes=[{"name": "b", "type": "batch"}, ..., "w"],
    )
    assert x.names == ("b", None, "w")
    assert x.axes[0] == {"name": "b", "type": "batch"}


def test_names_ellipsis_rejects_more_than_one_and_overflow():
    t = torch.zeros(2, 3)
    with pytest.raises(ValueError, match="only one '...'"):
        XTensor(t, names=("a", ..., "b", ...))
    with pytest.raises(ValueError, match="too many names"):
        XTensor(t, names=("a", "b", "c", ...))


def test_rename_ellipsis_keeps_the_spanned_names():
    # `rename` modifies, so `...` leaves the spanned axes unchanged
    x = XTensor(
        torch.zeros(2, 3, 4, 5),
        names=("b", "c", "h", "w"),
        coords={"w": ("a", "b", "c", "d", "e")},
    )
    renamed = x.rename("B", ..., "W")
    assert renamed.names == ("B", "c", "h", "W")
    assert renamed.coords == {"W": ("a", "b", "c", "d", "e")}


def test_permute_ellipsis_stands_for_the_remaining_axes():
    x = XTensor(
        torch.zeros(2, 3, 4, 5),
        names=("b", "c", "h", "w"),
        coords={"w": ("a", "b", "c", "d", "e")},
    )
    assert x.permute("w", ...).names == ("w", "b", "c", "h")
    assert x.permute(..., "b").names == ("c", "h", "w", "b")
    assert x.permute("w", ...).coords == {"w": ("a", "b", "c", "d", "e")}


def test_refine_names_rejects_renaming_a_named_axis():
    x = XTensor(torch.zeros(2, 3), names=("a", "b"))
    with pytest.raises(ValueError, match="cannot rename"):
        x.refine_names("a", "X")


def test_align_to_permutes_into_the_given_order():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    y = x.align_to("c", "a", "b")
    assert y.names == ("c", "a", "b")
    assert y.shape == (4, 2, 3)
    assert x.align_to(..., "a").names == ("b", "c", "a")


def test_align_as_reorders_and_inserts_size_one_axes():
    x = XTensor(torch.zeros(4, 2), names=("c", "a"))
    other = XTensor(torch.zeros(2, 3, 4, 5), names=("a", "b", "c", "d"))
    y = x.align_as(other)
    assert y.names == ("a", "b", "c", "d")
    assert y.shape == (2, 1, 4, 1)


def test_align_as_requires_every_axis_present_in_target():
    other = XTensor(torch.zeros(2, 3), names=("a", "b"))
    with pytest.raises(ValueError, match="not in the target"):
        XTensor(torch.zeros(2), names=("q",)).align_as(other)


@pytest.mark.skipif(
    not hasattr(torch.Tensor, "names"),
    reason="builtin named-tensor API not present in this torch build",
)
def test_names_do_not_use_builtin_named_tensors():
    # The self-managed names must not set the underlying tensor's builtin
    # (C-level) names -- that is what keeps us portable across torch versions.
    x = XTensor(torch.zeros(2, 3), names=("row", "col"))
    assert torch.Tensor.names.__get__(x) == (None, None)


def test_torch_func_returns_none_for_missing_op():
    assert _torch_func("permute") is not None
    assert _torch_func("this_op_does_not_exist_anywhere") is None


# ----------------------------------------------------------------------
# functional-form (torch.op(x, ...)) metadata parity
# ----------------------------------------------------------------------


@pytest.mark.skipif(
    not hasattr(torch, "permute"),
    reason="torch.permute (functional) was added in torch 1.9",
)
def test_functional_permute_matches_method_form():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert torch.permute(x, (2, 0, 1)).names == x.permute(2, 0, 1).names
    assert torch.permute(x, (2, 0, 1)).names == ("c", "a", "b")


def test_functional_unsqueeze_squeeze_match_method_form():
    x = XTensor(torch.zeros(2, 3), names=("row", "col"))
    assert torch.unsqueeze(x, 1).names == ("row", None, "col")
    y = XTensor(torch.zeros(1, 3), names=("s", "col"))
    assert torch.squeeze(y).names == ("col",)


@pytest.mark.skipif(
    not hasattr(torch, "permute"),
    reason="torch.permute (functional) was added in torch 1.9",
)
def test_functional_permute_carries_coordinates():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    y = torch.permute(x, (2, 0, 1))
    assert y.names == ("c", "a", "b")
    assert y.coords == {"c": ("w", "x", "y", "z")}


# ----------------------------------------------------------------------
# reshape / reorder op family (axis names)
# ----------------------------------------------------------------------


def test_transpose_family_swaps_axis_names():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.transpose(0, 2).names == ("c", "b", "a")
    assert torch.transpose(x, 0, 2).names == ("c", "b", "a")


@pytest.mark.skipif(
    not _HAS_SWAPAXES, reason="torch.swapaxes/swapdims added in torch 1.8"
)
def test_swapaxes_family_swaps_axis_names():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.swapaxes(0, 1).names == ("b", "a", "c")
    assert x.swapdims(1, 2).names == ("a", "c", "b")


def test_mT_transposes_last_two_axis_names():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.mT.names == ("a", "c", "b")
    assert x.mT.shape == (2, 4, 3)


def test_movedim_reorders_axis_names_like_torch():
    x = XTensor(torch.zeros(2, 3, 4, 5), names=("a", "b", "c", "d"))
    y = x.movedim(0, 2)
    assert y.names == ("b", "c", "a", "d")
    assert y.shape == tuple(torch.movedim(torch.zeros(2, 3, 4, 5), 0, 2).shape)


@pytest.mark.skipif(
    not _HAS_MOVEAXIS, reason="torch.moveaxis added in torch 1.8"
)
def test_moveaxis_reorders_axis_names_like_torch():
    x = XTensor(torch.zeros(2, 3, 4, 5), names=("a", "b", "c", "d"))
    assert x.moveaxis((0, 1), (2, 3)).names == ("c", "d", "a", "b")


def test_reshape_uses_same_name_rule_as_view():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("b", "h", "w"))
    assert x.reshape(2, 12).names == ("b", None)
    assert x.reshape(6, 4).names == (None, "w")
    assert torch.reshape(x, (2, -1)).names == ("b", None)


# ----------------------------------------------------------------------
# name-as-dim: a name may stand in for an integer `dim=`
# ----------------------------------------------------------------------


def test_permute_accepts_axis_names():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    y = x.permute("c", "a", "b")
    assert y.names == ("c", "a", "b")
    assert y.shape == (4, 2, 3)
    assert y.shape == x.permute(2, 0, 1).shape


def test_transpose_family_accepts_axis_names():
    # Name-as-dim is a method-form feature: the functional form
    # (`torch.transpose(x, "a", ...)`) cannot be relied on because newer
    # PyTorch rejects a non-int dim at the C dispatcher before
    # `__torch_function__` runs.
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.transpose("a", "c").names == ("c", "b", "a")


@pytest.mark.skipif(
    not _HAS_SWAPAXES, reason="torch.swapaxes/swapdims added in torch 1.8"
)
def test_swapaxes_family_accepts_axis_names():
    x = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    assert x.swapaxes("a", "b").names == ("b", "a", "c")
    assert x.swapdims("b", "c").names == ("a", "c", "b")


def test_movedim_accepts_axis_names_for_source():
    x = XTensor(torch.zeros(2, 3, 4, 5), names=("a", "b", "c", "d"))
    assert x.movedim("a", 2).names == ("b", "c", "a", "d")


@pytest.mark.skipif(
    not _HAS_MOVEAXIS, reason="torch.moveaxis added in torch 1.8"
)
def test_moveaxis_accepts_axis_names_for_source():
    x = XTensor(torch.zeros(2, 3, 4, 5), names=("a", "b", "c", "d"))
    assert x.moveaxis(("a", "b"), (2, 3)).names == ("c", "d", "a", "b")


def test_squeeze_accepts_axis_name():
    x = XTensor(torch.ones(2, 1, 3), names=("a", "one", "b"))
    y = x.squeeze("one")
    assert y.names == ("a", "b")
    assert y.shape == (2, 3)


def test_index_select_accepts_axis_name_and_slices_coords():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("batch", "feat", "chan"),
        coords={"chan": ("w", "x", "y", "z")},
    )
    by_name = x.index_select("chan", torch.tensor([0, 2]))
    by_int = x.index_select(2, torch.tensor([0, 2]))
    assert by_name.shape == by_int.shape == (2, 3, 2)
    assert by_name.coords == by_int.coords == {"chan": ("w", "y")}


def test_index_select_on_a_compact_numeric_coordinate():
    # #162: `index_select`'s coordinate handling used to hand the axis'
    # `Coordinate` straight to `_slice_labels`, which integer-indexes it
    # *positionally* -- but a numeric `Coordinate` is a dict-like mapping
    # keyed by strings ("value"/"spacing"/"origin"), so that hit
    # `Coordinate.__getitem__` and KeyErrored looking up the *key* `0`
    # instead of *position* `0` (the same failure shape as #85).
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.0}},
    )
    idx = torch.tensor([0, 2])
    out = x.index_select("t", idx)
    assert out.tolist() == [0.0, 2.0]
    assert out.coords["t"]["value"].tolist() == [0.0, 2.0]


def test_index_select_functional_form_matches_method_form_for_numeric_coord():
    # both `x.index_select(...)` and `torch.index_select(x, ...)` must go
    # through the same fix -- they share one override body (see #160/#161).
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 10.0}},
    )
    idx = torch.tensor([0, 2])
    method = x.index_select("t", idx)
    functional = torch.index_select(x, 0, idx)
    assert method.tolist() == functional.tolist() == [0.0, 2.0]
    expected = [10.0, 14.0]
    assert method.coords["t"]["value"].tolist() == expected
    assert functional.coords["t"]["value"].tolist() == expected


def test_index_select_on_an_explicit_numeric_coordinate():
    # the explicit/irregular form (`{"value": <tensor>}`) is a meaningfully
    # different branch of `_slice_coordinate` than the compact affine form
    # above -- both must work.
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": torch.tensor([10.0, 20.0, 30.0, 40.0, 50.0])},
    )
    out = x.index_select("t", torch.tensor([3, 0, 1]))
    assert out.tolist() == [3.0, 0.0, 1.0]
    assert out.coords["t"]["value"].tolist() == [40.0, 10.0, 20.0]


def test_index_select_on_a_label_coordinate_still_works():
    # non-regression: a label coordinate (tuple of strings) must keep going
    # through `_slice_labels` exactly as before the #162 fix.
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": ("a", "b", "c", "d", "e")},
    )
    out = x.index_select("t", torch.tensor([0, 2, 4]))
    assert out.tolist() == [0.0, 2.0, 4.0]
    assert out.coords["t"] == ("a", "c", "e")


def test_index_select_with_a_0d_index_on_a_numeric_coordinate():
    # a 0-D index is a valid index_select argument (e.g. what argmax/argmin
    # return) -- the first version of the #162 fix mismatched a 0-D
    # coordinate value against the size-1 result, producing an object that
    # returned successfully but raised later on any access to .coords,
    # repr(), or arithmetic. Must behave like the equivalent 1-D index.
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 10.0}},
    )
    out = x.index_select("t", torch.tensor(2))
    assert out.tolist() == [2.0]
    assert out.coords["t"]["value"].tolist() == [14.0]
    assert repr(out)  # must not raise
    assert (out + 1).tolist() == [3.0]


@pytest.mark.skipif(
    not _INDEX_SELECT_ACCEPTS_INT32,
    reason="torch 1.7's index_select itself rejects an int32 index",
)
def test_index_select_with_an_int32_index_on_a_numeric_coordinate():
    # torch documents index_select's index as IntTensor or LongTensor; the
    # first version of the #162 fix only recognised torch.long and silently
    # dropped the coordinate for an int32 index instead of re-slicing it.
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": 2.0, "origin": 10.0}},
    )
    idx = torch.tensor([0, 2], dtype=torch.int32)
    out = x.index_select("t", idx)
    assert out.tolist() == [0.0, 2.0]
    assert out.coords["t"]["value"].tolist() == [10.0, 14.0]


def test_name_as_dim_unknown_name_raises():
    x = XTensor(torch.zeros(2, 3), names=("a", "b"))
    with pytest.raises(ValueError, match="no axis named 'z'"):
        x.transpose("a", "z")


# ----------------------------------------------------------------------
# reductions: drop / keep the reduced axis' name (and its coords)
# ----------------------------------------------------------------------


def test_sum_drops_reduced_axis_name():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.sum(dim="b").names == ("a", "c")
    assert x.sum(dim="b").shape == (2, 4)
    assert x.sum(dim=1).names == ("a", "c")  # int still works


def test_sum_keepdim_preserves_reduced_axis_name():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    y = x.sum(dim="b", keepdim=True)
    assert y.names == ("a", "b", "c")
    assert y.shape == (2, 1, 4)


def test_sum_keepdim_drops_the_reduced_axis_compact_coordinate():
    # the reduced axis's name survives `keepdim`, but the size-1 result no
    # longer describes any single one of the original positions -- a compact
    # coordinate has no length of its own to invalidate the way a label does,
    # so it must be dropped explicitly rather than rebinding to "position 0
    # of the original extent" (#90).
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": {"spacing": 1.0, "origin": 0.0}},
    )
    y = x.sum(dim="t", keepdim=True)
    assert y.names == ("t",)
    assert "t" not in y.coords


def test_sum_keepdim_drops_the_reduced_axis_label():
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": ("w", "x", "y", "z")}
    )
    y = x.sum(dim="t", keepdim=True)
    assert "t" not in y.coords


def test_sum_keepdim_keeps_an_unreduced_axis_coordinate():
    x = XTensor(
        torch.arange(12.0).reshape(3, 4),
        names=("y", "x"),
        coords={"y": {"spacing": 1.0, "origin": 0.0}},
    )
    y = x.sum(dim="x", keepdim=True)
    assert "y" in y.coords


def test_reduction_over_multiple_named_axes():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.mean(dim=("a", "c")).names == ("b",)


def test_reduce_all_yields_scalar_with_no_names():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    s = x.sum()
    assert s.names == ()
    assert s.ndim == 0


def test_functional_reduction_carries_names_like_method():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert torch.mean(x, 1).names == x.mean(dim=1).names == ("a", "c")


def test_argmax_and_amax_track_names():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.amax(dim="c").names == ("a", "b")
    assert x.argmax(dim="a").names == ("b", "c")


def test_negative_dim_reduction():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.sum(dim=-1).names == ("a", "b")


def test_reduction_drops_coordinates_of_the_reduced_axis():
    x = XTensor(
        torch.arange(24.0).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"b": ("p", "q", "r"), "c": ("w", "x", "y", "z")},
    )
    # reduce the axis carrying the "b" labels
    r = x.sum(dim="b")
    assert r.names == ("a", "c")
    assert r.coords == {"c": ("w", "x", "y", "z")}
    # reduce a plain axis: the other labels are unchanged
    assert x.sum(dim="a").coords == {
        "b": ("p", "q", "r"),
        "c": ("w", "x", "y", "z"),
    }


# ----------------------------------------------------------------------
# irregular / (values, indices) reducers
# ----------------------------------------------------------------------


def _named():
    return XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))


def test_std_var_reduce_and_accept_a_name():
    x = _named()
    assert x.std(dim="b").names == ("a", "c")
    assert x.var(dim="c").names == ("a", "b")
    # a positional bool is `unbiased`, not a dim -> reduces everything
    assert x.std(False).names == ()
    assert x.std(False).ndim == 0


def test_norm_reduces_the_named_dim():
    x = _named()
    assert x.norm(dim="b").names == ("a", "c")
    assert torch.norm(x, 2, 1).names == ("a", "c")


def test_max_min_return_named_values_and_indices():
    x = _named()
    out = x.max(dim="b")
    assert out.values.names == out.indices.names == ("a", "c")
    assert x.max().names == ()  # scalar form
    # the two-tensor (elementwise) form reconciles names, keeps rank
    assert torch.max(x, x * 2).names == ("a", "b", "c")


def test_median_mode_kthvalue_track_names():
    x = _named()
    assert torch.median(x).names == ()  # global median -> scalar
    assert torch.median(x, 1).values.names == ("a", "c")
    assert x.mode(dim="b").values.names == ("a", "c")
    assert x.kthvalue(2, dim="b").values.names == ("a", "c")


def test_sort_preserves_names_and_drops_the_sorted_coordinate():
    x = XTensor(
        torch.randn(2, 3, 4),
        names=("a", "b", "c"),
        coords={"b": ("p", "q", "r"), "c": ("w", "x", "y", "z")},
    )
    s = x.sort(dim="c")
    assert s.values.names == s.indices.names == ("a", "b", "c")
    # the sorted axis' labels no longer match positions -> dropped
    assert s.values.coords == {"b": ("p", "q", "r")}


def test_topk_keeps_names_resizes_and_drops_the_dim_coordinate():
    x = XTensor(
        torch.randn(2, 3, 4),
        names=("a", "b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    t = x.topk(2, dim="c")
    assert t.values.names == ("a", "b", "c")
    assert t.values.shape == (2, 3, 2)
    assert t.values.coords == {}


# ----------------------------------------------------------------------
# scans: cumsum / cumprod / softmax / log_softmax / logcumsumexp / cummax /
# cummin -- dimension-preserving (rank, sizes, names, coords all unchanged)
# ----------------------------------------------------------------------


def test_cumsum_keeps_names_and_accepts_a_name_dim():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    y = x.cumsum("a")
    assert isinstance(y, XTensor)
    assert y.names == ("a", "b", "c")
    assert y.shape == (2, 3, 4)
    assert y[1, 0, 0].item() == x[0, 0, 0].item() + x[1, 0, 0].item()
    assert x.cumsum(1).names == ("a", "b", "c")  # int still works


def test_functional_softmax_keeps_names():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    y = torch.softmax(x, 1)
    assert isinstance(y, XTensor)
    assert y.names == ("a", "b", "c")
    assert y.shape == (2, 3, 4)


def test_log_softmax_keeps_names():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.log_softmax("c").names == ("a", "b", "c")


@pytest.mark.skipif(
    not hasattr(torch, "logcumsumexp"),
    reason="logcumsumexp added in a newer torch",
)
def test_logcumsumexp_keeps_names():
    x = XTensor(torch.arange(24.0).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.logcumsumexp("b").names == ("a", "b", "c")


def test_cumsum_keeps_the_labelled_dims_coordinates():
    x = XTensor(
        torch.arange(24.0).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"b": ("p", "q", "r"), "c": ("w", "x", "y", "z")},
    )
    y = x.cumsum("b")
    assert y.names == ("a", "b", "c")
    assert y.coords == {"b": ("p", "q", "r"), "c": ("w", "x", "y", "z")}


@pytest.mark.skipif(
    not hasattr(torch, "cummax"), reason="cummax added in a newer torch"
)
def test_cummax_keeps_names_and_coords_on_both_members():
    x = XTensor(
        torch.arange(24.0).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    out = x.cummax("a")
    assert out.values.names == out.indices.names == ("a", "b", "c")
    assert out.values.shape == out.indices.shape == (2, 3, 4)
    assert out.values.coords == {"c": ("w", "x", "y", "z")}
    assert out.indices.coords == {"c": ("w", "x", "y", "z")}


# ----------------------------------------------------------------------
# slice / split ops: select / narrow / unbind / split / chunk / flip / roll
# ----------------------------------------------------------------------


def test_select_drops_axis_name():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.select("b", 0).names == ("a", "c")
    assert x.select("b", 0).shape == (2, 4)
    assert torch.select(x, 1, 0).names == ("a", "c")


def test_narrow_keeps_names_and_accepts_name_dim():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    y = x.narrow("c", 1, 2)
    assert y.names == ("a", "b", "c")
    assert y.shape == (2, 3, 2)


def test_unbind_returns_pieces_without_the_unbound_axis():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    pieces = x.unbind("a")
    assert len(pieces) == 2
    assert all(p.names == ("b", "c") for p in pieces)
    assert all(p.shape == (3, 4) for p in pieces)


def test_split_and_chunk_keep_all_names():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    parts = x.split(2, dim="c")
    assert [p.shape for p in parts] == [(2, 3, 2), (2, 3, 2)]
    assert all(p.names == ("a", "b", "c") for p in parts)
    chunks = x.chunk(2, dim="b")
    assert [p.shape for p in chunks] == [(2, 2, 4), (2, 1, 4)]
    assert all(p.names == ("a", "b", "c") for p in chunks)


def test_flip_and_roll_preserve_names():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.flip(("a", "c")).names == ("a", "b", "c")
    assert x.roll(1, dims="b").names == ("a", "b", "c")
    assert x.roll(2).names == ("a", "b", "c")  # flattened roll


def test_select_drops_the_coordinates_of_the_selected_axis():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"b": ("p", "q", "r"), "c": ("w", "x", "y", "z")},
    )
    r = x.select("b", 0)
    assert r.names == ("a", "c")
    assert r.coords == {"c": ("w", "x", "y", "z")}


def test_narrow_and_split_slice_coordinates():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    assert x.narrow("c", 1, 2).coords == {"c": ("x", "y")}
    parts = x.split(2, dim="c")
    assert [p.coords for p in parts] == [{"c": ("w", "x")}, {"c": ("y", "z")}]


def test_flip_reverses_coordinates_on_the_flipped_axis():
    x = XTensor(
        torch.arange(12).reshape(3, 4),
        names=("a", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    assert x.flip("c").coords == {"c": ("z", "y", "x", "w")}


def test_roll_rolls_coordinates_on_the_rolled_axis():
    x = XTensor(
        torch.arange(12).reshape(3, 4),
        names=("a", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    # a right shift of 1 moves each label one step forward (cyclically)
    assert x.roll(1, dims="c").coords == {"c": ("z", "w", "x", "y")}


def test_flip_reverses_a_compact_numeric_coordinate_and_stays_compact():
    # #85: reversed()/indexing a Coordinate like a plain dict silently
    # dropped it (flip) or crashed (roll); it must materialise/negate the
    # numeric *values* instead.
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    out = x.flip("t")
    assert out.tolist() == [3.0, 2.0, 1.0, 0.0]
    coord = out.coords["t"]
    assert coord._compact()  # negating spacing keeps it exact + compact
    assert coord["value"].tolist() == [3.0, 2.0, 1.0, 0.0]
    assert out.flip("t").coords["t"]["value"].tolist() == [0.0, 1.0, 2.0, 3.0]


def test_flip_reverses_an_explicit_numeric_coordinate():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": torch.tensor([10.0, 20.0, 30.0, 40.0])},
    )
    assert x.flip("t").coords["t"]["value"].tolist() == [
        40.0,
        30.0,
        20.0,
        10.0,
    ]


def test_roll_rolls_a_compact_numeric_coordinate():
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    # a right shift of 1: same cyclic convention as the label test above
    out = x.roll(1, dims="t")
    assert out.tolist() == [3.0, 0.0, 1.0, 2.0]
    assert out.coords["t"]["value"].tolist() == [3.0, 0.0, 1.0, 2.0]


def test_roll_rolls_an_explicit_numeric_coordinate():
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": torch.tensor([10.0, 20.0, 30.0, 40.0])},
    )
    assert x.roll(-1, dims="t").coords["t"]["value"].tolist() == [
        20.0,
        30.0,
        40.0,
        10.0,
    ]


def test_flip_of_a_compact_numeric_coordinate_keeps_gradients_flowing():
    spacing = torch.tensor(2.0, requires_grad=True)
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": spacing}}
    )
    x.flip("t").coords["t"]["value"].sum().backward()
    assert spacing.grad is not None


def test_roll_preserves_the_position_unit_of_a_compact_numeric_coordinate():
    # #165: `roll`'s wrap-around can't stay affine, so it materialises the
    # compact coordinate to an explicit array via `_slice_coordinate`'s
    # advanced-index branch -- that branch used to rely on the generic
    # `__torch_function__` metadata-carry fallback to propagate the position
    # unit onto the materialised array, but `_slice_coordinate` is always
    # called from inside an override running under `_no_dispatch()`, where
    # that fallback never fires -- so the unit silently came back `None`.
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": (2.0, "mm"), "origin": (10.0, "mm")}},
    )
    assert x.coords["t"]["spacing"].unit == "mm"
    out = torch.roll(x, 1, dims=0)
    assert out.coords["t"]["value"].tolist() == [18.0, 10.0, 12.0, 14.0, 16.0]
    assert out.coords["t"]["value"].units == "mm"


def test_index_select_preserves_the_unit_of_a_compact_numeric_coordinate():
    # #165: same root cause as the `roll` case above, for `index_select`'s
    # (#162-fixed) advanced-index re-slice of a compact coordinate.
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": (2.0, "mm"), "origin": (10.0, "mm")}},
    )
    out = x.index_select("t", torch.tensor([0, 2]))
    assert out.coords["t"]["value"].tolist() == [10.0, 14.0]
    assert out.coords["t"]["value"].units == "mm"


def test_flip_still_preserves_the_unit_of_a_compact_numeric_coordinate():
    # non-regression: flip's compact case never materialises (it negates
    # `spacing`/shifts `origin` and stays compact), so it never went through
    # the buggy path and must be completely unaffected by the #165 fix.
    x = XTensor(
        torch.arange(5.0),
        names=("t",),
        coords={"t": {"spacing": (2.0, "mm"), "origin": (10.0, "mm")}},
    )
    out = torch.flip(x, [0])
    assert out.coords["t"]["value"].tolist() == [18.0, 16.0, 14.0, 12.0, 10.0]
    assert out.coords["t"]["value"].units == "mm"
    assert out.coords["t"]._compact()  # unchanged: still the compact form


def test_flip_preserves_the_position_unit_of_an_explicit_numeric_coordinate():
    # #165 also affected flip when the coordinate is *already* explicit
    # (negative-step slicing isn't valid on a tensor, so flip re-slices an
    # explicit coordinate through the same advanced-index branch as roll /
    # index_select, rather than through the compact affine shortcut above).
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": {"value": XTensor([10.0, 20.0, 30.0, 40.0], units="mm")}},
    )
    out = x.flip("t")
    assert out.coords["t"]["value"].tolist() == [40.0, 30.0, 20.0, 10.0]
    assert out.coords["t"]["value"].units == "mm"


def test_narrow_and_split_preserve_the_unit_of_an_explicit_coordinate():
    # #165's basic-slice sibling: `narrow`/`split`/`chunk` re-slice an
    # *explicit* numeric coordinate through `_slice_coordinate`'s basic slice
    # branch, also called from inside a `_no_dispatch()` override, and so
    # also silently dropped the unit. `select` passes a plain int, which
    # drops the whole coordinate (there's nothing left to lose a unit from),
    # so it was never actually affected despite reaching the same function.
    x = XTensor(
        torch.arange(4.0),
        names=("t",),
        coords={"t": {"value": XTensor([10.0, 20.0, 30.0, 40.0], units="mm")}},
    )
    narrowed = x.narrow("t", 1, 2)
    assert narrowed.coords["t"]["value"].tolist() == [20.0, 30.0]
    assert narrowed.coords["t"]["value"].units == "mm"
    parts = x.split(2, dim="t")
    assert [p.coords["t"]["value"].units for p in parts] == ["mm", "mm"]


def test_roll_of_a_non_unitful_compact_numeric_coordinate_still_works():
    # non-regression: a coordinate with no unit at all must not gain one
    # (and must not error) after the #165 fix -- don't assume every
    # coordinate carries a unit.
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    assert x.coords["t"]["spacing"].unit == ""
    out = torch.roll(x, 1, dims=0)
    assert out.coords["t"]["value"].tolist() == [3.0, 0.0, 1.0, 2.0]
    assert out.coords["t"]["value"].units == ""


def test_index_select_of_a_non_unitful_compact_coordinate_still_works():
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    out = x.index_select("t", torch.tensor([0, 2]))
    assert out.coords["t"]["value"].tolist() == [0.0, 2.0]
    assert out.coords["t"]["value"].units == ""


# ----------------------------------------------------------------------
# reshape-family (rank-changing): flatten / unflatten / expand / diagonal
# ----------------------------------------------------------------------


def test_flatten_marks_merged_axis_unnamed():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    assert x.flatten(0, 1).names == (None, "c")
    assert x.flatten(0, 1).shape == (6, 4)
    assert x.flatten("a", "b").names == (None, "c")  # by name
    assert x.flatten(1, 1).names == ("a", "b", "c")  # no-op keeps names
    assert torch.flatten(x, 1, 2).names == ("a", None)  # functional form


def test_unflatten_marks_split_axes_unnamed():
    x = XTensor(torch.arange(24).reshape(6, 4), names=("a", "b"))
    y = x.unflatten("a", (2, 3))
    assert y.names == (None, None, "b")
    assert y.shape == (2, 3, 4)
    assert x.unflatten(0, (6,)).names == ("a", "b")  # single split = no-op


def test_expand_prepends_unnamed_axes():
    x = XTensor(torch.zeros(3, 4), names=("b", "c"))
    assert x.expand(2, 3, 4).names == (None, "b", "c")


@pytest.mark.skipif(
    not _HAS_BROADCAST_TO, reason="torch.broadcast_to added in torch 1.8"
)
def test_broadcast_to_prepends_unnamed_axes():
    x = XTensor(torch.zeros(3, 4), names=("b", "c"))
    assert torch.broadcast_to(x, (2, 3, 4)).names == (None, "b", "c")


def test_diagonal_drops_the_two_axes_and_appends_unnamed():
    x = XTensor(torch.zeros(3, 3, 4), names=("a", "b", "c"))
    y = x.diagonal(0, "a", "b")
    assert y.names == ("c", None)
    assert y.shape == (4, 3)


def test_flatten_drops_coordinates_in_the_merged_range():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"b": ("p", "q", "r"), "c": ("w", "x", "y", "z")},
    )
    assert x.flatten(1, 2).coords == {}  # both labelled axes merged away


def test_expand_keeps_coordinates_of_existing_axes():
    x = XTensor(
        torch.zeros(3, 4),
        names=("b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    out = x.expand(2, 3, 4)
    assert out.names == (None, "b", "c")
    assert out.coords == {"c": ("w", "x", "y", "z")}


def test_expand_drops_a_compact_coordinate_on_a_broadcast_axis():
    # a size-1 axis expanded to N is still only ever one position's worth of
    # underlying data -- a compact coordinate has no length of its own to
    # invalidate the way a label does, so it must be dropped explicitly (#90).
    x = XTensor(
        torch.zeros(1, 4),
        names=("y", "x"),
        coords={"y": {"spacing": 1.0, "origin": 0.0}},
    )
    assert "y" not in x.expand(3, 4).coords
    if _HAS_BROADCAST_TO:
        assert "y" not in torch.broadcast_to(x, (3, 4)).coords


def test_expand_drops_a_label_on_a_broadcast_axis():
    x = XTensor(torch.zeros(1, 4), names=("y", "x"), coords={"y": ("only",)})
    assert "y" not in x.expand(3, 4).coords


def test_expand_of_an_unrelated_axis_keeps_a_compact_coordinate():
    # sanity check: only the axis whose *size* actually changed is dropped.
    x = XTensor(
        torch.zeros(3, 1),
        names=("y", "x"),
        coords={"y": {"spacing": 1.0, "origin": 0.0}},
    )
    assert "y" in x.expand(3, 4).coords


def test_diagonal_keeps_the_surviving_axis_coordinates():
    x = XTensor(
        torch.zeros(3, 3, 4),
        names=("a", "b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    out = x.diagonal(0, 0, 1)
    assert out.names == ("c", None)
    assert out.coords == {"c": ("w", "x", "y", "z")}


# ----------------------------------------------------------------------
# combine ops: cat / stack (name reconciliation across operands)
# ----------------------------------------------------------------------


def test_cat_reconciles_axis_names():
    a = XTensor(torch.zeros(2, 3), names=("r", "c"))
    b = XTensor(torch.zeros(4, 3), names=("r", "c"))
    out = torch.cat([a, b], 0)
    assert out.names == ("r", "c")
    assert out.shape == (6, 3)


def test_cat_conflicting_names_become_unnamed():
    a = XTensor(torch.zeros(2, 3), names=("r", "c"))
    conflicting = XTensor(torch.zeros(2, 3), names=("r", "x"))
    assert torch.cat([a, conflicting], 0).names == ("r", None)


def test_stack_inserts_an_unnamed_axis():
    a = XTensor(torch.zeros(2, 3), names=("r", "c"))
    assert torch.stack([a, a], 0).names == (None, "r", "c")
    assert torch.stack([a, a], 1).names == ("r", None, "c")
    assert torch.stack([a, a], 0).shape == (2, 2, 3)


def test_cat_concatenates_coordinates_along_the_join_axis():
    a = XTensor(torch.zeros(2, 2), names=("r", "c"), coords={"c": ("p", "q")})
    b = XTensor(
        torch.zeros(2, 3), names=("r", "c"), coords={"c": ("x", "y", "z")}
    )
    out = torch.cat([a, b], 1)
    assert out.coords == {"c": ("p", "q", "x", "y", "z")}
    assert out.shape == (2, 5)


def test_cat_with_a_plain_operand_drops_coordinates():
    a = XTensor(torch.zeros(2, 2), names=("r", "c"), coords={"c": ("p", "q")})
    out = torch.cat([a, XTensor(torch.zeros(2, 2))], 1)
    assert out.coords == {}


def test_stack_keeps_the_agreed_coordinates():
    a = XTensor(torch.zeros(2, 2), names=("r", "c"), coords={"c": ("p", "q")})
    out = torch.stack([a, a], 0)
    assert out.names == (None, "r", "c")
    assert out.coords == {"c": ("p", "q")}
    assert out.shape == (2, 2, 2)


# ----------------------------------------------------------------------
# combine ops: hstack / vstack / dstack (conservative: names reconciled only
# when every operand already has the result's rank; coordinates always drop)
# ----------------------------------------------------------------------


def test_hstack_reconciles_names_when_ranks_already_align():
    a = XTensor(torch.zeros(2, 3), names=("r", "c"))
    b = XTensor(torch.zeros(2, 3), names=("r", "c"))
    out = torch.hstack([a, b])
    assert out.names == ("r", "c")
    assert out.coords == {}
    assert out.shape == (2, 6)


def test_vstack_reconciles_names_when_ranks_already_align():
    a = XTensor(torch.zeros(2, 3), names=("r", "c"))
    b = XTensor(torch.zeros(2, 3), names=("r", "c"))
    out = torch.vstack([a, b])
    assert out.names == ("r", "c")
    assert out.coords == {}
    assert out.shape == (4, 3)


def test_dstack_reconciles_names_when_ranks_already_align():
    a = XTensor(torch.zeros(2, 3, 5), names=("r", "c", "d"))
    b = XTensor(torch.zeros(2, 3, 5), names=("r", "c", "d"))
    out = torch.dstack([a, b])
    assert out.names == ("r", "c", "d")
    assert out.coords == {}
    assert out.shape == (2, 3, 10)


def test_vstack_drops_names_when_an_operand_is_promoted():
    # a plain 1-D vector is promoted to 2-D by `vstack`, shifting its axis
    # relative to the already-2-D operand -- positional alignment is not
    # trustworthy here, so the conservative result is fully unnamed.
    a = XTensor(torch.zeros(3), names=("c",))
    b = XTensor(torch.zeros(2, 3), names=("r", "c"))
    out = torch.vstack([a, b])
    assert out.names == (None, None)
    assert out.coords == {}
    assert out.shape == (3, 3)


def test_dstack_conflicting_names_become_unnamed():
    a = XTensor(torch.zeros(2, 3, 5), names=("r", "c", "d"))
    conflicting = XTensor(torch.zeros(2, 3, 5), names=("r", "c", "x"))
    out = torch.dstack([a, conflicting])
    assert out.names == ("r", "c", None)


# ----------------------------------------------------------------------
# matmul family: matmul / mm / bmm / @  (axis names)
# ----------------------------------------------------------------------


def test_matmul_2d_names_rows_from_a_cols_from_b():
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    assert torch.matmul(a, b).names == ("m", "n")
    assert torch.mm(a, b).names == ("m", "n")
    assert (a @ b).names == ("m", "n")  # `@` dispatches Tensor.matmul
    assert (a @ b).shape == (2, 4)


def test_matmul_vector_cases():
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    v = XTensor(torch.zeros(3), names=("k",))
    assert (v @ b).names == ("n",)
    assert (a @ v).names == ("m",)
    assert (v @ v).names == ()


def test_matmul_batches_broadcast_and_reconcile_names():
    a = XTensor(torch.zeros(5, 2, 3), names=("b", "m", "k"))
    b = XTensor(torch.zeros(5, 3, 4), names=("b", "k", "n"))
    assert torch.bmm(a, b).names == ("b", "m", "n")
    assert (a @ b).names == ("b", "m", "n")
    flat = XTensor(torch.zeros(3, 4), names=("k", "n"))
    assert (a @ flat).names == ("b", "m", "n")
    a2 = XTensor(torch.zeros(5, 2, 3), names=("x", "m", "k"))
    assert (a2 @ b).names == (None, "m", "n")


def test_matmul_with_one_plain_operand():
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    assert torch.matmul(torch.zeros(2, 3), b).names == (None, "n")
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    assert torch.matmul(a, torch.zeros(3, 4)).names == ("m", None)


def test_matmul_drops_coordinates():
    a = XTensor(
        torch.zeros(2, 3), names=("m", "k"), coords={"k": ("a", "b", "c")}
    )
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    out = a @ b
    assert out.coords == {}
    assert out.names == ("m", "n")


# ----------------------------------------------------------------------
# einsum / tensordot
# ----------------------------------------------------------------------


def test_einsum_explicit_equation_names_output_from_operands():
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    out = torch.einsum("ij,jk->ik", a, b)
    assert out.names == ("m", "n")
    assert out.shape == (2, 4)


def test_einsum_operands_as_a_single_list():
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    out = torch.einsum("ij,jk->ik", [a, b])
    assert out.names == ("m", "n")


def test_einsum_batched_equation():
    a = XTensor(torch.zeros(5, 2, 3), names=("b", "i", "j"))
    b = XTensor(torch.zeros(5, 3, 4), names=("b", "j", "k"))
    out = torch.einsum("bij,bjk->bik", a, b)
    assert out.names == ("b", "i", "k")
    assert out.shape == (5, 2, 4)


def test_einsum_summed_axis_is_dropped_from_the_output():
    a = XTensor(torch.zeros(2, 3), names=("row", "col"))
    out = torch.einsum("ij->i", a)
    assert out.names == ("row",)
    assert out.shape == (2,)


def test_einsum_implicit_output_matches_explicit():
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    assert torch.einsum("ij,jk", a, b).names == ("m", "n")


def test_einsum_ellipsis_falls_back_to_unnamed():
    a = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    b = XTensor(torch.zeros(2, 4, 5), names=("a", "c", "d"))
    out = torch.einsum("...bc,...cd->...bd", a, b)
    assert out.names == (None, None, None)
    assert out.shape == (2, 3, 5)


def test_einsum_with_a_plain_operand():
    a = XTensor(torch.zeros(2, 3), names=("m", "k"))
    out = torch.einsum("ij,jk->ik", a, torch.zeros(3, 4))
    assert out.names == ("m", None)


def test_einsum_drops_coordinates():
    a = XTensor(
        torch.zeros(2, 3), names=("m", "k"), coords={"k": ("x", "y", "z")}
    )
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    out = torch.einsum("ij,jk->ik", a, b)
    assert out.coords == {}


def test_tensordot_int_dims_contracts_trailing_leading_axes():
    a = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    b = XTensor(torch.zeros(3, 4, 5), names=("b", "c", "d"))
    out = torch.tensordot(a, b, dims=2)
    assert out.names == ("a", "d")
    assert out.shape == (2, 5)


def test_tensordot_dims_as_a_pair_of_axis_lists():
    a = XTensor(torch.zeros(2, 3, 4), names=("a", "b", "c"))
    b = XTensor(torch.zeros(3, 4, 5), names=("b", "c", "d"))
    out = torch.tensordot(a, b, dims=([1, 2], [0, 1]))
    assert out.names == ("a", "d")
    assert out.shape == (2, 5)


def test_tensordot_with_a_plain_operand():
    a = XTensor(torch.zeros(2, 3), names=("a", "k"))
    out = torch.tensordot(a, torch.zeros(3, 4), dims=1)
    assert out.names == ("a", None)
    b = XTensor(torch.zeros(3, 2), names=("k", "a2"))
    out2 = torch.tensordot(torch.zeros(2, 3), b, dims=1)
    assert out2.names == (None, "a2")


def test_tensordot_drops_coordinates():
    a = XTensor(
        torch.zeros(2, 3), names=("a", "k"), coords={"k": ("p", "q", "r")}
    )
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    out = torch.tensordot(a, b, dims=1)
    assert out.coords == {}
    assert out.names == ("a", "n")


def test_einsum_keeps_descriptors_of_surviving_axes_only():
    # a preserved axis keeps its descriptor (type/…); a contracted one's
    # descriptor is dropped with the axis — no leak onto the survivor.
    a = XTensor(
        torch.zeros(5, 2, 3),
        axes=[
            {"name": "b", "type": "batch"},
            "i",
            {"name": "j", "type": "k"},
        ],
    )
    b = XTensor(torch.zeros(5, 3, 4), names=("b", "j", "k"))
    out = torch.einsum("bij,bjk->bik", a, b)
    assert out.names == ("b", "i", "k")
    assert out.axes == (
        {"name": "b", "type": "batch"},
        {"name": "i"},
        {"name": "k"},
    )


def test_tensordot_keeps_descriptors_of_surviving_axes_only():
    a = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "a", "type": "space"}, {"name": "k", "type": "chan"}],
    )
    b = XTensor(torch.zeros(3, 4), names=("k", "n"))
    out = torch.tensordot(a, b, dims=1)
    assert out.names == ("a", "n")
    assert out.axes == ({"name": "a", "type": "space"}, {"name": "n"})


def test_matmul_keeps_each_operands_surviving_descriptor():
    # each result axis keeps the descriptor of the operand it came from; the
    # right operand's trailing axis used to lose it.
    a = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "m", "type": "space"}, {"name": "k", "type": "chan"}],
    )
    b = XTensor(
        torch.zeros(3, 4),
        axes=[{"name": "k", "type": "chan"}, {"name": "n", "type": "time"}],
    )
    out = a @ b
    assert out.names == ("m", "n")
    assert out.axes == (
        {"name": "m", "type": "space"},
        {"name": "n", "type": "time"},
    )


def test_cat_merges_descriptors_keeping_agreement_dropping_conflicts():
    a = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "r", "type": "space"}, {"name": "c", "type": "chan"}],
    )
    b = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "r", "type": "space"}, {"name": "c", "type": "time"}],
    )
    out = torch.cat([a, b], dim=0)
    # "r" agrees -> kept; "c" conflicts (chan vs time) -> field dropped
    assert out.axes == ({"name": "r", "type": "space"}, {"name": "c"})


def test_stack_keeps_existing_descriptors_new_axis_unnamed():
    a = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "r", "type": "space"}, {"name": "c", "type": "chan"}],
    )
    out = torch.stack([a, a], dim=0)
    assert out.axes == (
        None,
        {"name": "r", "type": "space"},
        {"name": "c", "type": "chan"},
    )


def test_combine_op_strict_policy_raises_on_conflicting_descriptor():
    a = XTensor(torch.zeros(2, 3), axes=["r", {"name": "c", "type": "chan"}])
    b = XTensor(torch.zeros(2, 3), axes=["r", {"name": "c", "type": "time"}])
    with set_options(combine_axes="strict"):
        with pytest.raises(ValueError, match="conflicting 'type'"):
            torch.cat([a, b], dim=0)


# ----------------------------------------------------------------------
# gather / scatter / where / masked_select
# ----------------------------------------------------------------------


def test_gather_keeps_axis_names_and_accepts_name_dim():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    idx = torch.zeros(2, 3, 4, dtype=torch.long)
    assert x.gather("c", idx).names == ("a", "b", "c")
    assert torch.gather(x, 2, idx).names == ("a", "b", "c")


@pytest.mark.skipif(
    not hasattr(torch, "take_along_dim"),
    reason="take_along_dim not in this torch",
)
def test_take_along_dim_keeps_names():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    idx = torch.zeros(2, 3, 1, dtype=torch.long)
    assert x.take_along_dim(idx, "c").names == ("a", "b", "c")


def test_scatter_preserves_names_and_coordinates():
    x = XTensor(
        torch.arange(24).reshape(2, 3, 4),
        names=("a", "b", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    idx = torch.zeros(2, 3, 4, dtype=torch.long)
    src = torch.zeros(2, 3, 4, dtype=x.dtype)
    out = x.scatter("c", idx, src)
    assert out.names == ("a", "b", "c")
    # positions/sizes are unchanged, so the labels survive
    assert out.coords == {"c": ("w", "x", "y", "z")}


def test_gather_drops_the_gathered_axis_coordinates():
    x = XTensor(
        torch.arange(12).reshape(3, 4),
        names=("a", "c"),
        coords={"c": ("w", "x", "y", "z")},
    )
    idx = torch.zeros(3, 4, dtype=torch.long)
    assert x.gather("c", idx).coords == {}


def _where_supports_scalar():
    try:
        torch.where(torch.tensor([True]), torch.tensor([1.0]), 0.0)
    except (RuntimeError, TypeError):
        return False
    return True


def test_where_reconciles_operand_names():
    p = XTensor(torch.zeros(2, 3), names=("r", "k"))
    q = XTensor(torch.ones(2, 3), names=("r", "k"))
    assert torch.where(p > 0.5, p, q).names == ("r", "k")
    conflicting = XTensor(torch.ones(2, 3), names=("r", "z"))
    assert torch.where(p > 0.5, p, conflicting).names == ("r", None)


@pytest.mark.skipif(
    not _where_supports_scalar(),
    reason="scalar `where` operand not supported in this torch",
)
def test_where_with_a_scalar_operand_keeps_the_tensor_names():
    p = XTensor(torch.zeros(2, 3), names=("r", "k"))
    assert torch.where(p > 0.5, p, 0.0).names == ("r", "k")


def test_masked_select_collapses_to_one_unnamed_axis():
    x = XTensor(torch.arange(24).reshape(2, 3, 4), names=("a", "b", "c"))
    out = torch.masked_select(x, x > 5)
    assert out.names == (None,)
    assert out.ndim == 1


def test_masked_fill_keeps_names_and_coordinates():
    x = XTensor(
        torch.arange(6.0).reshape(2, 3),
        names=("r", "c"),
        coords={"c": ("a", "b", "d")},
    )
    # shape-preserving, so names and coordinates ride through (both forms)
    out = x.masked_fill(x > 2, 0.0)
    assert out.names == ("r", "c")
    assert out.coords == {"c": ("a", "b", "d")}
    assert torch.masked_fill(x, x > 2, 0.0).names == ("r", "c")
    x.masked_fill_(x > 2, 0.0)  # in-place keeps them too
    assert x.names == ("r", "c")


def test_nonzero_drops_names_the_output_axes_are_not_the_inputs():
    x = XTensor(
        torch.tensor([[0, 5], [7, 0]]),
        names=("r", "c"),
        coords={"c": ("a", "b")},
    )
    # default: a (nnz, ndim) index tensor -- its axes are NOT r/c
    idx = x.nonzero()
    assert idx.shape == (2, 2)
    assert idx.names == (None, None)
    assert idx.coords == {}
    assert torch.nonzero(x).names == (None, None)
    # as_tuple: one 1-D index tensor per input dim, each unnamed
    parts = x.nonzero(as_tuple=True)
    assert len(parts) == 2
    assert all(p.names == (None,) for p in parts)


# ----------------------------------------------------------------------
# index_add / index_copy / index_fill (name-as-dim, method-form only)
# ----------------------------------------------------------------------


def test_index_fill_keeps_names_and_coordinates():
    x = XTensor(
        torch.zeros(2, 4),
        names=("a", "b"),
        coords={"b": ("w", "x", "y", "z")},
    )
    idx = torch.tensor([1, 3])
    out = x.index_fill("b", idx, 5.0)
    assert out.names == ("a", "b")
    # positions/sizes are unchanged, so the labels survive
    assert out.coords == {"b": ("w", "x", "y", "z")}
    assert torch.equal(out[:, 1], torch.full((2,), 5.0))
    assert torch.equal(out[:, 3], torch.full((2,), 5.0))


def test_index_add_keeps_names_and_coordinates():
    x = XTensor(
        torch.zeros(2, 4),
        names=("a", "b"),
        coords={"b": ("w", "x", "y", "z")},
    )
    idx = torch.tensor([1, 3])
    src = torch.ones(2, 2)
    out = x.index_add("b", idx, src)
    assert out.names == ("a", "b")
    assert out.coords == {"b": ("w", "x", "y", "z")}
    assert torch.equal(out[:, 1], torch.ones(2))
    assert torch.equal(out[:, 0], torch.zeros(2))


def test_index_copy_keeps_names_and_coordinates():
    x = XTensor(
        torch.zeros(2, 4),
        names=("a", "b"),
        coords={"b": ("w", "x", "y", "z")},
    )
    idx = torch.tensor([1, 3])
    src = torch.full((2, 2), 7.0)
    out = x.index_copy("b", idx, src)
    assert out.names == ("a", "b")
    assert out.coords == {"b": ("w", "x", "y", "z")}
    assert torch.equal(out[:, 1], torch.full((2,), 7.0))
    assert torch.equal(out[:, 3], torch.full((2,), 7.0))


# ----------------------------------------------------------------------
# internal helpers
# ----------------------------------------------------------------------


def test_slice_labels_supports_int_slice_bool_and_advanced_indices():
    labels = ("a", "b", "c", "d")
    assert _slice_labels(labels, 1) == ("b",)
    assert _slice_labels(labels, slice(1, 3)) == ("b", "c")
    assert _slice_labels(labels, [True, False, True, False]) == ("a", "c")
    assert _slice_labels(labels, [3, 0]) == ("d", "a")


# ----------------------------------------------------------------------
# convenience specializations
# ----------------------------------------------------------------------


def test_xvector_names_and_labels_the_channel_axis():
    v = xvector(torch.zeros(2, 3), channels=("x", "y", "z"))
    assert type(v) is XTensor  # a plain XTensor, not a distinct subclass
    assert v.names == (None, "channel")
    assert v.coords == {"channel": ("x", "y", "z")}
    assert v.sel(channel="y").shape == (2,)
    assert torch.equal(v.y, v.as_subclass(torch.Tensor)[:, 1])


def test_xvector_channel_dim_and_default_unlabelled():
    v = xvector(torch.zeros(3, 2), channel_dim=0)
    assert v.names == ("channel", None)
    # the default `channels=(...,)` names the axis without labelling it
    assert v.coords == {"channel": (None, None, None)}


def test_xmatrix_labels_row_and_col():
    m = xmatrix(torch.zeros(2, 3), rows=("r0", "r1"), cols=("c0", "c1", "c2"))
    assert type(m) is XTensor
    assert m.names == ("row", "col")
    assert m.coords == {"row": ("r0", "r1"), "col": ("c0", "c1", "c2")}
    assert m.sel(row="r1", col="c2").ndim == 0


def test_xvector_reduction_returns_a_plain_xtensor():
    # dropping the channel axis yields a normal XTensor -- the type never
    # outlives its meaning (the old XVector subclass did not maintain this)
    v = xvector(torch.zeros(2, 3), channels=("x", "y", "z"))
    reduced = v.sum(dim="channel")
    assert type(reduced) is XTensor
    assert reduced.names == (None,)


# ----------------------------------------------------------------------
# pointwise: broadcast-by-name (xarray-style)
# ----------------------------------------------------------------------


def test_add_aligns_transposed_operands_by_name():
    a = XTensor(torch.arange(6.0).reshape(2, 3), names=("x", "y"))
    b = XTensor(
        torch.arange(6.0).reshape(3, 2), names=("y", "x")
    )  # transposed
    out = a + b
    assert out.names == ("x", "y")
    assert out.shape == (2, 3)
    # aligning b to (x, y) is b.T; the sum matches
    assert torch.equal(out, a + b.rename("y", "x").T.rename("x", "y"))


def test_disjoint_named_dims_broadcast_to_the_outer_grid():
    c = XTensor(torch.arange(2.0), names=("x",))
    d = XTensor(torch.arange(3.0), names=("y",))
    out = c + d
    assert out.names == ("x", "y")
    assert out.shape == (2, 3)


def test_shared_name_broadcasts_size_one():
    e = XTensor(torch.arange(3.0).reshape(1, 3), names=("x", "y"))
    f = XTensor(torch.arange(6.0).reshape(2, 3), names=("x", "y"))
    assert (e + f).names == ("x", "y")
    assert (e + f).shape == (2, 3)


def test_scalar_and_plain_operands_keep_the_tensor_names():
    a = XTensor(torch.arange(6.0).reshape(2, 3), names=("x", "y"))
    assert (a + 1).names == ("x", "y")
    assert (a * 2).names == ("x", "y")
    assert (a + torch.ones(2, 3)).names == ("x", "y")
    assert (a == a).names == ("x", "y")  # comparisons too


def test_identical_names_align_positionally_even_with_nonleading_none():
    # same names tuple -> axes correspond 1:1, so a non-leading None is fine
    g = XTensor(torch.zeros(2, 3), names=("x", None))
    assert (g + g).names == ("x", None)
    h = XTensor(torch.zeros(2, 3, 4), names=("b", None, "c"))
    assert (h + h).names == ("b", None, "c")


def test_all_unnamed_operand_behaves_like_a_plain_tensor():
    # an all-None XTensor has nothing to align on -> positional, no raise
    a = XTensor(torch.zeros(2, 3), names=("x", "y"))
    u = XTensor(torch.zeros(2, 3))
    assert u.names == (None, None)
    assert (a + u).names == ("x", "y")


def test_positional_fallback_keeps_coordinates_regardless_of_operand_order():
    # issue #157: the positional fallback used to pick its coordinate source
    # by `isinstance(a, XTensor)` alone, so a fully-anonymous XTensor on the
    # left (`z`) silently out-ranked a named XTensor on the right (`p`) and
    # dropped its coordinates -- even though wrapping a plain tensor bare
    # (`z`) should be a no-op, exactly like using the plain tensor itself.
    p = XTensor(
        torch.ones(2, 3), names=("x", "y"), coords={"y": ("a", "b", "c")}
    )
    plain = torch.ones(2, 3)
    z = XTensor(torch.ones(2, 3))  # fully anonymous -- same data as `plain`
    expected = {"y": ("a", "b", "c")}
    assert (p + plain).coords == expected
    assert (plain + p).coords == expected
    assert (p + z).coords == expected
    assert (z + p).coords == expected  # was {} before the fix


def test_positional_fallback_prefers_the_named_side_either_order():
    # both operands are XTensor, one anonymous and one named -- the named
    # side's coordinates/names should win regardless of which side it's on.
    named = XTensor(
        torch.zeros(2, 3), names=(None, "y"), coords={"y": ("a", "b", "c")}
    )
    anon = XTensor(torch.zeros(2, 3))
    for out in (named + anon, anon + named):
        assert out.names == (None, "y")
        assert out.coords == {"y": ("a", "b", "c")}


def test_positional_fallback_still_carries_the_left_operands_data_unit():
    # the coordinate/name donor picked in the positional fallback (issue
    # #157) must stay independent of the _carry donor: without a units
    # backend, a data unit (and any other __dict__/subclass state) still
    # comes from the left operand, same rule as everywhere else -- even
    # when the left operand is the fully-anonymous one and therefore loses
    # the coordinate/name pick to the named operand on the right.
    z = XTensor(torch.ones(3), units="V")
    p = XTensor(torch.ones(3), names=("y",), coords={"y": ("a", "b", "c")})
    out = z + p
    assert out.units == "V"  # left operand's unit, unaffected by #157's fix
    assert out.coords == {"y": ("a", "b", "c")}  # right operand's coords

    left = XTensor(torch.ones(3), units="A")
    right = XTensor(torch.ones(3), names=("y",), units="V")
    assert (left + right).units == "A"  # left-operand rule, not name-based


def test_partial_names_align_named_suffix_broadcast_leading_anon():
    # issue #75: unnamed axes all leading -> named suffix aligns by name,
    # anonymous prefix broadcasts positionally. The shared name is used, so
    # the old silent mis-pair (square shapes) is gone.
    a = XTensor(torch.arange(9.0).reshape(3, 3), names=("x", "y"))
    b = XTensor(torch.zeros(3, 3), names=(None, "x"))
    out = a + b
    assert out.names == (None, "x", "y")  # not the old ('x', None)
    assert out.shape == (3, 3, 3)
    # a missing named axis broadcasts; differing anon counts right-align
    c = XTensor(torch.zeros(2, 4, 3, 5), names=(None, None, "x", "y"))
    d = XTensor(torch.zeros(4, 5, 3), names=(None, "y", "x"))
    assert (c + d).names == (None, None, "x", "y")
    assert (c + d).shape == (2, 4, 3, 5)


def test_partial_names_reconcile_coordinates_on_the_named_suffix():
    a = XTensor(torch.zeros(3), names=("y",), coords={"y": ("A", "B", "C")})
    b = XTensor(
        torch.zeros(4, 3), names=(None, "y"), coords={"y": ("C", "B", "A")}
    )
    out = a + b
    # aligned by label (inner-join), anon axis broadcast in front
    assert out.names == (None, "y")
    assert out.coords == {"y": ("A", "B", "C")}


def test_partial_names_not_all_leading_raises():
    # a None after a named axis, with *different* names -> ambiguous -> raise
    a = XTensor(torch.zeros(2, 3), names=("x", None))
    b = XTensor(torch.zeros(2, 3), names=("y", None))
    with pytest.raises(ValueError, match="not all leading"):
        _ = a + b


def test_pointwise_aligns_coordinates_by_name():
    p = XTensor(
        torch.zeros(2, 3), names=("x", "y"), coords={"y": ("a", "b", "c")}
    )
    q = XTensor(
        torch.zeros(3, 2), names=("y", "x"), coords={"y": ("a", "b", "c")}
    )
    out = p + q
    assert out.names == ("x", "y")
    assert out.coords == {"y": ("a", "b", "c")}


def test_pointwise_reindexes_operands_to_align_reordered_labels():
    # a shared labelled dim in a different label order is aligned by *label*,
    # not by position (xarray semantics): A+A, B+B, C+C.
    a = XTensor(
        torch.tensor([1.0, 2.0, 3.0]),
        names=("x",),
        coords={"x": ("A", "B", "C")},
    )
    b = XTensor(
        torch.tensor([10.0, 20.0, 30.0]),
        names=("x",),
        coords={"x": ("C", "B", "A")},
    )
    out = a + b
    assert out.coords == {"x": ("A", "B", "C")}
    assert out.tolist() == [31.0, 22.0, 13.0]


def test_pointwise_inner_joins_partially_overlapping_labels():
    a = XTensor(
        torch.tensor([1.0, 2.0, 3.0]),
        names=("x",),
        coords={"x": ("A", "B", "C")},
    )
    b = XTensor(
        torch.tensor([10.0, 20.0, 30.0]),
        names=("x",),
        coords={"x": ("B", "C", "D")},
    )
    out = a + b
    # intersection in a's order: B, C
    assert out.coords == {"x": ("B", "C")}
    assert out.tolist() == [12.0, 23.0]


def test_pointwise_alignment_broadcasts_over_a_disjoint_dim():
    a = XTensor(
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
        names=("r", "x"),
        coords={"x": ("A", "B")},
    )
    b = XTensor(
        torch.tensor([10.0, 20.0]), names=("x",), coords={"x": ("B", "A")}
    )
    out = a + b
    assert out.names == ("r", "x")
    assert out.coords == {"x": ("A", "B")}
    # b reindexed to (A, B) = [20, 10], then broadcast over r
    assert out.tolist() == [[21.0, 12.0], [23.0, 14.0]]


def test_pointwise_one_sided_labels_stay_positional():
    # only one operand labels the shared dim: nothing to align against, so the
    # labels ride along and the op stays positional.
    a = XTensor(
        torch.tensor([1.0, 2.0]), names=("x",), coords={"x": ("A", "B")}
    )
    b = XTensor(torch.tensor([10.0, 20.0]), names=("x",))
    out = a + b
    assert out.coords == {"x": ("A", "B")}
    assert out.tolist() == [11.0, 22.0]


# ----------------------------------------------------------------------
# axis descriptors (OME-NGFF-style: type / unit / orientation)
# ----------------------------------------------------------------------


def test_descriptor_axes_expose_extra_fields_while_names_stay_bare():
    x = XTensor(
        torch.zeros(2, 3, 4),
        axes=[
            {"name": "b", "type": "batch"},
            "h",
            {"name": "w", "type": "space", "orientation": "left-to-right"},
        ],
    )
    assert x.names == ("b", "h", "w")  # ergonomic view stays bare
    assert x.axes == (
        {"name": "b", "type": "batch"},
        {"name": "h"},
        {"name": "w", "type": "space", "orientation": "left-to-right"},
    )


def test_axes_keyword_is_an_alias_for_names():
    x = XTensor(torch.zeros(2), axes=[{"name": "t", "type": "time"}])
    assert x.axes == ({"name": "t", "type": "time"},)


def test_descriptor_requires_a_name_and_valid_orientation():
    with pytest.raises(ValueError, match="must have a 'name'"):
        XTensor(torch.zeros(2), axes=[{"type": "space"}])
    with pytest.raises(ValueError, match="a}-to-{b"):
        XTensor(torch.zeros(2), axes=[{"name": "x", "orientation": "lr"}])


def test_names_takes_strings_only_descriptors_go_through_axes():
    with pytest.raises(TypeError, match="descriptor dict through axes="):
        XTensor(torch.zeros(2), names=[{"name": "x", "type": "space"}])


def test_axes_embeds_coordinates():
    x = XTensor(
        torch.zeros(3, 4),
        axes=[
            {"name": "c", "labels": ["r", "g", "b"]},
            {"name": "x", "type": "space", "coord": {"spacing": (0.5, "mm")}},
        ],
    )
    assert x.names == ("c", "x")
    assert x.coords["c"] == ("r", "g", "b")  # `labels` -> categorical coord
    assert x.coords["x"]["value"].tolist() == [0.0, 0.5, 1.0, 1.5]  # `coord`
    assert x.axes[1] == {"name": "x", "type": "space"}  # meta kept


def test_axes_and_coords_merge_at_construction():
    x = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "a"}, {"name": "b", "type": "chan"}],
        coords={"b": ["p", "q", "r"]},
    )
    assert x.coords["b"] == ("p", "q", "r")  # coords= merges onto axes=
    assert x.axes[1] == {"name": "b", "type": "chan"}


def test_axis_meta_follows_the_dimension_through_ops():
    x = XTensor(
        torch.zeros(2, 3, 4),
        axes=[
            {"name": "b", "type": "batch"},
            "h",
            {"name": "w", "type": "space"},
        ],
    )
    # permute reorders descriptors with their axes
    assert x.permute(2, 0, 1).axes[0] == {"name": "w", "type": "space"}
    # reducing a described axis drops its metadata (getter filters it)
    assert x.sum(dim="w").axes == (
        {"name": "b", "type": "batch"},
        {"name": "h"},
    )


def test_rename_moves_axis_metadata_to_the_new_name():
    x = XTensor(torch.zeros(2, 3), axes=[{"name": "w", "type": "space"}, "h"])
    assert x.rename(w="width").axes[0] == {"name": "width", "type": "space"}


def test_flip_reverses_the_orientation_of_a_flipped_axis():
    x = XTensor(
        torch.zeros(2, 3),
        axes=[{"name": "y", "orientation": "top-to-bottom"}, "x"],
    )
    assert x.flip("y").axes[0]["orientation"] == "bottom-to-top"
    # an axis without an orientation is untouched
    assert x.flip("x").axes[0] == {"name": "y", "orientation": "top-to-bottom"}


def test_descriptors_and_coordinates_coexist():
    z = XTensor(
        torch.zeros(3),
        axes=[{"name": "c", "type": "channel"}],
        coords={"c": ("r", "g", "b")},
    )
    assert z.axes == ({"name": "c", "type": "channel"},)
    assert z.coords == {"c": ("r", "g", "b")}
    assert z.sel(c="g").item() == 0


# ----------------------------------------------------------------------
# combining axis descriptors across operands (the `combine_axes` option)
# ----------------------------------------------------------------------


def _sp(name):
    return {"name": name, "type": "space"}


def _tm(name):
    return {"name": name, "type": "time"}


def test_broadcast_keeps_descriptors_of_each_disjoint_dim():
    # both operands contribute a dim; each keeps its own descriptor (the
    # right operand's used to be dropped).
    a = XTensor(torch.ones(2), axes=[_sp("x")])
    b = XTensor(torch.ones(3), axes=[_tm("y")])
    assert (a + b).axes == (
        {"name": "x", "type": "space"},
        {"name": "y", "type": "time"},
    )


def test_shared_dim_keeps_agreeing_fields_and_one_sided_fields():
    a = XTensor(torch.ones(3), axes=[_sp("x")])
    b = XTensor(torch.ones(3), axes=[_sp("x")])
    assert (a + b).axes == ({"name": "x", "type": "space"},)
    # a field on only one side is not a conflict -- it is kept
    c = XTensor(torch.ones(3), names=["x"])
    assert (a + c).axes == ({"name": "x", "type": "space"},)


def test_conflicting_field_is_dropped_and_order_independent():
    a = XTensor(torch.ones(3), axes=[_sp("x")])
    b = XTensor(torch.ones(3), axes=[_tm("x")])
    # type conflicts (space vs time) -> the field drops; the bare name stays
    assert (a + b).axes == ({"name": "x"},)
    assert (b + a).axes == ({"name": "x"},)  # no left-operand bias


def test_strict_policy_raises_on_conflict_and_restores_after_block():
    a = XTensor(torch.ones(3), axes=[_sp("x")])
    b = XTensor(torch.ones(3), axes=[_tm("x")])
    with set_options(combine_axes="strict"):
        with pytest.raises(ValueError, match="conflicting 'type'"):
            _ = a + b
        # compatible descriptors do not raise even under strict
        assert (a + a).axes == ({"name": "x", "type": "space"},)
    # the option is restored on exit -> conflict drops again
    assert (a + b).axes == ({"name": "x"},)


def test_override_policy_lets_the_left_operand_win():
    a = XTensor(torch.ones(3), axes=[_sp("x")])
    b = XTensor(torch.ones(3), axes=[_tm("x")])
    with set_options(combine_axes="override"):
        assert (a + b).axes == ({"name": "x", "type": "space"},)
        assert (b + a).axes == ({"name": "x", "type": "time"},)


def test_drop_policy_removes_all_descriptors():
    a = XTensor(torch.ones(2), axes=[_sp("x")])
    b = XTensor(torch.ones(3), axes=[_tm("y")])
    with set_options(combine_axes="drop"):
        assert (a + b).axes == ({"name": "x"}, {"name": "y"})


def test_set_options_rejects_unknown_option_or_value():
    with pytest.raises(ValueError, match="unknown option"):
        set_options(nope=1)
    with pytest.raises(ValueError, match="invalid combine_axes policy"):
        set_options(combine_axes="bogus")
    with pytest.raises(ValueError, match="invalid combine_axes policy"):
        set_options(combine_axes={"type": "bogus"})
    with pytest.raises(ValueError, match="must be a policy str or a"):
        set_options(combine_axes=5)
    with pytest.raises(ValueError, match="invalid unit_backend"):
        set_options(unit_backend="nope")
    with pytest.raises(ValueError, match="invalid unit_policy"):
        set_options(unit_policy="nope")


# ----------------------------------------------------------------------
# data units (Proposal 0003 — the .units annotation)
# ----------------------------------------------------------------------


def test_data_unit_is_stored_carried_and_opaque_without_a_backend():
    # default backend is None: a unit is an opaque string, stored and carried
    x = XTensor(torch.ones(2, 3), names=("a", "b"), units="V")
    assert x.units == "V"
    assert x.T.units == "V"  # rides through reshape/reorder
    assert x.sum(dim="a").units == "V"  # ... and reductions
    assert x[0].units == "V"  # ... and indexing
    x.units = None
    assert x.units is None
    x.units = "not_a_real_unit"  # no backend -> no validation
    assert x.units == "not_a_real_unit"


def test_data_unit_pint_backend_validates_and_normalises():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        y = XTensor(torch.ones(3), units="mV")
        assert y.units == "millivolt"  # canonicalised
        with pytest.raises(ValueError, match="invalid unit"):
            XTensor(torch.ones(2), units="not_a_unit_zz")


def test_to_units_converts_the_data_by_the_conversion_factor():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        y = XTensor(torch.ones(3), units="mV")
        z = y.to_units("V")
        assert z.units == "volt"
        assert torch.allclose(
            z.as_subclass(torch.Tensor), torch.full((3,), 0.001)
        )
        with pytest.raises(ValueError, match="no unit to convert"):
            XTensor(torch.ones(2)).to_units("V")


def _united(**kw):
    return {name: XTensor(torch.ones(3), units=u) for name, u in kw.items()}


def test_data_unit_algebra_mul_div_pow():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        u = _united(V="V", A="A", s="s")
        V, A, s = u["V"], u["A"], u["s"]
        assert (V * A).units == "ampere * volt"  # product
        assert (V / s).units == "volt / second"  # quotient
        assert (V * 2).units == "volt"  # a scalar is dimensionless
        assert (V**2).units == "volt ** 2"  # power (via the `**` operator)


def test_data_unit_algebra_add_and_compare():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        V = XTensor(torch.ones(3), units="V")
        A = XTensor(torch.ones(3), units="A")
        assert (V + V).units == "volt"  # same unit kept
        assert (V + A).units is None  # incompatible -> dropped (default)
        assert (V < V).units is None  # comparison result is unitless
        with set_options(unit_policy="strict"):
            with pytest.raises(ValueError, match="incompatible units"):
                _ = V + A


def test_data_unit_matmul_multiplies_units():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        M = XTensor(torch.ones(2, 2), units="V")
        N = XTensor(torch.ones(2, 2), units="A")
        assert (M @ N).units == "ampere * volt"


def test_data_unit_transcendental_requires_dimensionless():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        V = XTensor(torch.ones(3), units="V")
        assert torch.exp(V).units is None  # drop the unit (default policy)
        assert torch.log(XTensor(torch.ones(3))).units is None  # unitless
        with set_options(unit_policy="strict"):
            with pytest.raises(ValueError, match="dimensionless"):
                torch.exp(V)


def test_data_unit_algebra_preserves_autograd():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        leaf = torch.ones(3, requires_grad=True)
        A = XTensor(torch.ones(3), units="A")
        (XTensor(leaf, units="V") * A).sum().backward()
        assert leaf.grad is not None


def test_data_unit_algebra_is_inert_without_a_backend():
    # no backend: no algebra, the unit just rides along opaquely
    V = XTensor(torch.ones(3), units="V")
    A = XTensor(torch.ones(3), units="A")
    assert (V * A).units == "V"  # carried from the left operand, not combined
    assert torch.exp(V).units == "V"  # not dropped


# -- attaching a unit by multiplication (Proposal 0003 phase 4, §2.4) ---------


def test_multiplying_by_a_unit_attaches_it():
    pint = pytest.importorskip("pint")
    u = pint.UnitRegistry()
    with set_options(unit_backend="pint"):
        x = XTensor(torch.arange(3.0), names=("t",))
        attached = x * u.mm  # a bare Unit: data unchanged, unit attached
        assert attached.units == "millimeter"
        assert attached.tolist() == [0.0, 1.0, 2.0]
        assert attached.names == ("t",)  # names ride through
        assert x.units is None  # the original is never annotated in place


def test_multiplying_by_a_quantity_scales_the_data():
    pint = pytest.importorskip("pint")
    u = pint.UnitRegistry()
    with set_options(unit_backend="pint"):
        x = XTensor(torch.arange(3.0))
        scaled = x * (3 * u.mm)  # a Quantity carries a magnitude
        assert scaled.units == "millimeter"
        assert scaled.tolist() == [0.0, 3.0, 6.0]


def test_dividing_by_a_unit_derives_the_quotient_unit():
    pint = pytest.importorskip("pint")
    u = pint.UnitRegistry()
    with set_options(unit_backend="pint"):
        v = XTensor(torch.ones(3), units="V")
        assert (v / u.s).units == "volt / second"
        assert (v * u.ohm).units == "ohm * volt"


def test_unit_multiplication_leaves_ordinary_operands_untouched():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(torch.arange(3.0), units="m")
        assert (x * 2).tolist() == [0.0, 2.0, 4.0]  # scalar mul, not a unit
        assert (2 * x).tolist() == [0.0, 2.0, 4.0]  # reflected scalar
        assert (x * x).units == "meter ** 2"  # two united tensors: algebra
        assert (x / 2).units == "meter"  # scalar divide keeps the unit


# -- heterogeneous (per-axis) data units (Proposal 0003 phase 3) --------------


def _channel_stack():
    # a `q` axis whose positions carry different data units
    return XTensor(
        torch.arange(12.0).reshape(3, 4),
        names=("q", "t"),
        coords={
            "q": [
                {"name": "voltage", "unit": "V"},
                {"name": "current", "unit": "A"},
                {"name": "power", "unit": "W"},
            ]
        },
    )


def test_hetero_unit_folds_into_base_on_selection():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = _channel_stack()
        assert x.units is None  # heterogeneous: no single base unit
        assert x.sel(q="voltage").units == "V"  # fold the channel's unit
        assert x.sel(q="current").units == "A"
        assert x.isel(q=2).units == "W"  # isel folds too
        assert x[0].units == "V"  # and plain integer indexing


def test_hetero_unit_selection_multiplies_into_an_existing_base():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(
            torch.arange(12.0).reshape(3, 4),
            names=("q", "t"),
            coords={
                "q": [
                    {"name": "a", "unit": "m"},
                    {"name": "b", "unit": "m"},
                    {"name": "c", "unit": "m"},
                ]
            },
            units="s",
        )
        assert x.isel(q=0).units == "meter * second"  # base * coord unit


def test_hetero_unit_slice_keeps_axis_and_units():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = _channel_stack()
        sl = x.isel(q=slice(0, 2))  # keeps the axis -> no fold
        assert sl.units is None
        assert sl.coords["q"] == (
            {"name": "voltage", "unit": "V"},
            {"name": "current", "unit": "A"},
        )


def _uniform_stack(unit="V"):
    return XTensor(
        torch.arange(12.0).reshape(3, 4),
        names=("q", "t"),
        coords={"q": [{"name": n, "unit": unit} for n in "abc"]},
    )


def test_reduction_folds_a_uniform_axis_unit():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = _uniform_stack("V")
        assert x.sum(dim="q").units == "V"  # uniform axis unit folds in
        assert x.mean(dim="q").units == "V"
        assert x.sum(dim="q", keepdim=True).units == "V"  # keepdim too
        assert x.sum().units == "V"  # dim=None reduces the unit axis as well


def test_reduction_keeps_base_over_a_unitless_axis():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = _uniform_stack("V")
        assert x.sum(dim="t").units is None  # `t` carries no units; base stays


def test_reduction_over_incompatible_units_drops_or_raises():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = _channel_stack()
        assert x.sum(dim="q").units is None  # V/A/W incompatible -> dropped
        with set_options(unit_policy="strict"):
            with pytest.raises(ValueError, match="incompatible units"):
                x.sum(dim="q")


def test_hetero_units_are_inert_without_a_backend():
    x = _channel_stack()  # unit_backend=None
    assert x.sel(q="voltage").units is None  # no fold when the layer is inert
    assert x.sum(dim="q").units is None


# -- phase 4: detach, implicit conversion, heterogeneous contraction ----------


def test_magnitude_drops_the_data_unit_but_keeps_names():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(torch.arange(3.0), names=("t",), units="V")
        m = x.magnitude
        assert m.units is None  # the data unit is stripped
        assert m.names == ("t",)  # names/coords ride through
        assert m.tolist() == [0.0, 1.0, 2.0]
        assert x.units == "volt"  # the original is unchanged (a view)


def test_add_implicitly_converts_compatible_units():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        volts = XTensor(torch.ones(3), names=("t",), units="V")
        millivolts = XTensor(torch.full((3,), 500.0), names=("t",), units="mV")
        # right operand converts to the left's unit before adding
        left = volts + millivolts
        assert left.units == "volt"
        assert left.tolist() == [1.5, 1.5, 1.5]
        right = millivolts + volts
        assert right.units == "millivolt"
        assert right.tolist() == [1500.0, 1500.0, 1500.0]
        # comparisons convert too
        assert (volts > millivolts).tolist() == [True, True, True]


def test_add_incompatible_units_still_drops_or_raises():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        volts = XTensor(torch.ones(3), units="V")
        amps = XTensor(torch.ones(3), units="A")
        assert (volts + amps).units is None  # incompatible -> dropped
        with set_options(unit_policy="strict"):
            with pytest.raises(ValueError, match="incompatible units"):
                _ = volts + amps


def test_matmul_folds_uniform_contracted_axis_units():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        a = XTensor(
            torch.ones(2, 3),
            names=("i", "k"),
            coords={"k": [{"name": c, "unit": "m"} for c in "abc"]},
            units="V",
        )
        b = XTensor(
            torch.ones(3, 2),
            names=("k", "j"),
            coords={"k": [{"name": c, "unit": "s"} for c in "abc"]},
            units="A",
        )
        # (V·m) · (A·s) — each side's uniform contracted-axis unit folds in
        assert (a @ b).units == "ampere * meter * second * volt"


def test_contraction_over_non_uniform_axis_drops_or_raises():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        a = XTensor(
            torch.ones(2, 3),
            names=("i", "k"),
            coords={
                "k": [
                    {"name": "x", "unit": "m"},
                    {"name": "y", "unit": "s"},
                    {"name": "z", "unit": "kg"},
                ]
            },
        )
        b = XTensor(torch.ones(3, 2), names=("k", "j"))
        assert (a @ b).units is None  # non-uniform contracted axis -> dropped
        with set_options(unit_policy="strict"):
            with pytest.raises(ValueError, match="non-uniform"):
                _ = a @ b


def test_einsum_and_tensordot_multiply_operand_units():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        m = XTensor(torch.ones(2, 2), units="V")
        n = XTensor(torch.ones(2, 2), units="A")
        # einsum previously kept only the first operand's base unit
        assert torch.einsum("ij,jk->ik", m, n).units == "ampere * volt"
        assert torch.tensordot(m, n, dims=1).units == "ampere * volt"
        # an unparsable equation (ellipsis) falls back to the base product
        p = XTensor(torch.ones(2, 2, 2), units="V")
        q = XTensor(torch.ones(2, 2, 2), units="A")
        out = torch.einsum("...ij,...jk->...ik", p, q)
        assert out.units == "ampere * volt"


# ----------------------------------------------------------------------
# the pint.Quantity-shaped API (Proposal 0006)
# ----------------------------------------------------------------------


def test_dimensionality_reports_the_physical_dimensions():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        assert XTensor(torch.ones(3), units="m").dimensionality == "[length]"
        power = XTensor(torch.ones(3), units="W")
        assert power.dimensionality == "[mass] * [length] ** 2 / [time] ** 3"
        # no unit at all -> the empty string, not an error
        assert XTensor(torch.ones(3)).dimensionality == ""


def test_unitless_is_stricter_than_dimensionless_on_angles():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        # an angle is dimensionally trivial but still names a unit
        radians = XTensor(torch.ones(3), units="rad")
        assert radians.dimensionless is True
        assert radians.unitless is False
        volts = XTensor(torch.ones(3), units="V")
        assert volts.dimensionless is False
        assert volts.unitless is False
        # no unit, and an explicitly dimensionless one, are both
        bare = XTensor(torch.ones(3))
        assert bare.dimensionless is True
        assert bare.unitless is True
        empty = XTensor(torch.ones(3), units="")
        assert empty.dimensionless is True
        assert empty.unitless is True


def test_dimensionality_queries_degrade_clearly_without_a_backend():
    x = XTensor(torch.ones(3), units="V")
    with pytest.raises(ValueError, match="dimensionality requires"):
        _ = x.dimensionality
    # without a dimensionality system to consult, the two predicates
    # coincide: `not bool(units)`
    assert x.dimensionless is False
    assert x.unitless is False
    bare = XTensor(torch.ones(3))
    assert bare.dimensionless is True
    assert bare.unitless is True
    assert XTensor(torch.ones(3), units="").dimensionless is True


def test_units_dimensionless_helper_without_a_backend():
    from fiery.xtensor import _units

    # an explicitly empty unit *is* dimensionless with no backend; an opaque
    # unit string is not (there is no way to tell)
    assert _units.dimensionless("") is True
    assert _units.dimensionless(None) is True
    assert _units.dimensionless("V") is False


def test_is_compatible_with_matches_convertible_units():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        volts = XTensor(torch.ones(3), units="V")
        assert volts.is_compatible_with("mV") is True
        assert volts.is_compatible_with("A") is False
        # no unit at all is compatible with nothing
        assert XTensor(torch.ones(3)).is_compatible_with("V") is False
        with pytest.raises(ValueError, match="invalid unit"):
            volts.is_compatible_with("not_a_unit_zz")


def test_is_compatible_with_falls_back_to_string_equality():
    # no backend: no dimensionality system, so only the same spelling matches
    x = XTensor(torch.ones(3), units="V")
    assert x.is_compatible_with("V") is True
    assert x.is_compatible_with("mV") is False


def test_m_as_converts_and_drops_the_unit():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        volts = XTensor(torch.ones(3), names=("t",), units="V")
        bare = volts.m_as("mV")
        assert bare.units is None  # dropped
        assert bare.names == ("t",)  # still an XTensor
        assert bare.tolist() == [1000.0, 1000.0, 1000.0]  # converted
        with pytest.raises(ValueError, match="no unit to convert"):
            XTensor(torch.ones(3)).m_as("mV")


def test_to_units_converts_in_place():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(torch.ones(3), names=("t",), units="V")
        pointer = x.data_ptr()
        result = x.to_units_("mV")
        assert result is x  # returns self
        assert x.units == "millivolt"
        assert x.tolist() == [1000.0, 1000.0, 1000.0]
        assert x.data_ptr() == pointer  # same storage
        assert x.names == ("t",)


def test_to_units_in_place_needs_a_unit_and_a_backend():
    x = XTensor(torch.ones(3), units="V")
    with pytest.raises(ValueError, match="unit_backend='pint'"):
        x.to_units_("mV")
    assert x.units == "V"  # nothing changed
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        with pytest.raises(ValueError, match="no unit to convert"):
            XTensor(torch.ones(3)).to_units_("mV")
        with pytest.raises(ValueError, match="invalid unit"):
            XTensor(torch.ones(3), units="V").to_units_("not_a_unit_zz")


def test_to_units_in_place_has_the_usual_in_place_restrictions():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        # a grad-requiring leaf, exactly as for `mul_`/`add_`
        leaf = XTensor(torch.ones(3, requires_grad=True), units="V")
        with pytest.raises(RuntimeError, match="leaf Variable that requires"):
            leaf.to_units_("mV")
        # a non-leaf converts fine, and the gradient still flows
        source = torch.ones(3, requires_grad=True)
        y = XTensor(source * 2, units="V")
        y.to_units_("mV")
        assert y.units == "millivolt"
        y.sum().backward()
        assert source.grad is not None
        # an int dtype cannot absorb a non-integer factor
        ints = XTensor(torch.ones(3, dtype=torch.int64), units="V")
        with pytest.raises(RuntimeError, match="cast"):
            ints.to_units_("mV")


def test_to_recognises_a_backend_unit_positionally():
    pint = pytest.importorskip("pint")
    u = pint.UnitRegistry()
    with set_options(unit_backend="pint"):
        x = XTensor(torch.ones(3), names=("t",), units="m")
        converted = x.to(u.mm)
        assert converted.units == "millimeter"
        assert converted.tolist() == [1000.0, 1000.0, 1000.0]
        assert converted.names == ("t",)
        assert x.units == "meter"  # the original is untouched
        # a Quantity's own magnitude is not applied -- `.to()` converts
        assert x.to(3 * u.mm).tolist() == [1000.0, 1000.0, 1000.0]
        # a positional unit still combines with torch's own no-dtype/device
        # kwargs (copy=, non_blocking=, memory_format=) -- once the unit is
        # consumed there's nothing left for them to be ambiguous with
        also_converted = x.to(u.mm, copy=True)
        assert also_converted.units == "millimeter"
        assert also_converted.tolist() == [1000.0, 1000.0, 1000.0]


def test_to_converts_through_the_units_keyword():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(torch.ones(3), units="m")
        # unlike the positional form, the keyword takes a plain string
        assert x.to(units="mm").units == "millimeter"
        assert x.to(units="mm").tolist() == [1000.0, 1000.0, 1000.0]
        # ... and composes with dtype/device and the metadata overrides
        both = x.to(torch.float64, units="mm", names=("t",))
        assert both.dtype == torch.float64
        assert both.units == "millimeter"
        assert both.names == ("t",)
        with pytest.raises(ValueError, match="no unit to convert"):
            XTensor(torch.ones(3)).to(units="mm")


def test_to_overrides_names_and_coords_like_as_xtensor():
    x = XTensor(torch.ones(3))
    out = x.to(names=("t",), coords={"t": ("a", "b", "c")})
    assert out.names == ("t",)
    assert out.coords == {"t": ("a", "b", "c")}
    assert x.names == (None,)  # the original keeps its own metadata


def test_to_rejects_units_none_as_a_conversion_target():
    x = XTensor(torch.ones(3), units="V")
    with pytest.raises(ValueError, match="units=None is not a conversion"):
        x.to(units=None)
    with pytest.raises(ValueError, match="units=None is not a conversion"):
        x.to_(units=None)


def test_to_still_behaves_like_plain_torch_to():
    x = XTensor(torch.ones(3), names=("t",), units="V")
    assert x.to(torch.float64).dtype == torch.float64
    assert x.to(torch.float64).names == ("t",)  # metadata rides through
    assert x.to(torch.float64).units == "V"
    assert x.to(dtype=torch.float64).dtype == torch.float64
    # a positional *string* stays torch's device spelling, never a unit
    assert x.to("cpu").names == ("t",)
    assert x.to(torch.zeros(3, dtype=torch.float64)).dtype == torch.float64


def test_to_in_place_mutates_metadata_but_refuses_a_dtype_change():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(torch.ones(3), names=("t",), units="m")
        pointer = x.data_ptr()
        result = x.to_(
            torch.float32, units="mm", coords={"t": (1.0, 2.0, 3.0)}
        )
        assert result is x
        assert x.units == "millimeter"
        assert x.tolist() == [1000.0, 1000.0, 1000.0]
        assert x.data_ptr() == pointer
        assert x.coords["t"]["value"].tolist() == [1.0, 2.0, 3.0]
        # an actual dtype change is not an in-place operation
        with pytest.raises(ValueError, match="cannot change dtype or device"):
            x.to_(torch.float64)
        assert x.dtype == torch.float32  # and nothing was changed


def test_to_in_place_applies_units_last_so_a_bad_names_override_stays_atomic():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        x = XTensor(torch.ones(2, 3), names=("b", "t"), units="m")
        with pytest.raises(ValueError, match="Expected 2 names"):
            x.to_(units="mm", names=("a", "b", "c"))
        # names= raised before units= ever ran -- nothing changed
        assert x.units == "meter"
        assert x.tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]
        assert x.names == ("b", "t")


def test_to_in_place_accepts_a_no_op_copy_and_renames():
    x = XTensor(torch.ones(3), names=("t",))
    # `copy=True` produces a new tensor internally, but dtype/device match,
    # so it is not treated as a rejected conversion
    assert x.to_(copy=True) is x
    assert x.to_(names=("q",)).names == ("q",)
    with pytest.raises(ValueError, match="Expected 1 names"):
        x.to_(names=("a", "b"))


def test_to_in_place_needs_a_unit_for_a_units_override():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        with pytest.raises(ValueError, match="no unit to convert"):
            XTensor(torch.ones(3)).to_(units="mm")


def _force():
    # 5000 g·mm/s², the worked example from the proposal
    return XTensor(torch.tensor([5000.0]), names=("t",), units="g*mm/s**2")


def test_to_base_units_converts_to_si_base_units():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        out = _force().to_base_units()
        assert out.units == "kilogram * meter / second ** 2"
        assert out.names == ("t",)
        assert out.item() == pytest.approx(0.005)


def test_to_reduced_units_simplifies_the_unit():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        out = _force().to_reduced_units()
        assert out.units == "gram * millimeter / second ** 2"
        assert out.item() == pytest.approx(5000.0)
        # a unit that really does reduce
        squared = XTensor(torch.tensor([2.0]), units="m*mm")
        assert squared.to_reduced_units().units == "millimeter ** 2"


def test_to_compact_picks_a_nice_scale_for_the_values():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        out = _force().to_compact()
        assert out.units == "gram * meter / second ** 2"
        assert out.item() == pytest.approx(5.0)
        # the choice follows the data, not just the unit
        tiny = XTensor(torch.tensor([200e-9]), units="s")
        compacted = tiny.to_compact()
        assert compacted.units == "nanosecond"
        assert compacted.item() == pytest.approx(200.0)
        # an empty tensor has no magnitude to go on: it keeps its own unit
        empty = XTensor(torch.zeros(0), units="s")
        assert empty.to_compact().units == "second"
        # multi-element: the *largest* magnitude drives the choice
        multi = XTensor(torch.tensor([1e-9, 5e-6, 2e-9]), units="s")
        assert multi.to_compact().units == "microsecond"
        # a NaN must not veto the finite values' own max
        mixed = XTensor(torch.tensor([float("nan"), 5e-6, 2e-9]), units="s")
        assert mixed.to_compact().units == "microsecond"
        # no finite value anywhere: falls back to magnitude 1, keeps its unit
        allnan = XTensor(torch.tensor([float("nan"), float("inf")]), units="s")
        assert allnan.to_compact().units == "second"


def test_to_preferred_needs_a_table_of_units():
    pint = pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        out = _force().to_preferred(["N"])
        assert out.units == "newton"
        assert out.item() == pytest.approx(0.005)
        # with no table given, and none configured on the backend registry,
        # the backend itself refuses
        with pytest.raises(
            pint.UndefinedUnitError, match="default_preferred_units"
        ):
            _force().to_preferred()


def test_the_simplification_family_has_in_place_counterparts():
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        base = _force()
        assert base.to_base_units_() is base
        assert base.units == "kilogram * meter / second ** 2"
        assert base.item() == pytest.approx(0.005)

        reduced = XTensor(torch.tensor([2.0]), units="m*mm")
        reduced.to_reduced_units_()
        assert reduced.units == "millimeter ** 2"

        compact = _force()
        compact.to_compact_()
        assert compact.units == "gram * meter / second ** 2"
        assert compact.item() == pytest.approx(5.0)

        preferred = _force()
        preferred.to_preferred_(["N"])
        assert preferred.units == "newton"
        assert preferred.item() == pytest.approx(0.005)


def test_the_simplification_family_needs_a_unit_and_a_backend():
    for name in (
        "to_base_units",
        "to_reduced_units",
        "to_compact",
        "to_preferred",
    ):
        for spelling in (name, name + "_"):
            # no backend
            method = getattr(XTensor(torch.ones(3), units="V"), spelling)
            with pytest.raises(ValueError, match="unit_backend='pint'"):
                method()
    pytest.importorskip("pint")
    with set_options(unit_backend="pint"):
        for name in (
            "to_base_units",
            "to_reduced_units",
            "to_compact",
            "to_preferred",
        ):
            for spelling in (name, name + "_"):
                method = getattr(XTensor(torch.ones(3)), spelling)
                with pytest.raises(ValueError, match="no unit to convert"):
                    method()


def test_to_needs_no_backend_for_dtype_or_metadata():
    # with no backend a positional backend object cannot even exist, so
    # `.to()`/`.to_()` are exactly plain torch plus the metadata overrides
    x = XTensor(torch.ones(3), names=("t",), units="V")
    assert x.to(torch.float64, names=("q",)).names == ("q",)
    assert x.to_(names=("q",)).names == ("q",)
    with pytest.raises(ValueError, match="unit_backend='pint'"):
        x.to(units="mV")
    with pytest.raises(ValueError, match="unit_backend='pint'"):
        x.to_(units="mV")


def test_combine_axes_accepts_a_per_field_policy():
    # a custom, free-form descriptor field ("role") -- nothing is special
    # about it; the per-field policy applies to any key.
    a = XTensor(
        torch.ones(3),
        axes=[{"name": "x", "type": "space", "role": "readout"}],
    )
    b = XTensor(
        torch.ones(3),
        axes=[{"name": "x", "type": "time", "role": "readout"}],
    )
    # default drops every field, but an agreeing "role" is kept
    with set_options(combine_axes={"*": "drop", "role": "raise"}):
        assert (a + b).axes == ({"name": "x", "role": "readout"},)


def test_per_field_raise_policy_fires_only_for_that_field():
    a = XTensor(torch.ones(3), axes=[{"name": "x", "role": "readout"}])
    b = XTensor(torch.ones(3), axes=[{"name": "x", "role": "phase"}])
    with set_options(combine_axes={"*": "drop", "role": "raise"}):
        with pytest.raises(ValueError, match="conflicting 'role'"):
            _ = a + b


def test_per_field_override_keeps_the_left_value_for_one_field():
    a = XTensor(
        torch.ones(3),
        axes=[{"name": "x", "type": "space", "orientation": "left-to-right"}],
    )
    b = XTensor(
        torch.ones(3),
        axes=[{"name": "x", "type": "space", "orientation": "right-to-left"}],
    )
    with set_options(combine_axes={"orientation": "override"}):
        # "type" agrees (kept); "orientation" takes the left operand's value
        assert (a + b).axes[0]["orientation"] == "left-to-right"
        assert (b + a).axes[0]["orientation"] == "right-to-left"


def test_unlisted_fields_fall_back_to_drop_conflicts():
    a = XTensor(torch.ones(3), axes=[{"name": "x", "type": "space"}])
    b = XTensor(torch.ones(3), axes=[{"name": "x", "type": "time"}])
    # only "role" is customised; "type" uses the drop_conflicts default
    with set_options(combine_axes={"role": "strict"}):
        assert (a + b).axes == ({"name": "x"},)


def test_descriptor_keys_are_free_form_not_a_fixed_schema():
    # `type` is only a convention; any custom key is stored, carried through
    # ops, and queryable exactly the same way.
    x = XTensor(
        torch.zeros(2, 3),
        axes=[
            {"name": "c", "modality": "MRI", "role": "readout"},
            {"name": "t", "modality": "MRI"},
        ],
    )
    assert x.axes[0] == {"name": "c", "modality": "MRI", "role": "readout"}
    # a custom key follows the dim through an op ...
    assert x.T.axes[1]["role"] == "readout"
    # ... and addresses axes just like `type` would
    assert x.sum(dim={"modality": "MRI"}).ndim == 0
    assert x.sum(dim={"role": "readout"}).names == ("t",)


def test_movedim_by_type_moves_the_whole_block_to_the_back():
    x = XTensor(
        torch.zeros(2, 3, 4, 5),
        axes=[
            {"name": "b", "type": "batch"},
            {"name": "h", "type": "space"},
            {"name": "c", "type": "channel"},
            {"name": "w", "type": "space"},
        ],
    )
    moved = x.movedim({"type": "space"}, -1)
    # both space axes go to the back, keeping their relative order (h before w)
    assert moved.names == ("b", "c", "h", "w")
    assert tuple(moved.shape) == (2, 4, 3, 5)


@pytest.mark.skipif(
    not _HAS_MOVEAXIS, reason="torch.moveaxis added in torch 1.8"
)
def test_moveaxis_by_type_moves_the_block_to_the_front():
    x = XTensor(
        torch.zeros(2, 3, 4, 5),
        axes=[
            {"name": "b", "type": "batch"},
            {"name": "h", "type": "space"},
            {"name": "c", "type": "channel"},
            {"name": "w", "type": "space"},
        ],
    )
    moved = x.moveaxis({"type": "space"}, 0)
    assert moved.names == ("h", "w", "b", "c")
    assert tuple(moved.shape) == (3, 5, 2, 4)


def test_sum_by_type_reduces_every_matching_axis():
    y = XTensor(
        torch.ones(2, 3, 4),
        axes=[
            {"name": "b", "type": "batch"},
            {"name": "c1", "type": "channel"},
            {"name": "c2", "type": "channel"},
        ],
    )
    reduced = y.sum(dim={"type": "channel"})
    assert reduced.names == ("b",)
    assert reduced[0].item() == 12  # 3 * 4 ones summed


def test_single_match_query_stays_prod_safe():
    # a query hitting exactly one axis collapses to a bare int, so even
    # single-`dim`-only reducers (`prod`) accept it
    y = XTensor(
        torch.ones(2, 3, 4),
        axes=[
            {"name": "b", "type": "batch"},
            {"name": "c1", "type": "channel"},
            {"name": "c2", "type": "channel"},
        ],
    )
    assert y.prod(dim={"type": "batch"}).names == ("c1", "c2")


def test_deepcopy_produces_an_independent_copy_with_metadata():
    import copy

    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    y = copy.deepcopy(x)
    assert type(y) is XTensor
    assert y.names == x.names
    assert y.coords["t"]["value"].tolist() == x.coords["t"]["value"].tolist()
    y[0] = 99.0
    assert x[0].item() == 0.0  # independent storage, original untouched


def test_deepcopy_preserves_requires_grad_as_a_fresh_leaf():
    import copy

    g = torch.arange(4.0, requires_grad=True)
    x = XTensor(g, names=("t",))
    y = copy.deepcopy(x)
    assert y.requires_grad
    assert y.is_leaf
    assert y.data_ptr() != x.data_ptr()

    # a non-leaf source (an intermediate result) still deep-copies cleanly,
    # to a fresh, independent, grad-requiring leaf -- this implementation
    # doesn't try to preserve the backward graph across a deepcopy either
    # way, so there's no reason to refuse a non-leaf input the way vanilla
    # `Tensor.__deepcopy__` does.
    non_leaf = XTensor(torch.arange(4.0, requires_grad=True), names=("t",)) * 2
    copied = copy.deepcopy(non_leaf)
    assert copied.requires_grad
    assert copied.is_leaf


def test_repr_of_an_int_dtype_tensor_does_not_recurse():
    # issue #118: torch.Tensor.__format__ checks `type(self) is Tensor` and
    # falls back to `object.__format__` (== str(self)) for any subclass --
    # fatal specifically for a non-float (int/bool) dtype tensor, whose repr
    # formats each element via f"{value}" on a 0-dim slice of the same
    # subclass, which then recurses back into the very same tensor-printing
    # machinery forever. A float-dtype tensor's repr instead computes its
    # display width from `torch.masked_select(...)`'s output -- a *non-view*
    # op, so the result is a plain `Tensor`, not this subclass, and torch's
    # own `.item()` fast path is taken instead of recursing. The
    # `XTensor(...)` wrapper prefix itself is torch's own subclass-aware
    # repr, only present on torch versions that added it -- older torch
    # (this package's floor is 1.7) just prints `tensor(...)` for any
    # subclass, so only check the data renders and nothing raises, not the
    # exact prefix.
    x = XTensor(torch.arange(3))
    assert "0, 1, 2" in repr(x)
    assert "0, 1, 2" in str(x)
    b = XTensor(torch.tensor([True, False]))
    assert "True" in repr(b)


def test_repr_of_an_int_dtype_coordinate_does_not_recurse():
    # the actual reported entry point (#118's title): a numeric coordinate
    # promoted to int dtype (#107) is exactly the shape that made this easy
    # to hit by accident.
    x = XTensor(torch.arange(3), names=("t",), coords={"t": (10, 20, 30)})
    assert "0, 1, 2" in repr(x)
    assert "10, 20, 30" in repr(x.coords)


def test_format_of_a_zero_dim_tensor_extracts_the_scalar():
    s_int = XTensor(torch.tensor(3))
    assert f"{s_int}" == "3"
    s_float = XTensor(torch.tensor(3.0))
    assert f"{s_float}" == "3.0"
    # a format spec should still apply to the extracted scalar, not to the
    # tensor's own repr
    assert f"{s_float:.2f}" == "3.00"


def test_repr_of_a_multi_dim_int_tensor_with_names_does_not_recurse():
    m = XTensor(torch.arange(6).reshape(2, 3), names=("row", "col"))
    assert "0, 1, 2" in repr(m)
    assert "3, 4, 5" in repr(m)


def test_zero_dim_tensor_index_behaves_like_the_equivalent_int():
    x = XTensor(
        torch.arange(4.0), names=("t",), coords={"t": {"spacing": 1.0}}
    )
    r = x[torch.tensor(1)]
    assert r.ndim == 0
    assert r.item() == 1.0
    assert r.item() == x[1].item()

    # on a multi-axis tensor, only the indexed axis drops
    y = XTensor(torch.arange(8.0).reshape(2, 4), names=("b", "t"))
    r2 = y[torch.tensor(1), :]
    assert r2.names == ("t",)
    assert r2.tolist() == [4.0, 5.0, 6.0, 7.0]

    # a genuine advanced (multi-element) index is unaffected
    r3 = y[torch.tensor([0, 1]), :]
    assert r3.shape == (2, 4)
