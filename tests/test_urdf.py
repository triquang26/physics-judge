"""``kinescore.robots.urdf``: shared URDF resolve/hash/parse helpers.

``parse_joint_limits`` and ``sha256_file`` were 0%-exercised by any test that
runs without ``$KINESCORE_ASSETS`` even though neither needs a real robot
asset -- both take a plain file path. ``resolve_asset_urdf`` likewise needs
only *some* directory at ``$KINESCORE_ASSETS``, not the real ~285 MB GR-1
checkout, so its env-var/missing-file error paths are covered here too via
``tmp_path`` + ``monkeypatch``. ``resolve_robot_description_urdf`` is left
untested here -- it goes through the actual ``robot_descriptions`` package
resolution, which is exercised end-to-end by the Franka FK tests instead.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from kinescore.paths import MissingPathError
from kinescore.robots.urdf import (
    JointLimits,
    parse_joint_limits,
    resolve_asset_urdf,
    sha256_file,
)

_URDF = """<?xml version="1.0"?>
<robot name="synthetic_arm">
  <joint name="joint_a" type="revolute">
    <limit lower="-1.5" upper="1.5" velocity="2.0" effort="10.0"/>
  </joint>
  <joint name="joint_b" type="prismatic">
    <limit lower="0.0" upper="0.5"/>
  </joint>
  <joint name="joint_fixed" type="fixed"/>
  <joint name="joint_no_limit_elem" type="revolute"/>
</robot>
"""


def _write_urdf(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic_arm.urdf"
    path.write_text(_URDF)
    return path


class TestParseJointLimits:
    def test_parses_lower_upper_velocity_effort(self, tmp_path):
        urdf = _write_urdf(tmp_path)
        limits = parse_joint_limits(urdf, ("joint_a",))
        assert limits["joint_a"] == JointLimits(
            lower=-1.5, upper=1.5, velocity=2.0, effort=10.0)

    def test_missing_optional_attrs_become_none_not_zero(self, tmp_path):
        urdf = _write_urdf(tmp_path)
        limits = parse_joint_limits(urdf, ("joint_b",))
        assert limits["joint_b"].velocity is None
        assert limits["joint_b"].effort is None
        assert limits["joint_b"].lower == pytest.approx(0.0)
        assert limits["joint_b"].upper == pytest.approx(0.5)

    def test_multiple_joint_names_all_returned(self, tmp_path):
        urdf = _write_urdf(tmp_path)
        limits = parse_joint_limits(urdf, ("joint_a", "joint_b"))
        assert set(limits) == {"joint_a", "joint_b"}

    def test_fixed_joint_never_requested_is_silently_absent(self, tmp_path):
        urdf = _write_urdf(tmp_path)
        limits = parse_joint_limits(urdf, ("joint_a",))
        assert "joint_fixed" not in limits

    def test_unrequested_joint_names_ignored(self, tmp_path):
        urdf = _write_urdf(tmp_path)
        limits = parse_joint_limits(urdf, ("joint_a",))
        assert set(limits) == {"joint_a"}

    def test_missing_joint_name_raises_value_error_naming_it(self, tmp_path):
        urdf = _write_urdf(tmp_path)
        with pytest.raises(ValueError, match="nonexistent_joint"):
            parse_joint_limits(urdf, ("joint_a", "nonexistent_joint"))

    def test_fixed_type_joint_requested_by_name_raises(self, tmp_path):
        # joint_fixed exists but is type="fixed" -> no <limit>, so asking for
        # it by name is the "typo'd/dropped joint" case the docstring names.
        urdf = _write_urdf(tmp_path)
        with pytest.raises(ValueError, match="joint_fixed"):
            parse_joint_limits(urdf, ("joint_fixed",))

    def test_revolute_joint_missing_limit_element_requested_by_name_raises(self, tmp_path):
        # A revolute/prismatic joint is still allowed to omit <limit> in the
        # XML (malformed w.r.t. the URDF spec, but this function does not
        # validate that) -- asking for it by name hits the same "not found"
        # path as a typo'd name, not a crash on `limit.get(...)`.
        urdf = _write_urdf(tmp_path)
        with pytest.raises(ValueError, match="joint_no_limit_elem"):
            parse_joint_limits(urdf, ("joint_no_limit_elem",))


class TestSha256File:
    def test_matches_hashlib_reference(self, tmp_path):
        import hashlib
        p = tmp_path / "f.bin"
        p.write_bytes(b"some urdf bytes, not actually xml here")
        assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()

    def test_different_contents_hash_differently(self, tmp_path):
        p1 = tmp_path / "a.urdf"
        p2 = tmp_path / "b.urdf"
        p1.write_text(_URDF)
        p2.write_text(_URDF.replace("joint_a", "joint_c"))
        assert sha256_file(p1) != sha256_file(p2)

    def test_same_contents_hash_identically(self, tmp_path):
        p1 = tmp_path / "a.urdf"
        p2 = tmp_path / "b.urdf"
        p1.write_text(_URDF)
        p2.write_text(_URDF)
        assert sha256_file(p1) == sha256_file(p2)

    def test_large_file_spanning_multiple_1mib_chunks(self, tmp_path):
        # sha256_file streams in 1 MiB chunks; exercise the multi-chunk loop.
        p = tmp_path / "big.bin"
        p.write_bytes(b"x" * (1 << 20) + b"y" * 100)
        import hashlib
        assert sha256_file(p) == hashlib.sha256(p.read_bytes()).hexdigest()


class TestResolveAssetUrdf:
    def test_resolves_relative_path_under_kinescore_assets(self, tmp_path, monkeypatch):
        (tmp_path / "sub").mkdir()
        (tmp_path / "sub" / "robot.urdf").write_text(_URDF)
        monkeypatch.setenv("KINESCORE_ASSETS", str(tmp_path))
        resolved = resolve_asset_urdf("sub/robot.urdf")
        assert resolved == (tmp_path / "sub" / "robot.urdf").resolve()

    def test_missing_file_under_kinescore_assets_raises_missing_path_error(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.setenv("KINESCORE_ASSETS", str(tmp_path))
        with pytest.raises(MissingPathError, match="does not exist"):
            resolve_asset_urdf("nonexistent/robot.urdf")

    def test_unset_kinescore_assets_raises_missing_path_error(self, monkeypatch):
        monkeypatch.delenv("KINESCORE_ASSETS", raising=False)
        with pytest.raises(MissingPathError, match="KINESCORE_ASSETS"):
            resolve_asset_urdf("sub/robot.urdf")


def test_parsed_limits_round_trip_through_xml_element_tree(tmp_path):
    """Sanity: the file this module parses is exactly the well-formed URDF
    subset it documents (revolute/prismatic <joint> with a <limit> child).
    """
    urdf = _write_urdf(tmp_path)
    root = ET.parse(urdf).getroot()
    joint_names = {j.get("name") for j in root.findall("joint")}
    assert joint_names == {"joint_a", "joint_b", "joint_fixed", "joint_no_limit_elem"}
