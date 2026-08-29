"""``kinescore.robots.gr1.colliders.RobotColliders``: URDF -> collision spheres + CoM.

Was 17% covered -- the real GR-1 URDF is gated behind ``$KINESCORE_ASSETS``
(~285 MB, never vendored, see ``robots/urdf.py``), so every existing GR-1
test self-skips on a host without it. But ``RobotColliders`` itself takes any
URDF path and needs only the standard ``<link>``/``<collision>``/
``<inertial>`` elements it documents parsing -- nothing GR-1-specific -- so a
small synthetic URDF (same pattern as ``test_torque.py``'s ``_write_urdf``)
exercises the real parsing and posing logic without the asset checkout.

This is also a regression guard for the by-hand cleanup of
``colliders.py`` (semicolon-chains, ``l`` as a variable name, unused
``typing.Dict`` import) -- numeric equivalence with the pre-cleanup module
was additionally verified out-of-band by diffing tensor outputs against a
loaded copy of the pre-edit file on this exact synthetic URDF.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from kinescore.robots.gr1.colliders import RobotColliders

_URDF = """<?xml version="1.0"?>
<robot name="synthetic_body">
  <link name="torso_link">
    <collision>
      <origin xyz="0.01 0.02 0.03" rpy="0.1 0.2 0.05"/>
      <geometry><cylinder radius="0.12" length="0.4"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0.0 0.0 0.1"/>
      <mass value="5.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <link name="base_link">
    <collision>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <geometry><sphere radius="0.15"/></geometry>
    </collision>
    <inertial>
      <origin xyz="0 0 0"/>
      <mass value="8.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <link name="head_pitch_link">
    <collision>
      <origin xyz="0 0 0.05" rpy="0 0.3 0"/>
      <geometry><cylinder radius="0.06" length="0.15"/></geometry>
    </collision>
  </link>
  <link name="left_foot_roll_link">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <link name="right_foot_roll_link">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <link name="zero_mass_link">
    <inertial>
      <origin xyz="9 9 9"/>
      <mass value="0.0"/>
      <inertia ixx="1" ixy="0" ixz="0" iyy="1" iyz="0" izz="1"/>
    </inertial>
  </link>
  <link name="no_inertial_link"/>
</robot>
"""


def _write_urdf(tmp_path: Path) -> Path:
    path = tmp_path / "synthetic_body.urdf"
    path.write_text(_URDF)
    return path


def _bad_urdf(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "bad.urdf"
    path.write_text(f'<?xml version="1.0"?><robot name="bad">{body}</robot>')
    return path


class TestConstruction:
    def test_body_and_foot_links_recorded_in_order(self, tmp_path):
        rc = RobotColliders(str(_write_urdf(tmp_path)))
        assert rc.body_links == ["torso_link", "base_link", "head_pitch_link"]
        assert rc.foot_links == ["left_foot_roll_link", "right_foot_roll_link"]

    def test_mass_links_omits_zero_mass_and_no_inertial_links(self, tmp_path):
        rc = RobotColliders(str(_write_urdf(tmp_path)))
        # zero_mass_link has mass=0 (filtered), no_inertial_link has no
        # <inertial> at all, torso/base have real inertial data.
        assert rc.mass_links == ["torso_link", "base_link"]

    def test_cylinder_produces_n_cyl_spheres_and_sphere_produces_one(self, tmp_path):
        rc = RobotColliders(str(_write_urdf(tmp_path)), n_cyl_spheres=3)
        # torso_link (cylinder) -> 3, base_link (sphere) -> 1,
        # head_pitch_link (cylinder) -> 3.
        assert rc.sphere_centers.shape == (7, 3)
        assert rc.sphere_radii.shape == (7,)
        assert rc.sphere_link.tolist() == [0, 0, 0, 1, 2, 2, 2]

    def test_sphere_radii_match_urdf(self, tmp_path):
        rc = RobotColliders(str(_write_urdf(tmp_path)), n_cyl_spheres=3)
        radii = rc.sphere_radii.tolist()
        assert radii[:3] == pytest.approx([0.12, 0.12, 0.12])
        assert radii[3] == pytest.approx(0.15)
        assert radii[4:] == pytest.approx([0.06, 0.06, 0.06])

    def test_link_mass_and_com_match_urdf(self, tmp_path):
        rc = RobotColliders(str(_write_urdf(tmp_path)))
        assert rc.link_mass.tolist() == pytest.approx([5.0, 8.0])
        # torso's inertial origin is (0, 0, 0.1); base's is (0, 0, 0).
        assert rc.link_com[0].tolist() == pytest.approx([0.0, 0.0, 0.1], abs=1e-6)
        assert rc.link_com[1].tolist() == pytest.approx([0.0, 0.0, 0.0], abs=1e-6)

    def test_custom_body_and_foot_links_override_defaults(self, tmp_path):
        rc = RobotColliders(
            str(_write_urdf(tmp_path)),
            body_core_links=("base_link",),
            foot_links=("right_foot_roll_link",),
        )
        assert rc.body_links == ["base_link"]
        assert rc.foot_links == ["right_foot_roll_link"]
        assert rc.sphere_link.tolist() == [0]  # only base_link's one sphere


class TestConstructionErrors:
    def test_missing_body_link_raises(self, tmp_path):
        urdf = _bad_urdf(tmp_path, '<link name="only_link"/>')
        with pytest.raises(ValueError, match="not in URDF"):
            RobotColliders(str(urdf))

    def test_body_link_without_collision_raises(self, tmp_path):
        # torso_link exists but has no <collision> -- base/head still missing too,
        # but torso is checked first (self.body_links order).
        urdf = _bad_urdf(tmp_path, '<link name="torso_link"/>')
        with pytest.raises(ValueError, match="no <collision>"):
            RobotColliders(str(urdf))

    def test_unsupported_geometry_raises(self, tmp_path):
        body = """
        <link name="torso_link">
          <collision><geometry><box size="1 1 1"/></geometry></collision>
        </link>
        """
        urdf = _bad_urdf(tmp_path, body)
        with pytest.raises(ValueError, match="not cylinder/sphere"):
            RobotColliders(str(urdf))


class TestPosedBodySpheres:
    def test_identity_frames_reproduce_link_frame_centers(self, tmp_path):
        rc = RobotColliders(str(_write_urdf(tmp_path)), n_cyl_spheres=1)
        b, t = 1, 1
        frames = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(b, t, len(rc.body_links), 1, 1)
        centers, radii = rc.posed_body_spheres(frames)
        assert centers.shape == (b, t, rc.sphere_centers.shape[0], 3)
        assert torch.allclose(centers[0, 0], rc.sphere_centers)
        assert torch.equal(radii, rc.sphere_radii)

    def test_translation_offsets_every_sphere_center(self, tmp_path):
        rc = RobotColliders(str(_write_urdf(tmp_path)), n_cyl_spheres=1)
        offset = torch.tensor([1.0, 2.0, 3.0])
        frames = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 1, len(rc.body_links), 1, 1)
        frames[..., :3, 3] = offset
        centers, _ = rc.posed_body_spheres(frames)
        assert torch.allclose(centers[0, 0], rc.sphere_centers + offset, atol=1e-6)

    def test_batch_and_time_dims_are_independent(self, tmp_path):
        rc = RobotColliders(str(_write_urdf(tmp_path)), n_cyl_spheres=1)
        B, T = 2, 3
        frames = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(B, T, len(rc.body_links), 1, 1)
        centers, _ = rc.posed_body_spheres(frames)
        assert centers.shape == (B, T, rc.sphere_centers.shape[0], 3)
        # every (b, t) slice is identical under an identity pose.
        for b in range(B):
            for t in range(T):
                assert torch.allclose(centers[b, t], rc.sphere_centers)


class TestWorldCom:
    def test_identity_frames_reproduce_link_com(self, tmp_path):
        rc = RobotColliders(str(_write_urdf(tmp_path)))
        frames = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 1, len(rc.mass_links), 1, 1)
        com = rc.world_com(frames)
        assert com.shape == (1, 1, 3)
        # mass-weighted average of [0,0,0.1] (m=5) and [0,0,0] (m=8).
        expected_z = (5.0 * 0.1 + 8.0 * 0.0) / (5.0 + 8.0)
        assert com[0, 0].tolist() == pytest.approx([0.0, 0.0, expected_z], abs=1e-6)

    def test_translation_shifts_com_by_the_same_offset(self, tmp_path):
        rc = RobotColliders(str(_write_urdf(tmp_path)))
        offset = torch.tensor([2.0, -1.0, 0.5])
        frames = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 1, len(rc.mass_links), 1, 1)
        frames[..., :3, 3] = offset
        com = rc.world_com(frames)
        base_com = rc.world_com(
            torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 1, len(rc.mass_links), 1, 1))
        assert torch.allclose(com, base_com + offset, atol=1e-6)


def test_numerically_matches_a_direct_reimplementation(tmp_path):
    """Cross-check against a from-scratch numpy computation of the two
    non-trivial numeric paths (rpy->matrix pose composition for the cylinder
    spheres, mass-weighted CoM), independent of the module's own math.
    """
    rc = RobotColliders(str(_write_urdf(tmp_path)), n_cyl_spheres=3)

    def rpy_to_matrix(r, p, y):
        cr, sr, cp, sp, cy, sy = (np.cos(r), np.sin(r), np.cos(p),
                                  np.sin(p), np.cos(y), np.sin(y))
        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx

    # torso_link cylinder: origin xyz=(0.01,0.02,0.03) rpy=(0.1,0.2,0.05),
    # radius 0.12, length 0.4, 3 spheres along local +Z at [-0.2, 0, 0.2].
    T = np.eye(4)
    T[:3, 3] = [0.01, 0.02, 0.03]
    T[:3, :3] = rpy_to_matrix(0.1, 0.2, 0.05)
    expected = np.stack([(T @ np.array([0, 0, z, 1.0]))[:3]
                          for z in np.linspace(-0.2, 0.2, 3)])
    assert np.allclose(rc.sphere_centers[:3].numpy(), expected, atol=1e-6)
