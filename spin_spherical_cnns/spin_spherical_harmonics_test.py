# coding=utf-8
# Copyright 2022 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for spin_spherical_harmonics.py."""

import functools
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
import tensorflow as tf

from spin_spherical_cnns import np_spin_spherical_harmonics
from spin_spherical_cnns import sphere_utils
from spin_spherical_cnns import spin_spherical_harmonics


@functools.lru_cache()
def _get_transformer():
  return spin_spherical_harmonics.SpinSphericalFourierTransformer(
      resolutions=[8, 16], spins=(0, -1, 1, -2))


class SpinSphericalHarmonicsTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters(dict(resolution=8, spin=-1),
                            dict(resolution=16, spin=0),
                            dict(resolution=16, spin=-1),
                            dict(resolution=16, spin=-2))
  def test_swsft_forward_matches_np(self, resolution, spin):
    transformer = _get_transformer()
    sphere = (jnp.linspace(-1, 1, resolution**2)
              .reshape((resolution, resolution)))
    coeffs_np = np_spin_spherical_harmonics.swsft_forward_naive(sphere, spin)
    coeffs_jax = transformer.swsft_forward(sphere, spin)

    self.assertAllClose(
        coeffs_jax, spin_spherical_harmonics.coefficients_to_matrix(coeffs_np))

  @parameterized.parameters(dict(resolution=8, spin=-1),
                            dict(resolution=16, spin=0),
                            dict(resolution=16, spin=1))
  def test_swsft_forward_matches_with_symmetry(self, resolution, spin):
    """Check whether the versions with and without symmetry match."""
    transformer = _get_transformer()
    sphere = (jnp.linspace(-1, 1, resolution**2)
              .reshape((resolution, resolution)))
    self.assertAllClose(transformer.swsft_forward(sphere, spin),
                        transformer.swsft_forward_with_symmetry(sphere, spin))

  def test_swsft_forward_validate_raises(self):
    """Check that swsft_forward() raises exception if constants are invalid."""
    transformer = spin_spherical_harmonics.SpinSphericalFourierTransformer(
        resolutions=[4, 8], spins=(0, 1))
    # Wrong resolution, right spin:
    resolution = 16
    sphere = (jnp.linspace(-1, 1, resolution**2)
              .reshape((resolution, resolution)))
    self.assertRaises(ValueError, transformer.swsft_forward,
                      sphere, spin=0)

    # Right resolution, wrong spin:
    resolution = 8
    sphere = (jnp.linspace(-1, 1, resolution**2)
              .reshape((resolution, resolution)))
    self.assertRaises(ValueError, transformer.swsft_forward,
                      sphere, spin=2)

  @parameterized.parameters(dict(num_coefficients=16, spin=-1),
                            dict(num_coefficients=16, spin=1),
                            dict(num_coefficients=64, spin=0),
                            dict(num_coefficients=64, spin=-2))
  def test_swsft_backward_matches_np(self, num_coefficients, spin):
    transformer = _get_transformer()
    coeffs_np = (jnp.linspace(-1, 1, num_coefficients) +
                 1j*jnp.linspace(0, 1, num_coefficients))
    coeffs_jax = spin_spherical_harmonics.coefficients_to_matrix(coeffs_np)
    sphere_np = np_spin_spherical_harmonics.swsft_backward_naive(coeffs_np,
                                                                 spin)
    sphere_jax = transformer.swsft_backward(coeffs_jax, spin)

    self.assertAllClose(sphere_np, sphere_jax)

  @parameterized.parameters(dict(num_coefficients=16, spin=1),
                            dict(num_coefficients=16, spin=0),
                            dict(num_coefficients=64, spin=-2))
  def test_swsft_backward_matches_with_symmetry(self, num_coefficients, spin):
    transformer = _get_transformer()
    coeffs = (jnp.linspace(-1, 1, num_coefficients) +
              1j*jnp.linspace(0, 1, num_coefficients))
    coeffs = spin_spherical_harmonics.coefficients_to_matrix(coeffs)
    sphere = transformer.swsft_backward(coeffs, spin)
    with_symmetry = transformer.swsft_backward_with_symmetry(coeffs, spin)

    self.assertAllClose(sphere, with_symmetry)

  def test_swsft_backward_validate_raises(self):
    """Check that swsft_backward() raises exception if constants are invalid."""
    transformer = spin_spherical_harmonics.SpinSphericalFourierTransformer(
        resolutions=[4, 8], spins=(0, 1))
    n_coeffs = 64  # Corresponds to resolution == 16.
    # Wrong ell_max, right spin:
    coeffs_np = jnp.linspace(-1, 1, n_coeffs) + 1j*jnp.linspace(0, 1, n_coeffs)
    coeffs_jax = spin_spherical_harmonics.coefficients_to_matrix(coeffs_np)
    self.assertRaises(ValueError, transformer.swsft_backward,
                      coeffs_jax, spin=1)

    n_coeffs = 16  # Corresponds to resolution == 8.
    # Right ell_max, wrong spin:
    coeffs_np = jnp.linspace(-1, 1, n_coeffs) + 1j*jnp.linspace(0, 1, n_coeffs)
    coeffs_jax = spin_spherical_harmonics.coefficients_to_matrix(coeffs_np)
    self.assertRaises(ValueError, transformer.swsft_backward,
                      coeffs_jax, spin=-1)

  def test_swsft_forward_spins_channels_matches_swsft_forward(self):
    transformer = _get_transformer()
    resolution = 16
    n_channels = 2
    spins = (0, 1)
    shape = (resolution, resolution, len(spins), n_channels)
    sphere_set = jnp.linspace(-1, 1, np.prod(shape)).reshape(shape)
    coefficients = transformer.swsft_forward_spins_channels(sphere_set,
                                                            spins)
    for channel in range(n_channels):
      for spin in spins:
        # Slices must match swsft_forward().
        sliced = transformer.swsft_forward(sphere_set[Ellipsis, spin, channel],
                                           spin)
        self.assertAllClose(coefficients[Ellipsis, spin, channel], sliced)

  def test_swsft_forward_spins_channels_matches_with_symmetry(self):
    transformer = _get_transformer()
    resolution = 16
    n_channels = 2
    spins = (0, 1)
    shape = (resolution, resolution, len(spins), n_channels)
    sphere_set = jnp.linspace(-1, 1, np.prod(shape)).reshape(shape)
    coefficients = transformer.swsft_forward_spins_channels(
        sphere_set, spins)
    with_symmetry = transformer.swsft_forward_spins_channels_with_symmetry(
        sphere_set, spins)

    self.assertAllClose(coefficients, with_symmetry)

  def test_swsft_backward_spins_channels_matches_swsft_backward(self):
    transformer = _get_transformer()
    ell_max = 7
    n_channels = 2
    spins = (0, 1)
    shape = (ell_max+1, 2*ell_max+1, n_channels, len(spins))
    coefficients = jnp.linspace(-1, 1, np.prod(shape)).reshape(shape)
    sphere = transformer.swsft_backward_spins_channels(coefficients,
                                                       spins)
    for channel in range(n_channels):
      for spin in spins:
        # Slices must match swsft_backward().
        sliced = transformer.swsft_backward(
            coefficients[Ellipsis, spin, channel], spin)
        self.assertAllClose(sphere[Ellipsis, spin, channel], sliced)

  @parameterized.parameters(2, 3)
  def test_coefficients_vector_to_matrix_has_right_shape(self, num_ell):
    num_coefficients = num_ell**2
    coefficients = jnp.ones(num_coefficients)
    matrix = spin_spherical_harmonics.coefficients_to_matrix(coefficients)
    target_shape = (num_ell, 2*num_ell - 1)

    self.assertEqual(matrix.shape, target_shape)

  @parameterized.parameters(0, 1, 2, 4, 7)
  def test_SpinSphericalFourierTransformer_wigner_delta_matches_np(self, ell):
    transformer = _get_transformer()
    wigner_delta = transformer._slice_wigner_deltas(
        ell, include_negative_m=True)[ell]
    wigner_delta_np = sphere_utils.compute_wigner_delta(ell)
    self.assertAllClose(wigner_delta, wigner_delta_np)

  @parameterized.parameters(dict(spin=0, ell=0),
                            dict(spin=0, ell=1),
                            dict(spin=1, ell=2),
                            dict(spin=-2, ell=3),
                            dict(spin=-1, ell=7))
  def test_SpinSphericalFourierTransformer_forward_constants_matches_np(self,
                                                                        spin,
                                                                        ell):
    transformer = _get_transformer()
    ell_max = transformer.swsft_forward_constants[spin].shape[0] - 1
    slice_ell = slice(ell_max - ell, ell_max + ell + 1)
    constants = transformer.swsft_forward_constants[spin][ell, slice_ell]
    constants_np = sphere_utils.swsft_forward_constant(spin, ell,
                                                       jnp.arange(-ell, ell+1))
    self.assertAllClose(constants, constants_np)

  def test_SpinSphericalFourierTransformer_get_set_attributes(self):
    original_transformer = _get_transformer()
    # This behavior is required to make SpinSphericalFourierTransformer
    # instances jit-able.
    attributes = original_transformer.get_attributes()
    transformer = (spin_spherical_harmonics.SpinSphericalFourierTransformer
                   .set_attributes(*attributes))
    for attribute, value in vars(transformer).items():
      self.assertIs(value, getattr(original_transformer, attribute))

  @parameterized.parameters(4, 8, 9)
  def test_fourier_transform_2d(self, dimension):
    """Ensures that our DFT implementation matches the FFT."""
    x = jnp.linspace(0, 1, dimension**2).reshape(dimension, dimension)
    fft = jnp.fft.fft2(x)
    ours = spin_spherical_harmonics._fourier_transform_2d(x)
    self.assertAllClose(fft, ours, atol=5e-6)


if __name__ == '__main__':
  tf.test.main()
