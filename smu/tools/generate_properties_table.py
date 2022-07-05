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
"""Generates tables of fields of Molecule to got into papers."""

from absl import app

from smu import dataset_pb2

MOLECULE_FIELDS = [
    (r'bond\_topologies', '$>0$ struct',
     'Description of all bond topologies that adequately match this geometry'),
    (r'\quad .atoms', 'n * enum',
     'Atom types, including charge. See AtomType for names and numbers'),
    (r'\quad .bonds.atom\_a', r'$\geq 0$ * I', 'Index of one atom in bond'),
    (r'\quad .bonds.atom\_b', r'$\geq 0$ * I', 'Index of other atom in bond'),
    (r'\quad .bonds.bond\_type', r'$\geq 0$ * enum',
     'See BondType for names and numbers'),
    (r'\quad .smiles', 'S', 'SMILES canonicalied by RDKit'),
    (r'\quad .bond\_topology\_id', 'I',
     r'Unique ID for this topology. See bond\ _topology.csv'),
    (r'\quad .is\_starting\_topology', 'B',
     'Is this the topology used during geometry creation?'),
    (r'\quad .topology\_score', 'F',
     r'See Section~\ref{sec:topology_detection}'),
    (r'\quad .geometry\_score', 'F',
     r'See Section~\ref{sec:topology_detection}'),
    (r'molecule\_id', 'I', 'Unique ID for this molecule'),
    (r'duplicated\_by', 'I',
     'If this molecule did not proceed to full calculation because it was a ' +
     'duplicate, the molecule id that did proceed to full calculation'),
    (r'duplicate\_of', r'$\geq 0$ * I',
     'For molecule that proceeded to full calculation, the molecule ids of ' +
     'any other molecules that were duplicates of this one'),
    (r'fate', 'enum',
     'A simple categorical summary of how successful the calculations were.' +
     'See FateCategory for names and numbers'),
    (r'initial\_geometries', '$>0$ struct',
     (r'List of intial geometries that produced this optimized geometry. '
      r'May not have the same length as bond\_topologies or duplicate\ of, '
      r'see Section~\ref{sec:duplicates} for details')),
    (r'\quad .atom\_positions', 'n * V', '3D vector for each atom in .atoms'),
    (r'optimized\_geometries', '1 struct',
     'Single geometry used for all Stage 2 calculations'),
    (r'\quad .atom\_positions', 'n * V', '3D vector for each atom in .atoms'),
    (r'properties', 'struct',
     r'See Table~\ref{tab:properties_fields} for details'),
    (r'which\_database', 'enum', 'STANDARD(2) or COMPLETE(3)'),
]


def molecule_table(outf):
  """Prints out a molecule table.

  Args:
    outf: File handler.
  """
  print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%', file=outf)
  print(
      '%%% This section is automatically generated. Before editing, talk with Pat',
      file=outf)
  print(
      '% This is the contents of a table describing fields of Molecule',
      file=outf)
  for name, field_type, description in MOLECULE_FIELDS:
    print(f'{name:22s}& {field_type:10s} & \n    {description} \\\\', file=outf)
  print('%%% End automatically generated section', file=outf)


# Maps suffixes of field names to the level of theory used
LEVEL_DICT = {
    'pbe0d3_6_311gd': 'PBE0D3/6-311Gd',
    'pbe0_6_311gd': 'PBE0/6-311Gd',
    'pbe0_6_311gd_mrcc': 'PBE0D3/6-311Gd (MRCC)',
    'pbe0_6_311gd_orca': 'PBE0D3/6-311Gd (ORCA)',
    'pbe0_6_311gd_cat': 'PBE0/6-311Gd(CAT) (Turbomole)',
    'pbe0_6_311gd_cat_mrcc': 'PBE0/6-311Gd(CAT) (MRCC)',
    'pbe0_6_311gd_cat_orca': 'PBE0/6-311Gd(CAT) (ORCA)',
    'pbe0_aug_pc_1': 'PBE0/aug-pc-1',
    'hf_6_31gd': 'HF/6-31Gd',
    'b3lyp_6_31ppgdp': 'B3LYP/6-31++Gdp',
    'b3lyp_aug_pcs_1': 'B3LYP/aug-pcS-1',
    'pbe0_6_31ppgdp': 'PBE0/6-31++Gdp',
    'pbe0_aug_pcs_1': 'PBE0/aug-pcS-1',
    'hf_tzvp': 'HF/TZVP',
    'mp2_tzvp': 'MP2/TZVP',
    'cc2_tzvp': 'CC2/TZVP',
    'hf_3': 'HF/3',
    'mp2_3': 'MP2/3',
    'hf_4': 'HF/4',
    'mp2_4': 'MP2/4',
    'hf_34': 'HF/(34)',
    'mp2_34': 'MP2/(34)',
    'hf_cvtz': 'HF/CVTZ',
    'mp2ful_cvtz': 'MP2ful/CVTZ',
    'hf_2sp': 'HF/2sp',
    'mp2_2sp': 'MP2/2sp',
    'ccsd_2sp': 'CCSD/2sp',
    'ccsd_2sp_excess': 'CCSD/2sp',
    'ccsd_t_2sp': 'CCSD(T)/2sp',
    'hf_2sd': 'HF/2sd',
    'mp2_2sd': 'MP2/2sd',
    'ccsd_2sd': 'CCSD/2sd',
    'ccsd_t_2sd': 'CCSD(T)/2sd',
    'hf_3psd': 'HF/3Psd',
    'mp2_3psd': 'MP2/3Psd',
    'ccsd_3psd': 'CCSD/3Psd',
    'atomic_b5': 'ATOMIC-2, B5',
    'atomic_b5_um': 'ATOMIC-2, B5',
    'atomic_b5_um_ci': 'ATOMIC-2, B5',
    'atomic_b6': 'ATOMIC-2, B6',
    'atomic_b6_um': 'ATOMIC-2, B6',
    'atomic_b6_um_ci': 'ATOMIC-2, B6',
    'eccsd': '$E_{CCSD}$',
    'eccsd_um': '$E_{CCSD}$',
    'eccsd_um_ci': '$E_{CCSD}$',
}


def properties_table_line(name, field_descriptor):
  """Returns a line of a properties table.

  Args:
    name: Property name.
    field_descriptor: Describes field properties.
  """
  # Let's throw out a few special cases.
  if (name == 'compute_cluster_info' or
      name == 'symmetry_used_in_calculation' or
      name == 'gaussian_sanity_check' or name == 'calculation_statistics' or
      name == 'number_imaginary_frequencies' or
      name == 'number_of_optimization_runs'):
    return None

  if not field_descriptor.message_type:
    if field_descriptor.type == field_descriptor.TYPE_STRING:
      field_type = 'S'
    elif field_descriptor.type == field_descriptor.TYPE_INT32:
      field_type = 'I'
    else:
      raise ValueError(f'Unknown field type {field_descriptor}')
  elif field_descriptor.message_type.name == 'StringMolecularProperty':
    field_type = 'S'
  elif field_descriptor.message_type.name == 'ScalarMolecularProperty':
    field_type = 'F'
  elif field_descriptor.message_type.name == 'StringMolecularProperty':
    field_type = 'S'
  elif field_descriptor.message_type.name == 'MultiScalarMolecularProperty':
    field_type = 'TODO * F'
  elif field_descriptor.message_type.name == 'AtomicMolecularProperty':
    field_type = 'n * F'
  elif field_descriptor.message_type.name == 'Vector3DMolecularProperty':
    field_type = 'V'
  elif field_descriptor.message_type.name == 'Rank2MolecularProperty':
    field_type = 'T2'
  elif field_descriptor.message_type.name == 'Rank3MolecularProperty':
    field_type = 'T3'
  elif field_descriptor.message_type.name == 'NormalMode':
    field_type = '3 * n * n * V'
  elif field_descriptor.message_type.name == 'CalculationStatistics':
    # Internal only, ignoring
    return None
  else:
    raise ValueError(f'Unknown field type {field_descriptor.message_type.name}')

  avail_enum = field_descriptor.GetOptions().Extensions[
      dataset_pb2.availability]
  if avail_enum == dataset_pb2.AvailabilityEnum.INTERNAL_ONLY:
    return None
  elif avail_enum == dataset_pb2.AvailabilityEnum.STANDARD:
    availability = r'\checkmark'
  elif avail_enum == dataset_pb2.AvailabilityEnum.COMPLETE:
    availability = ''
  else:
    if name == 'errors.status':
      availability = r'\checkmark'
    elif name.startswith('errors.'):
      availability = ''
    else:
      raise ValueError(f'Unknown availiability {avail_enum}')

  level = ''
  for suffix, value in LEVEL_DICT.items():
    if name.endswith(suffix):
      level = value
      break

  armored_name = name.replace('_', r'\_')
  return f'{armored_name:56s}& {availability:10s} & {field_type:13s} & {level:30s}\\\\'


# Fields to consider adding: description, symbol, units
def properties_table(outf):
  """Prints a properties table."""
  print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%', file=outf)
  print(
      '% This is the contents of a table describing fields of Properties',
      file=outf)
  print(
      '%%% This section is automatically generated. Before editing, talk with Pat',
      file=outf)

  descriptors = sorted(
      dataset_pb2.Properties.DESCRIPTOR.fields, key=lambda d: d.name)
  for field_descriptor in descriptors:
    name = field_descriptor.name

    if field_descriptor.message_type and field_descriptor.message_type.name == 'Errors':
      # Errors has subfields so we deal with it separately
      for error_field_descriptor in field_descriptor.message_type.fields:
        if error_field_descriptor.name.startswith('error_'):
          continue
        line = properties_table_line('errors.' + error_field_descriptor.name,
                                     error_field_descriptor)
        if not line:
          raise ValueError(
              f'Did not understand error field {error_field_descriptor}')
        print(line, file=outf)
    else:
      line = properties_table_line(name, field_descriptor)
      if line:
        print(line, file=outf)

  print('%%% End automatically generated section', file=outf)


def main(argv):
  del argv  # Unused.
  with open('molecule_table.tex', 'w') as outf:
    molecule_table(outf)
  with open('properties_table.tex', 'w') as outf:
    properties_table(outf)


if __name__ == '__main__':
  app.run(main)
