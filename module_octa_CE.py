import math
import numpy as np
from itertools import product
from scipy.spatial.transform import Rotation as R
from functools import lru_cache
import json

def posreader(PosName='POSCAR'):
    POS = {}
    with open(PosName, 'r') as Fid:
        POS['CellName'] = Fid.readline().strip()
        POS['LattConst'] = float(Fid.readline().split()[0])
        
        POS['Base'] = []
        for _ in range(3):
            line = Fid.readline().split()
            POS['Base'].append([float(x) * POS['LattConst'] for x in line[:3]])
        
        POS['EleName'] = Fid.readline().split()
        POS['EleNum'] = len(POS['EleName'])
        
        atom_nums = Fid.readline().split()
        POS['AtomNum'] = [int(x) for x in atom_nums]
        POS['AtomSum'] = sum(POS['AtomNum'])
        
        line = Fid.readline().split()
        FL = line[0][0].upper()
        
        if FL == 'S':
            POS['IsSel'] = 1
            POS['SelMat'] = [['X']*3 for _ in range(POS['AtomSum'])]
            line = Fid.readline().split()
            FL = line[0][0].upper()
        else:
            POS['IsSel'] = 0
        
        POS['LatType'] = 'Direct' if FL == 'D' else 'Cartesian'
        POS['LattPnt'] = []
        
        if POS['LatType'] == 'Direct':
            for i in range(POS['AtomSum']):
                line = Fid.readline().split()
                POS['LattPnt'].append([float(line[j]) for j in range(3)])
                if POS['IsSel']:
                    POS['SelMat'][i] = [line[j+3] for j in range(3)]
        else:
            BaseInv = np.linalg.inv(POS['Base'])
            for i in range(POS['AtomSum']):
                line = Fid.readline().split()
                cart_coord = [float(line[j]) for j in range(3)]
                POS['LattPnt'].append(list(np.dot(BaseInv, cart_coord)))
                if POS['IsSel']:
                    POS['SelMat'][i] = [line[j+3] for j in range(3)]
    
    return POS

def poswriter(PosName, POS):
    with open(PosName, 'w') as Fid:
        Fid.write(f"{POS['CellName']}\n")
        Fid.write(f"{POS['LattConst']:.6f}\n")
        
        for base_vec in POS['Base']:
            Fid.write(f"{base_vec[0]:.6f} {base_vec[1]:.6f} {base_vec[2]:.6f}\n")
        
        Fid.write(' '.join(POS['EleName']) + '\n')
        Fid.write(' '.join(map(str, POS['AtomNum'])) + '\n')
        
        if POS['IsSel']:
            Fid.write('Selective Dynamics\n')
        
        Fid.write(f"{POS['LatType']}\n")
        
        for i, pnt in enumerate(POS['LattPnt']):
            Fid.write(f"{pnt[0]:.6f} {pnt[1]:.6f} {pnt[2]:.6f}")
            if POS['IsSel']:
                Fid.write(f" {POS['SelMat'][i][0]} {POS['SelMat'][i][1]} {POS['SelMat'][i][2]}")
            Fid.write('\n')

def dismatcreate(POS):
    N = POS['AtomSum']
    POS['dismat'] = np.zeros((N, N))
    
    for i in range(N):
        for j in range(N):
            delta = np.array(POS['LattPnt'][i]) - np.array(POS['LattPnt'][j])
            delta = np.where(delta > 0.5, delta - 1, delta)
            delta = np.where(delta <= -0.5, delta + 1, delta)
            delta = np.abs(delta)
            
            cart_delta = np.dot(delta, POS['Base'])
            POS['dismat'][i][j] = np.linalg.norm(cart_delta)
    
    return POS

def dismatswap(dismat, Ind1, Ind2):
    dismat[[Ind1, Ind2]] = dismat[[Ind2, Ind1]]
    dismat[:, [Ind1, Ind2]] = dismat[:, [Ind2, Ind1]]
    return dismat

def generate_full_octahedral_symmetries():
    vecs = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    
    group_o = R.create_group('O')
    symmetries = []
    
    for r in group_o:
        perm = []
        for v in vecs:
            rotated = r.apply(v)
            idx = np.argmin(np.sum((vecs - rotated)**2, axis=1))
            perm.append(idx)
        symmetries.append(perm)
    
    inversion_map = [1, 0, 3, 2, 5, 4]
    for rot_perm in symmetries[:24]:
        inverted_perm = [inversion_map[rot_perm[i]] for i in range(6)]
        symmetries.append(inverted_perm)
    
    unique_symmetries = []
    for sym in symmetries:
        if sym not in unique_symmetries:
            unique_symmetries.append(sym)
    
    return unique_symmetries

def apply_symmetry(cluster, symmetry_op):
    core = cluster[0]
    b_positions = [cluster[1+i] for i in symmetry_op]
    o_positions = [cluster[7+i] for i in symmetry_op]
    
    if len(cluster) > 13:
        a_positions = [cluster[13 + (i % 8)] for i in symmetry_op[:8]]
        return [core] + b_positions + o_positions + a_positions
    
    return [core] + b_positions + o_positions

@lru_cache(maxsize=10000)
def get_canonical_form(cluster_tuple):
    symmetries = generate_full_octahedral_symmetries()
    equivalent_clusters = [tuple(apply_symmetry(list(cluster_tuple), sym)) for sym in symmetries]
    return min(equivalent_clusters)

def generate_reference_clusters(atom_ind_group):
    b_types = atom_ind_group[1]
    o_types = atom_ind_group[2]
    a_types = atom_ind_group[0] if len(atom_ind_group) > 0 else []
    
    cluster_set = set()
    symmetries = generate_full_octahedral_symmetries()
    
    if a_types:
        for core_type in b_types:
            for b_combo in product(b_types, repeat=6):
                for o_combo in product(o_types, repeat=6):
                    for a_combo in product(a_types, repeat=8):
                        cluster = [core_type] + list(b_combo) + list(o_combo) + list(a_combo)
                        canonical = get_canonical_form(tuple(cluster))
                        cluster_set.add(canonical)
    else:
        for core_type in b_types:
            for b_combo in product(b_types, repeat=6):
                for o_combo in product(o_types, repeat=6):
                    cluster = [core_type] + list(b_combo) + list(o_combo)
                    canonical = get_canonical_form(tuple(cluster))
                    cluster_set.add(canonical)
    
    return list(cluster_set)

def create_atom_type_mapping(poscar):
    atom_types = []
    for atom_type, count in enumerate(poscar['AtomNum']):
        atom_types.extend([atom_type] * count)
    return atom_types

def find_positioned_neighbors(core_idx, poscar, atom_types, atom_ind_group):
    positions = [np.array(pos) for pos in poscar['LattPnt']]
    distances = poscar['dismat'][core_idx]
    core_pos = positions[core_idx]
    
    b_range, o_range = (3.8, 4.2), (1.8, 2.2)
    a_range = (3.0, 4.0)
    
    positioned_neighbors = {
        'b_positions': [None] * 6,
        'o_positions': [None] * 6,
        'a_positions': [None] * 8
    }
    
    candidates = {'b': [], 'o': [], 'a': []}
    
    for i, dist in enumerate(distances):
        if i == core_idx:
            continue
        
        atom_type = atom_types[i]
        vec = positions[i] - core_pos
        vec = np.where(vec > 0.5, vec - 1, vec)
        vec = np.where(vec <= -0.5, vec + 1, vec)
        
        if atom_type in atom_ind_group[1] and b_range[0] <= dist <= b_range[1]:
            candidates['b'].append((atom_type, vec))
        elif atom_type in atom_ind_group[2] and o_range[0] <= dist <= o_range[1]:
            candidates['o'].append((atom_type, vec))
        elif len(atom_ind_group) > 0 and atom_type in atom_ind_group[0] and a_range[0] <= dist <= a_range[1]:
            candidates['a'].append((atom_type, vec))
    
    for cand in candidates['b'][:6]:
        assign_b_o_direction(cand, positioned_neighbors['b_positions'])
    
    for cand in candidates['o'][:6]:
        assign_b_o_direction(cand, positioned_neighbors['o_positions'])
    
    for cand in candidates['a'][:8]:
        assign_a_direction(cand, positioned_neighbors['a_positions'])
    
    return positioned_neighbors

def assign_b_o_direction(candidate, position_array):
    atom_type, vec = candidate
    x, y, z = vec
    abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)
    
    if abs_z > max(abs_x, abs_y) * 1.2:
        idx = 4 if z > 0 else 5
    else:
        if x >= 0 and y >= 0:
            idx = 0
        elif x < 0 and y < 0:
            idx = 1
        elif x >= 0 and y < 0:
            idx = 2
        else:
            idx = 3
    
    if position_array[idx] is None:
        position_array[idx] = atom_type

def assign_a_direction(candidate, position_array):
    atom_type, vec = candidate
    x, y, z = vec
    
    idx = (1 if x > 0 else 0) + (2 if y > 0 else 0) + (4 if z > 0 else 0)
    
    if position_array[idx] is None:
        position_array[idx] = atom_type

def generate_single_positioned_cluster(core_type, positioned_neighbors):
    b_positions = positioned_neighbors['b_positions']
    o_positions = positioned_neighbors['o_positions']
    a_positions = positioned_neighbors.get('a_positions', [])
    
    valid_b = sum(1 for x in b_positions if x is not None)
    valid_o = sum(1 for x in o_positions if x is not None)
    
    if valid_b < 4 or valid_o < 4:
        return None
    
    def fill_none_values(positions):
        filled = positions.copy()
        valid_types = [x for x in positions if x is not None]
        if valid_types:
            most_common = max(set(valid_types), key=valid_types.count)
            filled = [most_common if x is None else x for x in filled]
        return filled
    
    filled_b = fill_none_values(b_positions)
    filled_o = fill_none_values(o_positions)
    
    if a_positions and any(x is not None for x in a_positions):
        filled_a = fill_none_values(a_positions)
        return [core_type] + filled_b + filled_o + filled_a
    
    return [core_type] + filled_b + filled_o

def load_reference_clusters(file_path):
    with open(file_path, 'r') as f:
        clusters = json.load(f)
    return [tuple(c) for c in clusters]

def count_clusters_in_structure(file_poscar, atom_ind_group, reference_file):
    reference_clusters = load_reference_clusters(reference_file)
    
    poscar = posreader(file_poscar)
    poscar = dismatcreate(poscar)
    atom_types = create_atom_type_mapping(poscar)
    b_site_indices = [i for i, t in enumerate(atom_types) if t in atom_ind_group[1]]
    
    cluster_counts = [0] * len(reference_clusters)
    
    for b_idx in b_site_indices:
        core_type = atom_types[b_idx]
        positioned_neighbors = find_positioned_neighbors(b_idx, poscar, atom_types, atom_ind_group)
        cluster = generate_single_positioned_cluster(core_type, positioned_neighbors)
        
        if cluster:
            canonical = get_canonical_form(tuple(cluster))
            try:
                ref_idx = reference_clusters.index(canonical)
                cluster_counts[ref_idx] += 1
            except ValueError:
                pass
    
    return cluster_counts, reference_clusters

def count_cluster(file_poscar, atom_ind_group, cluster_dir, verbose=False):
    cluster_counts, reference_clusters = count_clusters_in_structure(
        file_poscar, atom_ind_group, reference_file=cluster_dir
    )
    
    if cluster_counts:
        poscar = posreader(file_poscar)
        atom_types = create_atom_type_mapping(poscar)
        
        ti_count = sum(1 for atom_type in atom_types if atom_type == 1)
        o_count = sum(1 for atom_type in atom_types if atom_type == 3)
        
        result_counts = [ti_count, o_count] + cluster_counts
        
        if verbose:
            print(f"\nRESULTS (Oh Symmetry):")
            print(f"Ti atom count: {ti_count}")
            print(f"O atom count: {o_count}")
            print(f"Reference clusters generated: {len(reference_clusters)}")
            print(f"Total positioned clusters found: {sum(cluster_counts)}")
            print(f"Non-zero cluster types: {sum(1 for c in cluster_counts if c > 0)}")
    else:
        result_counts = None
    
    return result_counts, reference_clusters