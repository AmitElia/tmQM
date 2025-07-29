import rdkit.Chem as Chem
import numpy as np
from scipy.special import softmax
from scipy.spatial.distance import pdist, squareform
from tqdm import tqdm
import os
import pandas as pd

def get_topological_distance_matrix(smiles: str):
    #get the topological distance between any pair of atoms, and save them using their mapping number index

    ps = Chem.SmilesParserParams()
    ps.removeHs = False
    ps.sanitize = False
    try:
        mol = Chem.MolFromSmiles(smiles, ps)
    except:  # Print the SMILES if there was an error in converting
        print(smiles)

    matrix = Chem.rdmolops.GetDistanceMatrix(mol)
    matrix_in_mapping_index = np.zeros_like(matrix)
    r, c = matrix.shape
    for i in range(r):
        for j in range(c):
            mapped_i, mapped_j = mol.GetAtomWithIdx(i).GetAtomMapNum() - 1, mol.GetAtomWithIdx(j).GetAtomMapNum() - 1
            matrix_in_mapping_index[mapped_i][mapped_j] = matrix[i][j]
    return matrix_in_mapping_index


def get_coords(blocks_raw):
    all_coords = []
    all_types = []
    all_charges = []
    for block in blocks_raw:
        lines = block.splitlines()
        # Skip metadata line(s) that aren't atomic data
        atom_lines = [line for line in lines if len(line.split()) == 4]
        atom_types = [line.split()[0] for line in atom_lines]
        coords = np.array([[float(x) for x in line.split()[1:4]] for line in atom_lines], dtype=float)
        header = lines[1]
        q_value = None
        for part in header.split("|"):
            part = part.strip()
            if part.startswith("q ="):
                q_value = int(part.split("=")[1].strip())
                break
        all_coords.append(coords)
        all_types.append(atom_types)
        all_charges.append(q_value)
    return all_types, all_coords, all_charges

def read_y(file_path: str):
    contents = pd.read_csv(file_path, sep=";")[:500]
    smiles_array = contents["SMILES"].to_numpy()
    topological_dist = [get_topological_distance_matrix(x) for x in smiles_array if pd.notna(x)]
    
    return topological_dist

def read_xyz(file_path: str):
    #file_path = os.path.join(os.getcwd(), file_name)
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            content = file.read()[:500]
             # Split by blank lines (lines with only whitespace or nothing)
        blocks_raw = [block.strip() for block in content.split('\n\n') if block.strip()]

        all_types, all_coords, all_charges = get_coords(blocks_raw)

        

    else:
        print(f"File not found: {file_path}")
    return all_types, all_coords, all_charges

def read_tmQM(dir_name: str, xyz_name: str, y_name: str):
    xyz_path = dir_name + xyz_name
    y_path = dir_name + y_name
    all_types, all_coords, all_charges = read_xyz(xyz_path)
    all_topological_dist = read_y(y_path)
    print(f"Parsed {len(all_types)} blocks.")
    print("First atom in first block:", all_types[0][0], all_coords[0][0], all_charges[0], all_topological_dist[0].shape)
    return all_types, all_coords, all_charges, all_topological_dist
    
    