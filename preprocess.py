from collections import defaultdict
import numpy as np
from rdkit import Chem
import torch
import pickle
from molbert import SmilesTokenizer

atom_dict = defaultdict(lambda: len(atom_dict))
bond_dict = defaultdict(lambda: len(bond_dict))
fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
edge_dict = defaultdict(lambda: len(edge_dict))
radius=1

smiles_tokenizer = None
max_smiles_length = 256

# Feature dictionaries (atom/bond/fingerprint/edge) are populated exclusively
# from the training subset and then frozen; validation and test data are encoded
# with these frozen dictionaries (unseen patterns map to an <unk> index) and
# never extend the vocabularies.
_feature_dicts_frozen = False
_unk_indices = {}


def freeze_feature_dicts():
    global _feature_dicts_frozen
    for name, d in (('atom', atom_dict), ('bond', bond_dict),
                    ('fingerprint', fingerprint_dict), ('edge', edge_dict)):
        if '<unk>' not in d:
            d['<unk>'] = len(d)
        _unk_indices[name] = d['<unk>']
    _feature_dicts_frozen = True


def _dict_index(d, key, name):
    if key in d:
        return d[key]
    if _feature_dicts_frozen:
        return _unk_indices[name]
    d[key] = len(d)
    return d[key]

def dump_dictionary(dictionary, filename):
    with open(filename, 'wb') as f:
        pickle.dump(dict(dictionary), f)

def init_smiles_tokenizer(smiles_list=None, max_length=256):
    global smiles_tokenizer, max_smiles_length
    max_smiles_length = max_length
    
    smiles_tokenizer = SmilesTokenizer(max_length=max_length)
    
    if smiles_list is not None:
        smiles_tokenizer.build_vocab(smiles_list)
    
    return smiles_tokenizer

def tokenize_smiles(smiles, max_length=None):
    global smiles_tokenizer, max_smiles_length
    
    if max_length is None:
        max_length = max_smiles_length
    
    if smiles_tokenizer is None:
        raise ValueError("Tokenizer not initialized. Call init_smiles_tokenizer() first.")
    
    encoded = smiles_tokenizer.encode(smiles, max_length=max_length)
    return encoded['input_ids'], encoded['attention_mask']

def tokenize_smiles_batch(smiles_list, max_length=None):
    global smiles_tokenizer, max_smiles_length
    
    if max_length is None:
        max_length = max_smiles_length
    
    if smiles_tokenizer is None:
        raise ValueError("Tokenizer not initialized. Call init_smiles_tokenizer() first.")
    
    encoded = smiles_tokenizer.batch_encode(smiles_list, max_length=max_length)
    return encoded['input_ids'], encoded['attention_mask']

def save_tokenizer(path, dataname):
    global smiles_tokenizer
    if smiles_tokenizer is not None:
        import json
        vocab_dict = dict(smiles_tokenizer.vocab)
        with open(path + dataname + '-tokenizer_vocab.json', 'w') as f:
            json.dump(vocab_dict, f)

def load_tokenizer(path, dataname):
    global smiles_tokenizer
    import json
    try:
        with open(path + dataname + '-tokenizer_vocab.json', 'r') as f:
            vocab_dict = json.load(f)
        smiles_tokenizer = SmilesTokenizer(vocab=vocab_dict)
        return smiles_tokenizer
    except FileNotFoundError:
        return None


def molecule_identity(smiles):
    """Return a canonical molecular identity used to group chemically identical records.

    Different SMILES spellings of the same molecule share the same canonical SMILES,
    so grouping by this key prevents one molecule from being split across subsets.
    Unparsable SMILES fall back to the raw string so they are never merged with valid records.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return Chem.MolToSmiles(mol, canonical=True)


def augment_smiles(smiles, num_augments=5, augmentation_types=None):
    """
    SMILES数据增强
    
    Args:
        smiles: 原始SMILES字符串
        num_augments: 增强数量
        augmentation_types: 增强类型列表，可选:
            - 'random': 随机SMILES（原子顺序随机化）
            - 'kekulize': Kekule化SMILES
            - 'canonical': 标准SMILES
            - 'isomeric': 非异构体SMILES
    
    Returns:
        list: 增强后的SMILES列表
    """
    if augmentation_types is None:
        augmentation_types = ['random', 'canonical']
    
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [smiles]
    
    augmented = [smiles]
    
    for aug_type in augmentation_types:
        if aug_type == 'random':
            for _ in range(min(num_augments - len(augmented) + 1, num_augments)):
                try:
                    random_smiles = Chem.MolToSmiles(mol, doRandom=True)
                    if random_smiles and random_smiles not in augmented:
                        augmented.append(random_smiles)
                except:
                    pass
        
        elif aug_type == 'kekulize':
            try:
                kekulized = Chem.MolToSmiles(mol, kekuleSmiles=True)
                if kekulized and kekulized not in augmented:
                    augmented.append(kekulized)
            except:
                pass
        
        elif aug_type == 'canonical':
            try:
                canonical = Chem.MolToSmiles(mol, canonical=True)
                if canonical and canonical not in augmented:
                    augmented.append(canonical)
            except:
                pass
        
        elif aug_type == 'isomeric':
            try:
                non_isomeric = Chem.MolToSmiles(mol, isomericSmiles=False)
                if non_isomeric and non_isomeric not in augmented:
                    augmented.append(non_isomeric)
            except:
                pass
    
    return augmented[:num_augments]


def augment_smiles_batch(smiles_list, properties, num_augments=5, augmentation_types=None):
    """
    批量SMILES数据增强
    
    Args:
        smiles_list: SMILES列表
        properties: 对应的属性值列表
        num_augments: 每个SMILES的增强数量
        augmentation_types: 增强类型列表
    
    Returns:
        tuple: (增强后的SMILES列表, 增强后的属性列表)
    """
    augmented_smiles = []
    augmented_properties = []
    
    for smiles, prop in zip(smiles_list, properties):
        aug_smiles = augment_smiles(smiles, num_augments, augmentation_types)
        augmented_smiles.extend(aug_smiles)
        augmented_properties.extend([prop] * len(aug_smiles))
    
    return augmented_smiles, augmented_properties


class SMILESAugmentor:
    """SMILES数据增强器类"""
    
    def __init__(self, num_augments=5, augmentation_types=None, augment_probability=0.5):
        """
        Args:
            num_augments: 每个分子的增强数量
            augmentation_types: 增强类型列表
            augment_probability: 训练时应用增强的概率
        """
        self.num_augments = num_augments
        self.augmentation_types = augmentation_types or ['random', 'canonical']
        self.augment_probability = augment_probability
    
    def __call__(self, smiles, training=True):
        """
        应用数据增强
        
        Args:
            smiles: SMILES字符串
            training: 是否在训练模式
        
        Returns:
            str: 增强后的SMILES（训练时随机选择一个增强版本）
        """
        if not training or np.random.random() > self.augment_probability:
            return smiles
        
        augmented = augment_smiles(smiles, self.num_augments, self.augmentation_types)
        return np.random.choice(augmented)
    
    def augment_dataset(self, smiles_list, properties, training=True):
        """
        增强整个数据集
        
        Args:
            smiles_list: SMILES列表
            properties: 属性列表
            training: 是否在训练模式
        
        Returns:
            tuple: (增强后的SMILES列表, 属性列表)
        """
        if not training:
            return smiles_list, properties
        
        return augment_smiles_batch(
            smiles_list, properties, 
            self.num_augments, 
            self.augmentation_types
        )


def create_dataset_augment_after_split(filename, path, dataname, split_ratio=0.9,
                                       num_augments=3, augmentation_types=None,
                                       seed=1234):
    """
    Split by canonical molecular identity first, then apply SMILES augmentation
    exclusively to the training subset, so that augmented copies of a molecule
    never appear in the validation set. The tokenizer and feature dictionaries
    are built from the training subset only and frozen before the validation
    subset is encoded.

    Returns:
        tuple: (train_dataset, dev_dataset)
    """
    dir_dataset = path
    print(filename)

    with open(dir_dataset + filename, 'r') as f:
        _ = f.readline().strip().split()
        data_original = f.read().strip().split('\n')

    data_original = [data for data in data_original if '.' not in data.split()[0]]

    smiles_list = [data.strip().split()[0] for data in data_original]
    property_list = [float(data.strip().split()[1]) for data in data_original]

    # Split by canonical molecular identity: rows describing the same molecule
    # (possibly written as different SMILES) are kept on the same side of the split
    canonical_keys = [molecule_identity(smiles) for smiles in smiles_list]

    groups = defaultdict(list)
    for i, key in enumerate(canonical_keys):
        groups[key].append(i)

    np.random.seed(seed)
    group_keys = list(groups.keys())
    np.random.shuffle(group_keys)
    n_train_groups = int(split_ratio * len(group_keys))
    train_keys = set(group_keys[:n_train_groups])

    train_idx = [i for key in group_keys if key in train_keys for i in groups[key]]
    dev_idx = [i for key in group_keys if key not in train_keys for i in groups[key]]

    train_smiles = [smiles_list[i] for i in train_idx]
    train_props = [property_list[i] for i in train_idx]
    dev_smiles = [smiles_list[i] for i in dev_idx]
    dev_props = [property_list[i] for i in dev_idx]

    # Apply SMILES augmentation exclusively to the training subset
    print(f"Applying SMILES augmentation to training set (num_augments={num_augments})...")
    aug_train_smiles, aug_train_props = augment_smiles_batch(
        train_smiles, train_props, num_augments, augmentation_types
    )
    print(f"Train: {len(train_smiles)} -> {len(aug_train_smiles)} (augmented)")
    print(f"Dev: {len(dev_smiles)} (no augmentation)")

    # Build the tokenizer exclusively from the training subset; validation data
    # is encoded with this frozen vocabulary (unseen tokens map to <UNK>)
    init_smiles_tokenizer(aug_train_smiles)

    def _build(smiles_props):
        ds = []
        for smiles, prop in smiles_props:
            mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
            if mol is None:
                continue
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
            adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
            input_ids, attention_mask = tokenize_smiles(smiles)
            input_ids = torch.LongTensor(input_ids).to(device)
            attention_mask = torch.LongTensor(attention_mask).to(device)
            fingerprints = torch.LongTensor(fingerprints).to(device)
            adjacency = torch.FloatTensor(adjacency).to(device)
            prop_tensor = torch.FloatTensor([[float(prop)]]).to(device)
            ds.append((smiles, fingerprints, adjacency, molecular_size,
                       prop_tensor, input_ids, attention_mask))
        return ds

    # Build the training subset first so the feature dictionaries see training
    # data only, then freeze them before encoding the validation subset
    train_dataset = _build(zip(aug_train_smiles, aug_train_props))
    freeze_feature_dicts()
    dev_dataset = _build(zip(dev_smiles, dev_props))

    dump_dictionary(fingerprint_dict, dir_dataset + dataname + '-fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir_dataset + dataname + '-atom_dict.pickle')
    dump_dictionary(bond_dict, dir_dataset + dataname + '-bond_dict.pickle')
    dump_dictionary(edge_dict, dir_dataset + dataname + '-edge_dict.pickle')
    save_tokenizer(dir_dataset, dataname)

    return train_dataset, dev_dataset

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses a GPU!')
else:
    device = torch.device('cpu')
    print('The code uses a CPU...')
	
def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [_dict_index(atom_dict, a, 'atom') for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = _dict_index(bond_dict, str(b.GetBondType()), 'bond')
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        nodes = [_dict_index(fingerprint_dict, a, 'fingerprint') for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(_dict_index(fingerprint_dict, fingerprint, 'fingerprint'))

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = _dict_index(edge_dict, (both_side, edge), 'edge')
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)

def create_dataset(filename, path, dataname, build_tokenizer=True):
    dir_dataset = path
    print(filename)
    """Load a dataset."""
    with open(dir_dataset + filename, 'r') as f:
        smiles_property = f.readline().strip().split()
        data_original = f.read().strip().split('\n')

        """Exclude the data contains '.' in its smiles."""
    data_original = [data for data in data_original
                        if '.' not in data.split()[0]]

    # build_tokenizer=False: reuse the frozen tokenizer built from the training
    # subset and never rebuild it from this file (unseen tokens map to <UNK>)
    if build_tokenizer:
        smiles_list = [data.strip().split()[0] for data in data_original]
        init_smiles_tokenizer(smiles_list)
    elif smiles_tokenizer is None:
        raise ValueError(
            "Tokenizer is not initialized. Load the training set first or set build_tokenizer=True."
        )

    # Tokenizer and feature dictionaries are frozen from the training subset;
    # encoding this file never extends them (unseen tokens/patterns map to
    # <UNK>/<unk> indices)
    dataset = []
    for data in data_original:

        smiles, property = data.strip().split()

        """Create each data with the above defined functions."""
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
        fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
        
        input_ids, attention_mask = tokenize_smiles(smiles)
        input_ids = torch.LongTensor(input_ids).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device)

#Transform the above each data of numpy to pytorch tensor on a device (i.e., CPU or GPU).
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        property = torch.FloatTensor([[float(property)]]).to(device)
        dataset.append((smiles,fingerprints, adjacency, molecular_size, property, input_ids, attention_mask))
    dir_dataset=path
    dump_dictionary(fingerprint_dict, dir_dataset +dataname+ '-fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir_dataset +dataname+ '-atom_dict.pickle')
    dump_dictionary(bond_dict, dir_dataset  +dataname+ '-bond_dict.pickle')
    dump_dictionary(edge_dict, dir_dataset +dataname+ '-edge_dict.pickle')
    save_tokenizer(dir_dataset, dataname)
    return dataset
	
def check_identity_overlap(dataset_a, dataset_b, name_a='dataset_a', name_b='dataset_b'):
    """Report whether any canonical molecular identity appears in both datasets."""
    def _keys(dataset):
        return {molecule_identity(row[0]) for row in dataset}

    overlap = _keys(dataset_a) & _keys(dataset_b)
    if overlap:
        print(f'WARNING: {len(overlap)} molecular identities appear in both {name_a} and {name_b}!')
    else:
        print(f'Check passed: no molecular identity is shared between {name_a} and {name_b}.')
    return overlap


def create_dataset_randomsplit(x,y,path,dataname):
    dir_input = path + 'SMRT-'
    with open(dir_input + 'atom_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            atom_dict.get(k)
            atom_dict[k]=c[k]
    with open(dir_input+ 'bond_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            bond_dict.get(k)
            bond_dict[k]=c[k]
        
    with open(dir_input + 'edge_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            edge_dict.get(k)
            edge_dict[k]=c[k]
        
    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            fingerprint_dict.get(k)
            fingerprint_dict[k]=c[k]
    
    load_tokenizer(path, 'SMRT')
    
    dataset = []  
    for i in range(len(x)):
        smiles=x[i]
        property=y[i]         
        """Create each data with the above defined functions."""
        mol = Chem.MolFromInchi(smiles)     
        mol = Chem.AddHs(Chem.MolFromInchi(smiles))
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
        fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
        
        smiles_canonical = Chem.MolToSmiles(mol)
        input_ids, attention_mask = tokenize_smiles(smiles_canonical)
        input_ids = torch.LongTensor(input_ids).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device)
        
#Transform the above each data of numpy to pytorch tensor on a device (i.e., CPU or GPU).
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        property = torch.FloatTensor([[float(property)]]).to(device)

        dataset.append((smiles,fingerprints, adjacency, molecular_size, property, input_ids, attention_mask))
    dir_dataset=path
    dump_dictionary(fingerprint_dict, dir_dataset +dataname+ '-fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir_dataset +dataname+ '-atom_dict.pickle')
    dump_dictionary(bond_dict, dir_dataset  +dataname+ '-bond_dict.pickle')
    dump_dictionary(edge_dict, dir_dataset +dataname+ '-edge_dict.pickle')
    save_tokenizer(dir_dataset, dataname)
    return dataset
	
def create_dataset_kfold(x,y,path,dataname):
    dir_input =path+'SMRT-'
    with open(dir_input + 'atom_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            atom_dict.get(k)
            atom_dict[k]=c[k]
    with open(dir_input+ 'bond_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            bond_dict.get(k)
            bond_dict[k]=c[k]
        
    with open(dir_input + 'edge_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            edge_dict.get(k)
            edge_dict[k]=c[k]
        
    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            fingerprint_dict.get(k)
            fingerprint_dict[k]=c[k]
    
    load_tokenizer(path, 'SMRT')
    
    dataset = []
    for i in range(len(x)):
        smiles=x[i]
        property=y[i]
        """Create each data with the above defined functions."""
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
        fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
        
        input_ids, attention_mask = tokenize_smiles(smiles)
        input_ids = torch.LongTensor(input_ids).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device)

#Transform the above each data of numpy to pytorch tensor on a device (i.e., CPU or GPU).
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        property = torch.FloatTensor([[float(property)]]).to(device)

        dataset.append((smiles,fingerprints, adjacency, molecular_size, property, input_ids, attention_mask))
    dir_dataset=path
    dump_dictionary(fingerprint_dict, dir_dataset +dataname+ '-fingerprint_dict.pickle')
    dump_dictionary(atom_dict, dir_dataset +dataname+ '-atom_dict.pickle')
    dump_dictionary(bond_dict, dir_dataset  +dataname+ '-bond_dict.pickle')
    dump_dictionary(edge_dict, dir_dataset +dataname+ '-edge_dict.pickle')
    save_tokenizer(dir_dataset, dataname)
    return dataset


def transferlearning_dataset_predict(x,path):
    dir_input = path+'SMRT-'
    with open(dir_input + 'atom_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            atom_dict.get(k)
            atom_dict[k]=c[k]
    with open(dir_input+ 'bond_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            bond_dict.get(k)
            bond_dict[k]=c[k]
        
    with open(dir_input + 'edge_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            edge_dict.get(k)
            edge_dict[k]=c[k]
        
    with open(dir_input + 'fingerprint_dict.pickle', 'rb') as f:
        c=pickle.load(f)
        for k in c.keys():
            fingerprint_dict.get(k)
            fingerprint_dict[k]=c[k]
    
    load_tokenizer(path, 'SMRT')
    
    dataset = []
    
    for i in range(len(x)):
        smiles=x[i]
        """Create each data with the above defined functions."""       
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue           
        else:
            smi = Chem.MolToSmiles(mol)            
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        atoms = create_atoms(mol, atom_dict)
        molecular_size = len(atoms)
        i_jbond_dict = create_ijbonddict(mol, bond_dict)
        fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
        adjacency = np.float32((Chem.GetAdjacencyMatrix(mol)))
        
        input_ids, attention_mask = tokenize_smiles(smi)
        input_ids = torch.LongTensor(input_ids).to(device)
        attention_mask = torch.LongTensor(attention_mask).to(device)
        
        #Transform the above each data of numpy to pytorch tensor on a device (i.e., CPU or GPU).
        fingerprints = torch.LongTensor(fingerprints).to(device)
        adjacency = torch.FloatTensor(adjacency).to(device)
        dataset.append((smiles,fingerprints, adjacency, molecular_size, input_ids, attention_mask)) 
    return dataset
