ATOM_ELEMENTS = ["C", "O", "N", "S", "F", "Cl", "P", "H", "Br", "I", "B"]
N_POCKET_ELEMENTS = 4
N_LIGAND_ELEMENTS = 7

ELEMENTS_HASH_CROSSDOCKED = {
    "C": 0,
    "O": 1,
    "N": 2,
    "S": 3,
    "F": 4,
    "Cl": 5,
    "P": 6,
    "H": 7,
}

CHANNEL_TO_ATM_NB_CROSSDOCKED = {
    0: 6,   # C
    1: 8,   # O
    2: 7,   # N
    3: 16,  # S
    4: 9,   # F
    5: 17,  # Cl
    6: 15,  # P
    7: 1,   # H
}

# atomic radii from https://en.wikipedia.org/wiki/Van_der_Waals_radius
RADIUS_PER_ATOM = {
    "MOL": {
        "C": 1.7,
        "O": 1.52,
        "N": 1.55,
        "S": 1.8,
        "F": 1.47,
        "Cl": 1.75,
        "Br": 1.85,
        "P": 1.8,
        "H": 1.2,
        "I": 1.98,
        "B": 1.92,
    }
}
