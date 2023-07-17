
from transformers import AutoModel, AutoTokenizer
import torch
from typing import TypedDict
import os
import logging
import h5py
import math

UmlsRecord = TypedDict("UmlsRecord", {"cui": str, "name": str, "description": str })

UMLS_EMBEDDINGS_FILE = "umls_embeds.h5"
OUTPUT_DATASET = "embeddings"
BATCH_SIZE = 32
SAVE_INTERVAL = 100 # Save embeddings every 100 batches

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# C0630105|ENG|P|L1127426|PF|S1353962|Y|A1314408||M0147312|C051808|MSH|NM|C051808|6-amino-1-arabinofuranosyl-1H-pyrrolo(3,2-c)pyridin-4(5H)-one|0|N|256|

def load_definitions(def_filename) -> dict[str, str]:
    """
    Load UMLS definitions from def file file

    Args:
        def_filename: Path to def file
    """
    lookup = {}
    with open(def_filename, 'r') as f:
        # scan to the first line that starts with the CUI
        for line in f:
            values = line.strip().split('|')
            lookup[f"{values[0]}-{values[4]}"] = values[5]

    logger.info("Loaded %s definitions", len(lookup))
    return lookup


def load_umls_kb(umls_dir: str) -> list[UmlsRecord]:
    """
    Load UMLS entities from MRCONSO.RRF and MRDEF.RRF files in `umls_dir`

    Args:
        umls_dir: Path to directory containing MRCONSO.RRF and MRDEF.RRF files
    """
    logger.info("Loading UMLS entities from %s", umls_dir)
    term_filename, def_filename = [os.path.join(umls_dir, filename) for filename in ['MRCONSO.RRF', 'MRDEF.RRF']]
    with open(term_filename, 'r') as f:
        # get lines that are English, preferred
        # TODO: synonyms?
        lines = [line for line in f.readlines() if '|ENG|P|' in line]

    definitions = load_definitions(def_filename)
    umls_kb = []
    for idx, line in enumerate(lines):
        line = line.strip()
        fields = line.split('|')
        cui = fields[0]
        name = fields[14]
        source = fields[11]
        umls_kb.append({
            'cui': cui,
            'name': name,
            'description': definitions.get(f"{cui}-{source}") or ""
        })
        if idx % 100000 == 0:
            logger.info("Loaded %s UMLS lines, last %s", idx, umls_kb[-1])
    logger.info("Loaded %s UMLS entities", len(umls_kb))
    return umls_kb


def encode_umls_kb(config, rebuild: bool = False, batch_size: int = BATCH_SIZE, save_interval: int = SAVE_INTERVAL, embeddings_file: str = UMLS_EMBEDDINGS_FILE) -> torch.Tensor:
    """
    Usage: umls_entities = encode_umls_kb(config, umls_kb) 

    Args:
        config: BinderConfig object
        rebuild: If True, rebuild the embeddings from scratch
        batch_size: Number of entities to encode at once
        save_interval: Save embeddings to disk every `save_interval` batches
    """
    # load or build embeddings
    if not rebuild and os.path.isfile(embeddings_file):
        return load_embeddings_from_disk(embeddings_file, OUTPUT_DATASET)

    umls_kb = load_umls_kb(config.umls_dir)

    # Load pre-trained encoder model 
    encoder = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)

    # Encode each UMLS entity 
    umls_embeds = []

    with h5py.File(embeddings_file, "w") as h5_file:
        for idx in range(0, math.ceil(len(umls_kb) / batch_size)):
            batch = idx * batch_size
            logger.info("Starting on batch %s", batch)
            batch_entities = umls_kb[batch : batch + batch_size]
            batch_texts = [entity["name"] for entity in batch_entities]

            inputs = tokenizer.batch_encode_plus(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=256,
            )

            outputs = encoder(**inputs)

            # Use the CLS token embedding as the entity representation
            embeds = outputs.last_hidden_state[:, 0, :]
            umls_embeds.extend(embeds)

            if (idx + 1) % save_interval == 0 or idx + batch_size >= len(umls_kb):
                embeds_tensor = torch.stack(umls_embeds, dim=0)
                logger.info("Encoded/saved %s UMLS entities (%s, %s)", (idx + 1) * batch_size, len(umls_embeds), embeds_tensor.shape)
                save_embeddings_to_disk(h5_file, OUTPUT_DATASET, (idx + 1) * batch_size, embeds_tensor)
                umls_embeds = []
    h5_file.close()

    umls_embeds = load_embeddings_from_disk(embeddings_file, OUTPUT_DATASET)
    logger.info("Encoded UMLS entities total: %s", len(umls_embeds))

    return umls_embeds


def save_embeddings_to_disk(h5_file: h5py.File, dataset_name: str, start_index: int, embeddings):
    """
    Save embeddings to disk

    Args:
        h5_file: h5py file object
        dataset_name: Name of the dataset to save to
        start_index: Index to start saving at
        embeddings: Embeddings to save
    """
    if dataset_name not in h5_file:
        logger.info("Creating file %s or dataset %s", h5_file, dataset_name)
        h5_file.create_dataset(dataset_name, shape=(embeddings.size(0), embeddings.size(1)), dtype=float)
    else:
        logger.info("Appending to dataset %s", dataset_name)

    logger.info("From %s to %s", start_index - embeddings.size(0), start_index)
    h5_file[dataset_name][start_index - embeddings.size(0):start_index, :] = embeddings.detach().cpu().numpy()

def load_embeddings_from_disk(file_path: str, dataset_name: str):
    """
    Load embeddings from disk

    Args:
        file_path: Path to the h5py file
        dataset_name: Name of dataset from which to load embeddings
    """
    with h5py.File(file_path, "r") as h5_file:
        embeddings = h5_file[dataset_name][:]
    return torch.tensor(embeddings)