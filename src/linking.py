
from transformers import AutoModel, AutoTokenizer
import torch
from typing import TypedDict
import os
import logging

UmlsRecord = TypedDict("UmlsRecord", {"cui": str, "name": str, "description": str })

logger = logging.getLogger(__name__)
logger.setLevel("INFO")

# C0630105|ENG|P|L1127426|PF|S1353962|Y|A1314408||M0147312|C051808|MSH|NM|C051808|6-amino-1-arabinofuranosyl-1H-pyrrolo(3,2-c)pyridin-4(5H)-one|0|N|256|

def load_definitions(def_filename) -> dict[str, str]:
    lookup = {}
    with open(def_filename, 'r') as f:
        # scan to the first line that starts with the CUI
        for line in f:
            values = line.strip().split('|')
            lookup[f"{values[0]}-{values[4]}"] = values[5]

    logger.info("Loaded %s definitions", len(lookup))
    return lookup


def load_umls_kb(umls_dir: str) -> list[UmlsRecord]:
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
        if idx % 10000 == 0:
            logger.info("Loaded %s UMLS lines, last %s", idx, umls_kb[-1])
    logger.info("Loaded %s UMLS entities", len(umls_kb))
    return umls_kb

def encode_umls_kb(config):
    """
    Usage: umls_entities = encode_umls_kb(config, umls_kb) 
    """
    umls_kb = load_umls_kb(config.umls_dir)
    # Load pre-trained encoder model 
    encoder = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)

    # Encode each UMLS entity 
    umls_embeds = []
    for idx, entity in enumerate(umls_kb):
        entity_text = entity['name'] + ' ' + entity['description']
        inputs = tokenizer(entity_text, return_tensors='pt')
        outputs = encoder(**inputs)

        # Use the CLS token embedding as the entity representation
        embed = outputs.last_hidden_state[:,0,:]  
        umls_embeds.append(embed)

        if idx % 10000 == 0:
            logger.info("Encoded %s UMLS entities", idx)

    logger.info("Encoded UMLS entities", len(umls_embeds))
    
    umls_embeds = torch.stack(umls_embeds)

    return umls_embeds