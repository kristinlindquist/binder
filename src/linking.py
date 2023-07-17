
from transformers import AutoModel, AutoTokenizer
import torch
from typing import TypedDict

# C0630105|ENG|P|L1127426|PF|S1353962|Y|A1314408||M0147312|C051808|MSH|NM|C051808|6-amino-1-arabinofuranosyl-1H-pyrrolo(3,2-c)pyridin-4(5H)-one|0|N|256|

def get_definition(cui, source, def_filename):
    with open(def_filename, 'r') as f:
        # scan to the first line that starts with the CUI
        for line in f:
            if line.startswith(cui) and line.split('|')[4] == source:
                return line.strip().split('|')[5]

    return ""

UmlsRecord = TypedDict("UmlsRecord", {"cui": str, "name": str, "description": str })

def load_umls_kb(umls_dir: str) -> list[UmlsRecord]:
    term_filename, def_filename = [os.path.join(umls_dir, filename) for filename in ['MRCONSO.RRF', 'MRDEF.RRF']]
    with open(term_filename, 'r') as f:
        # get lines that are English, preferred
        # TODO: synonyms?
        terms = [line for line in f.readlines() if '|ENG|P|' in line]

    umls_kb = []
    for line in lines:
        line = line.strip()
        fields = line.split('|')
        cui = fields[0]
        name = fields[14]
        source = fields[11]
        umls_kb.append({
            'cui': cui,
            'name': name,
            'description': get_definition(cui, source, def_filename)
        })

def encode_umls_kb(config, umls_kb):
    """
    Usage: umls_entities = encode_umls_kb(config, umls_kb) 
    """
    # Load pre-trained encoder model 
    encoder = AutoModel.from_pretrained(config.pretrained_model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(config.pretrained_model_name_or_path)

    # Encode each UMLS entity 
    umls_embeds = []
    for entity in umls_kb:
        entity_text = entity['name'] + ' ' + entity['description']
        inputs = tokenizer(entity_text, return_tensors='pt')
        outputs = encoder(**inputs)

        # Use the CLS token embedding as the entity representation
        embed = outputs.last_hidden_state[:,0,:]  
        umls_embeds.append(embed)

    umls_embeds = torch.stack(umls_embeds)
    
    return umls_embeds