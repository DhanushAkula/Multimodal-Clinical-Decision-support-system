import json
import os
from typing import List, Dict, Optional

class MedPixDataLoader:
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        with open(os.path.join(data_dir, 'Case_topic.json'), 'r') as f:
            self.cases = json.load(f)
        with open(os.path.join(data_dir, 'Descriptions.json'), 'r') as f:
            self.descriptions = json.load(f)
        self.case_index = {c['U_id']: c for c in self.cases}
        self.desc_index = {}
        for desc in self.descriptions:
            uid = desc['U_id']
            if uid not in self.desc_index:
                self.desc_index[uid] = []
            self.desc_index[uid].append(desc)
        print(f"✓ Loaded {len(self.cases)} cases, {len(self.descriptions)} images")
    
    def get_case(self, u_id: str) -> Optional[Dict]:
        return self.case_index.get(u_id)
    
    def get_descriptions(self, u_id: str) -> List[Dict]:
        return self.desc_index.get(u_id, [])
    
    def get_all_cases(self) -> List[Dict]:
        return self.cases
    
    def get_split(self, split_name: str) -> List[Dict]:
        # Try .jsonl format first
        split_file = os.path.join(self.data_dir, 'splitted_dataset', f'data_{split_name}.jsonl')
        if os.path.exists(split_file):
            split_cases = []
            with open(split_file, 'r') as f:
                for line in f:
                    if line.strip():
                        split_cases.append(json.loads(line))
            return split_cases
        # Fallback to .json
        split_file = os.path.join(self.data_dir, 'splitted_dataset', f'{split_name}.json')
        if os.path.exists(split_file):
            with open(split_file, 'r') as f:
                split_uids = json.load(f)
            return [self.case_index[uid] for uid in split_uids if uid in self.case_index]
        return []
    
    def get_image_path(self, image_id: str) -> str:
        return os.path.join(self.data_dir, 'images', f'{image_id}.png')
