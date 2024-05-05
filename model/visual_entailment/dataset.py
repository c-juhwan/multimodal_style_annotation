# Standard Library Modules
import pickle
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
from torch.utils.data.dataset import Dataset

class VEDataset(Dataset):
    def __init__(self, data_path: str):
        super(VEDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []

        for idx in tqdm(range(len(data_['image']))):
            self.data_list.append({
                'image': data_['image'][idx], # PIL Image
                'image_id': data_['image_id'][idx],
                'premise': data_['premise'][idx],
                'hypothesis': data_['hypothesis'][idx],
                'label': data_['label'][idx],
                'domain_id': data_['domain_id'][idx],
            })

        del data_

    def __getitem__(self, idx: int) -> dict:
        return self.data_list[idx]

    def __len__(self) -> int:
        return len(self.data_list)

def collate_fn(batch):
    return {
        'image': [item['image'].convert("RGB") for item in batch], # List[PIL Image]
        'image_id': [item['image_id'] for item in batch], # List[str]
        'premise': [item['premise'] for item in batch], # List[str]
        'hypothesis': [item['hypothesis'] for item in batch], # List[str]
        'label': [item['label'] for item in batch], # List[int]
        'domain_id': [item['domain_id'] for item in batch], # List[int]
    }
