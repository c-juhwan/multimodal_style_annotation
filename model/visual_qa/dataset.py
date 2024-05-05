# Standard Library Modules
import pickle
# 3rd-party Modules
from tqdm.auto import tqdm
# Pytorch Modules
from torch.utils.data.dataset import Dataset

class VQADataset(Dataset):
    def __init__(self, data_path: str):
        super(VQADataset, self).__init__()
        with open(data_path, 'rb') as f:
            data_ = pickle.load(f)

        self.data_list = []

        for idx in tqdm(range(len(data_['image']))):
            self.data_list.append({
                'image': data_['image'][idx], # PIL Image
                'image_id': data_['image_id'][idx],
                'question_id': data_['question_id'][idx],
                'question': data_['question'][idx],
                'full_answer': data_['full_answer'][idx],
                'answer': data_['answer'][idx],
                'question_type': data_['question_type'][idx],
                'answer_type': data_['answer_type'][idx],
                'caption': data_['caption'][idx],
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
        'question_id': [item['question_id'] for item in batch], # List[str]
        'question': [item['question'] for item in batch], # List[str]
        'full_answer': [item['full_answer'] for item in batch], # List[str]
        'answer': [item['answer'] for item in batch], # List[str]
        'question_type': [item['question_type'] for item in batch], # List[str]
        'answer_type': [item['answer_type'] for item in batch], # List[str]
        'caption': [item['caption'] for item in batch], # List[str]
        'domain_id': [item['domain_id'] for item in batch], # List[int]
    }
