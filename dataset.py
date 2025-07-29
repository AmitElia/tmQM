class Spice2Dataset(Dataset):
    def __init__(self, npz_file_path: str):
        self.file_path = npz_file_path
        self.npz_file = np.load(self.file_path, allow_pickle=True)
        self.key_list = list(self.npz_file.keys())
    
    def __len__(self):
        return len(self.key_list)

    def __getitem__(self, index):
        return self.npz_file[self.key_list[index]].item()