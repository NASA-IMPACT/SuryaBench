import torch
from ds_datasets.ar import AREmergenceDataset
from tqdm import tqdm

train_data_path = (
    "/rgroup/aifm/aremerge_skasapis/train_indexed_data_ar_emergence_kasapis_rohit.h5"
)
train_subset = AREmergenceDataset(train_data_path)

# Initialize min and max tensors
data_min = None
data_max = None

# Iterate through the dataset
# for d, _ in tqdm(train_subset):  # Assuming the dataset returns (data, label)
#     data = d['input']  # Adjust this based on your dataset structure
#     batch_min = data.amin(dim=(0, 2))
#     batch_max = data.amax(dim=(0, 2))

#     if data_min is None:
#         data_min = batch_min
#         data_max = batch_max
#     else:
#         data_min = torch.min(data_min, batch_min)
#         data_max = torch.max(data_max, batch_max)

# data_min = torch.tensor([-7.4745e+07, -3.6508e+08, -1.6605e+08, -3.5536e+07, -7.0342e+01])
# data_max = torch.tensor([2.3280e+07, 1.4658e+08, 5.8470e+07, 2.7218e+07, 4.9013e+02])


for d, _ in tqdm(train_subset):  # Assuming the dataset returns (data, label)
    data = d["output"]  # Adjust this based on your dataset structure

    batch_min = data.min()
    batch_max = data.max()

    if data_min is None:
        data_min = batch_min
        data_max = batch_max
    else:
        data_min = torch.min(data_min, batch_min)
        data_max = torch.max(data_max, batch_max)

# Channel-wise Min: tensor(-12419.5938)
# Channel-wise Max: tensor(2505.3042)

print("Channel-wise Min:", data_min)
print("Channel-wise Max:", data_max)
