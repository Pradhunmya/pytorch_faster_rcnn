# path to your own data and coco file
train_data_dir = "trainval/images"
train_coco = "trainval/annotations/bbox-annotations.json"

# Batch size
train_batch_size = 2

# Params for dataloader
train_shuffle_dl = False
num_workers_dl = 0

# Params for training

# Two classes; Only target class or background
num_classes = 3
num_epochs = 100

lr = 0.005
momentum = 0.9
weight_decay = 0.005
