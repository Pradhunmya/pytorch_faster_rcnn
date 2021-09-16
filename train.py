""" This is the file through which we can start our training process"""

# import the necessary packages
import config
import torch
from utils import (
    get_model_instance_segmentation,
    collate_fn,
    get_transform,
    PersonCarDataset,
)

# torch version check
print("Torch version:", torch.__version__)

# create Person/Car Dataset
dataset = PersonCarDataset(
    root=config.train_data_dir, annotation=config.train_coco, transforms=get_transform()
)

# split the dataset in train and test set
indices = torch.randperm(len(dataset)).tolist()
dataset = torch.utils.data.Subset(dataset, indices[:-50])
dataset_test = torch.utils.data.Subset(dataset, indices[-50:])


# define training DataLoader
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config.train_batch_size,
    shuffle=config.train_shuffle_dl,
    num_workers=config.num_workers_dl,
    collate_fn=collate_fn,
)

# define validation DataLoader
data_loader = torch.utils.data.DataLoader(
    dataset_test,
    batch_size=config.train_batch_size,
    shuffle=config.train_shuffle_dl,
    num_workers=config.num_workers_dl,
    collate_fn=collate_fn,
)

# select device (whether GPU or CPU)
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# DataLoader is iterable over Dataset
for imgs, annotations in data_loader:
    imgs = list(img.to(device) for img in imgs)
    annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    print(annotations)

# get the model using our helper function
model = get_model_instance_segmentation(config.num_classes)

# move model to the right device
model.to(device)

# parameters
params = [p for p in model.parameters() if p.requires_grad]

# construct an optimizer
optimizer = torch.optim.SGD(
    params, lr=config.lr, momentum=config.momentum, weight_decay=config.weight_decay
)

len_dataloader = len(data_loader)

# Training
for epoch in range(config.num_epochs):
    print(f"Epoch: {epoch}/{config.num_epochs}")
    model.train()
    i = 0
    for imgs, annotations in data_loader:
        i += 1
        print(i)
        if i in [381, 690, 691]:
          continue
      
        imgs = list(img.to(device) for img in imgs)
        annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]        
        
        # calculating the loss for every iteration
        loss_dict = model(imgs, annotations)        
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Iteration: {i}/{len_dataloader}, Loss: {losses}")
torch.save(model.state_dict(), "model/model.pt")         
