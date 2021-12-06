def augmentation_step(X, rotate=False):

    transform = transforms.RandomApply([transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip()], p=0.5)
    rotation = transforms.RandomApply([transforms.RandomRotation(degrees=(0, 270))], p=0.3)
    normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    transforms.CenterCrop(300)])

    X = transform(X)
    if rotate:
        X = rotation(X)
    X = normalize(X)

    return X

# as part of the datasets class
if self.augmentation:
	X = augmentation_step(X, rotate=True)
else:
	transform_norm = transforms.Compose([
		transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
		transforms.CenterCrop(300)
	])
	X = transform_norm(X)

return X, y

# As part of the model definition method
model.classifier = nn.Linear(1280, OUTPUTS_a)

