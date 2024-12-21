from DiffusionModel import DenoisingDiffusion

model = DenoisingDiffusion("cifar", "cpu") 

model.sample_images(1, None)


#