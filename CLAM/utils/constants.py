IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
optimus_mean = [0.707223, 0.578729, 0.703617]
optimus_std =[0.211883, 0.230117, 0.177517]

MODEL2CONSTANTS = {
	"resnet": {
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"uni":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
    "uni2":
	{
		"mean": IMAGENET_MEAN,
		"std": IMAGENET_STD
	},
	"optimus1":
	{
		"mean": optimus_mean,
		"std": optimus_std
	}
}