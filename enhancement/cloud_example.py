from cloud_enhancement import CloudEnhancer

models = [
    "model_clahe.pth",
    "model_hist.pth",
    "model_sharpen.pth",
    "model_bilateral.pth",
    "model_gamma.pth"
]

targets = [
    "dataset/target_clahe",
    "dataset/target_hist",
    "dataset/target_sharpen",
    "dataset/target_bilateral",
    "dataset/target_gamma"
]

cloud = CloudEnhancer(models)

enhanced_images, weights = cloud.enhance(
    "dataset/input/Y_2015_00005.jpg",
    targets
)

print("Weights:", weights)