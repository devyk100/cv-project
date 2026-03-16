from evaluation.exdark_eval import evaluate_dataset

dataset_images = [
    "2015_00001.jpg",
    "2015_00002.jpg"
]

predictions = {

    "2015_00001.jpg":[
        {"bbox":[50,40,200,180],"score":0.9,"class":1}
    ]

}

results = evaluate_dataset(
    dataset_images,
    "dataset/annotations",
    predictions
)

print(results)