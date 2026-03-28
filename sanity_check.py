import os

for cls in os.listdir("dataset/annotations"):
    folder = os.path.join("dataset/annotations", cls)
    for f in os.listdir(folder)[:50]:
        print(f)
    break