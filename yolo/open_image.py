import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset("quickstart")
print(dataset)
# session = fo.launch_app(dataset)
