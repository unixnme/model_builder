## ModelBuilder
Separate out the model definition as a json file from actual python code!

### Example
```
cat model.json
{
  "sequential": [
    "Reshape 3 28 28",
    "ReLU",
    "Conv1d 3 16 5",
    "MaxPool2d 2",
    "ReLU",
    "Conv2d 16 32 5",
    "MaxPool2d 2",
    "ReLU",
    "Reshape 512",
    "Linear 512 10"
  ]
} 
>> import model_builder
>> model = model_builder.build_from_file("model.json")
Sequential(
  (0): Reshape(3, 28, 28)
  (1): ReLU()
  (2): Conv1d(3, 16, kernel_size=(5,), stride=(1,))
  (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (4): ReLU()
  (5): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
  (6): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (7): ReLU()
  (8): Reshape(512,)
  (9): Linear(in_features=512, out_features=10, bias=True)
)
``` 

### Supported layers
all pytorch default layers from torch.nn with the exact same params<br>
Reshape(*channels) will reshape the output to (B, *channels)
