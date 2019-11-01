## ModelBuilder
Separate out the model definition as a json file from actual python code!

### Example
```
cat examples/model.json
{
  "sequential": [
    "Reshape 3 28 28",
    "Conv2d 3 16 5",
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
>> model = model_builder.build_from_file("examples/model.json")
Sequential(
  (0): Reshape(3, 28, 28)
  (1): Conv2d(3, 16, kernel_size=(5,), stride=(1,))
  (2): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (3): ReLU()
  (4): Conv2d(16, 32, kernel_size=(5, 5), stride=(1, 1))
  (5): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
  (6): ReLU()
  (7): Reshape(512,)
  (8): Linear(in_features=512, out_features=10, bias=True)
)
``` 

### Supported layers
all pytorch default layers from torch.nn with the exact same params<br>
Reshape(*channels) will reshape the output to (B, *channels)
