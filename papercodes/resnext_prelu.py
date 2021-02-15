from torchvision.models import resnext50_32x4d


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.om = resnext50_32x4d(pretrained = True)

    def forward(self, x):
        x = self.om(x)
        return x



mod = Model()

for module in mod.modules():
    module.om.relu = nn.PReLU()
    for m in module.om.layer1.modules():
        for k in m.children():
            k.relu = nn.PReLU()
        break     
    for m in module.om.layer2.modules():
        for k in m.children():
            k.relu = nn.PReLU()
        break     
    for m in module.om.layer3.modules():
        for k in m.children():
            k.relu = nn.PReLU()
        break     
    for m in module.om.layer4.modules():
        for k in m.children():
            k.relu = nn.PReLU()
        break     
    break



for module in mod.modules():
    print(module)
    break





