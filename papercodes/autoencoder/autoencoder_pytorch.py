import torch
import torch.nn as nn

class autoencodernet(nn.Module):
    def __init__(self):
        super(autoencodernet, self).__init__()
        self.compress1 = nn.Conv2d(in_channels = 3, out_channels = 3, kernel_size = 4, stride = 4, padding = 212)
        self.nlinear1 = nn.ReLU()
        self.decompress = nn.ConvTranspose2d(in_channels = 3, out_channels = 3, kernel_size  = 90, stride = 2)
        self.nlinear2 = nn.ReLU()

    def forward(self, x):
        x = self.compress1(x)
        x = self.nlinear1(x)
        y = self.decompress(x)
        y = self.nlinear2(y)
        return x,y


input = torch.randn([1,3,600,600])
model = autoencodernet()



#torch.save(model, 'ae.pth')
op = model(input)
print(type(op))
print(op[0].shape)
print(op[1].shape)


#for name, parameters in model.named_parameters():
#    print(name, parameters.numel())



