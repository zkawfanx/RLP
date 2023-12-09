import torch.nn as nn

# U-Net
class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=3, n_feat=32):
        super(UNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c, n_feat*2, 5, 1, 2),
            nn.ReLU()
            )
        self.conv2 = nn.Sequential(
            nn.Conv2d(n_feat*2, n_feat*4, 3, 2, 1),
            nn.ReLU()
            )
        self.conv3 = nn.Sequential(
            nn.Conv2d(n_feat*4, n_feat*4, 3, 1, 1),
            nn.ReLU()
            )
        self.conv4 = nn.Sequential(
            nn.Conv2d(n_feat*4, n_feat*8, 3, 2, 1),
            nn.ReLU()
            )
        self.conv5 = nn.Sequential(
            nn.Conv2d(n_feat*8, n_feat*8, 3, 1, 1),
            nn.ReLU()
            )
        self.conv6 = nn.Sequential(
            nn.Conv2d(n_feat*8, n_feat*8, 3, 1, 1),
            nn.ReLU()
            )
        
        self.diconv1 = nn.Sequential(
            nn.Conv2d(n_feat*8, n_feat*8, 3, 1, 2, dilation = 2),
            nn.ReLU()
            )
        self.diconv2 = nn.Sequential(
            nn.Conv2d(n_feat*8, n_feat*8, 3, 1, 4, dilation = 4),
            nn.ReLU()
            )
        self.diconv3 = nn.Sequential(
            nn.Conv2d(n_feat*8, n_feat*8, 3, 1, 8, dilation = 8),
            nn.ReLU()
            )
        self.diconv4 = nn.Sequential(
            nn.Conv2d(n_feat*8, n_feat*8, 3, 1, 16, dilation = 16),
            nn.ReLU()
            )        
        self.conv7 = nn.Sequential(
            nn.Conv2d(n_feat*8, n_feat*8, 3, 1, 1),
            nn.ReLU()
            )
        self.conv8 = nn.Sequential(
            nn.Conv2d(n_feat*8, n_feat*8, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(n_feat*8, n_feat*4, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        
        self.conv9 = nn.Sequential(
            nn.Conv2d(n_feat*4, n_feat*4, 3, 1, 1),
            nn.ReLU()
            )
        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(n_feat*4, n_feat*2, 4, 2, 1),
            nn.ReflectionPad2d((1, 0, 1, 0)),
            nn.AvgPool2d(2, stride = 1),
            nn.ReLU()
            )
        self.conv10 = nn.Sequential(
            nn.Conv2d(n_feat*2, n_feat, 3, 1, 1),
            nn.ReLU()
            )
        self.output = nn.Sequential(
            nn.Conv2d(n_feat, out_c, 3, 1, 1)
            )


    def forward(self, input):
        x = input
        
        x = self.conv1(x)
        res1 = x
        x = self.conv2(x)
        x = self.conv3(x)
        res2 = x
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.diconv1(x)
        x = self.diconv2(x)
        x = self.diconv3(x)
        x = self.diconv4(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.deconv1(x)
        x = x + res2
        x = self.conv9(x)
        x = self.deconv2(x)
        x = x + res1
        x = self.conv10(x)
        x = self.output(x)

        return x