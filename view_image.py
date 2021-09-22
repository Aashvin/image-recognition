import torch as T
import torchvision as tv
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

NUMBER = 300

def imshow(img):
    img = img / 2 + 0.5   # unnormalize
    npimg = img.numpy()   # convert from tensor
    plt.imshow(np.transpose(npimg, (1, 2, 0))) 
    plt.show()

def main():
    transform = transforms.Compose( [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5))])

    trainset = tv.datasets.CIFAR10(root='.\\data', train=False,
        download=False, transform=transform)
    trainloader = T.utils.data.DataLoader(trainset,
        batch_size=350, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # get first 100 training images
    dataiter = iter(trainloader)
    imgs, lbls = dataiter.next()

    print(classes[lbls[NUMBER]])
    print(imgs[NUMBER])
    imshow(tv.utils.make_grid(imgs[NUMBER]))

    # for i in range(100):  # show just the frogs
    #     if lbls[i] == 6:  # 6 = frog
    #         imshow(tv.utils.make_grid(imgs[i]))

if __name__ == "__main__":
    main()