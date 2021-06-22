from ode_genre_classification_image import *
from preprocessing import *

device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

is_odenet = args.network == 'odenet'

if args.downsampling_method == 'conv':
    downsampling_layers = [
        nn.Conv2d(3, 64, 3, 1),
        norm(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1),
        norm(64),
        nn.ReLU(inplace=True),
        nn.Conv2d(64, 64, 4, 2, 1),
    ]
elif args.downsampling_method == 'res':
    downsampling_layers = [
        nn.Conv2d(3, 64, 3, 1),
        ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
        ResBlock(64, 64, stride=2, downsample=conv1x1(64, 64, 2)),
    ]

feature_layers = [ODEBlock(ODEfunc(64))] if is_odenet else [ResBlock(64, 64) for _ in range(6)]
fc_layers = [norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 3)]
    
model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(device)

model.load_state_dict(torch.load('experiment4/model.pth')['state_dict'])
def classify_audio_png(root_dir, filename):
    img_path = join(root_dir, filename)
    image = io.imread(img_path)
    image = transforms.ToTensor()(image)
    image = image.narrow(0,0,3)
    image = image.unsqueeze(0)
    image = image.to(device)
    prediction = torch.argmax(model(image))
    genres = ['classical','jazz','rock']
    return genres[prediction]

def classify_audio_wav(root_dir, filename,s):
    print('Converting to Mel spectrogram')
    audio_to_squeezed_spectrogram(root_dir, filename, root_dir, s=s)
    print('Spectrogram saved in '+join(root_dir,filename[:-3]+str(s)+".png"))
    img_path = join(root_dir,filename[:-3]+str(s)+".png")
    image = io.imread(img_path)
    image = transforms.ToTensor()(image)
    image = image.narrow(0,0,3)
    image = image.unsqueeze(0)
    image = image.to(device)
    prediction = torch.argmax(model(image))
    genres = ['classical','jazz','rock']
    return genres[prediction]
