# bot.py
import os
os.chdir('LossBot/')
import discord
import torch
from torch import nn
from torchvision import models, transforms
from dotenv import dotenv_values
from PIL import Image, ImageOps
import numpy as np
import torch.nn.functional as F
import uuid

env = dotenv_values('.env')
TOKEN = env['DISCORD_TOKEN']#os.getenv('DISCORD_TOKEN')
VGG_LOCATION = env['VGG_LOCATION']
INCEPTION_LOCATION = env['INCEPTION_LOCATION']
RESNET_LOCATION = env['RESNET_LOCATION']

device = torch.device('cpu')


class LossClient(discord.Client):
    def __init__(self):
        super(LossClient, self).__init__()
        self.PILToTensor = transforms.ToTensor()
        # get vgg ready
        self.vgg_model = VGGNet.get_instance()
        # get inception ready
        self.inception_model = InceptionNet.get_instance()
        # get resnet ready
        self.resnet_model = ResNet.get_instance()

    async def on_ready(self):
        print(f"{self.user} is ready!")
        activity = discord.Game(name="\"-LossBot help\" for info")
        await client.change_presence(status=discord.Status.online, activity=activity)

    async def on_message(self, message):
        if self.proc_condition(message):
            url = ""
            atts = message.attachments
            if len(atts) == 1:
                url = message.attachments[0].url
                if url.split('.')[-1] not in ['png', 'jpg', 'bmp', 'jpeg']:
                    await message.channel.send("I can only read detect loss in images of type .png, .jp(e)g, or .bmp.")
                else:
                    #generate a unique id to save the picture
                    id = str(uuid.uuid4())
                    if '-yes' in message.content:
                        await message.attachments[0].save(os.path.join('./data/loss/', id + '.png'))
                    elif '-no' in message.content:
                        await message.attachments[0].save(os.path.join('./data/not_loss/', id + '.png'))
                    await message.attachments[0].save(os.path.join('./img'))
                    im = Image.open('./img')
                    result = self.test_image(im)
                    if '-v' in message.content:
                        await message.channel.send("This is {:.4f}% loss and {:.4f}% not loss"
                                                   .format(result[1][0]*100, result[1][1]*100))
                    else:
                        await message.channel.send("This is {}".format(result[0]))
        elif message.content.lower() == '-lossbot help':
            await message.channel.send("Begin your message with \"Is this loss\" "
                                       "and upload an image to test alongside it.\n "
                                       "Options:\n  "
                                       "-v: print out approximate percentages\n  "
                                       "-yes: tell the bot that this is loss (doesn't affect output)\n  "
                                       "-no: tell the bot that this is not loss (doesn't affect output)")

    def proc_condition(self, message):
        if message.content.lower().startswith('is this loss'):
            return True
        elif message.content.lower().startswith('is this not loss'):
            return True

    def test_image(self, image):
        im_tensor = self.PILToTensor(self.resize_PIL(image, 299))[:3, :, :].unsqueeze(0).detach()
        results = []
        results.append(F.softmax(self.vgg_model(im_tensor), dim=1).detach().numpy())
        results.append(F.softmax(self.resnet_model(im_tensor), dim=1).detach().numpy())
        results.append(F.softmax(self.inception_model(im_tensor), dim=1).detach().numpy())
        npresults = np.array(results)
        npresults = np.mean(npresults, axis=0).flatten()
        modify = True
        if modify and npresults[1] < 0.8:
            npresults = np.array([npresults[0]-0.2, npresults[1]+.2])
        bool_result = np.argmax(npresults)
        print(npresults)
        return (1 - bool_result) * "loss" + bool_result * "not loss", npresults

    def resize_PIL(self, im, output_size):
        scale = output_size / max(im.size)
        new = Image.new(im.mode, (output_size, output_size))
        paste = im.resize((int(im.width * scale), int(im.height * scale)), resample=Image.NEAREST)
        new.paste(paste, (0, 0))
        return new


class ResNet:
    resnet_model = None

    @staticmethod
    def get_instance():
        if ResNet.resnet_model is None:
            ResNet()
        return ResNet.resnet_model

    def __init__(self):
        if self.resnet_model is not None:
            raise Exception("Tried to make another Resnet!")
        else:
            print("importing resnet...")
            self.resnet_model = models.resnet101()
            self.resnet_model.fc = nn.Linear(2048, 2)
            self.resnet_model.load_state_dict(
                torch.load(RESNET_LOCATION, map_location=device)
            )
            self.resnet_model.eval()
            ResNet.resnet_model = self.resnet_model
            print("done")


class InceptionNet:
    inception_model = None

    @staticmethod
    def get_instance():
        if InceptionNet.inception_model is None:
            InceptionNet()
        return InceptionNet.inception_model

    def get_model(self):
        return self.inception_model

    def __init__(self):
        if InceptionNet.inception_model is not None:
            raise Exception("Tried to make a duplicate of a singleton class!")
        else:
            print("importing inception...")
            self.inception_model = models.inception_v3()
            self.inception_model.AuxLogits.fc = nn.Linear(
                self.inception_model.AuxLogits.fc.in_features, 2
            )
            self.inception_model.fc = nn.Linear(self.inception_model.fc.in_features, 2)
            self.inception_model.load_state_dict(
                torch.load(INCEPTION_LOCATION, map_location=device)
            )
            self.inception_model.eval()
            InceptionNet.inception_model = self.inception_model
            print("done")


class VGGNet:
    vgg_model = None

    @staticmethod
    def get_instance():
        if VGGNet.vgg_model is None:
            VGGNet()
        return VGGNet.vgg_model

    def get_model(self):
        return self.vgg_model

    def __init__(self):
        if VGGNet.vgg_model is not None:
            raise Exception("Tried to make another VGG!")
        else:
            print("importing vgg...")
            self.vgg_model = models.vgg13_bn()
            self.vgg_model.classifier[6] = nn.Linear(
                self.vgg_model.classifier[6].in_features, 2
            )
            self.vgg_model.fc = nn.Linear(512, 2)
            self.vgg_model.load_state_dict(
                torch.load(VGG_LOCATION, map_location=device)
            )
            self.vgg_model.eval()
            VGGNet.vgg_model = self.vgg_model
            print("done")

print(f"Starting LossBot…")
client = LossClient()
client.run(TOKEN)
