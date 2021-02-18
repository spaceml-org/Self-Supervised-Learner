from termcolor import colored
import torch

#internal imports
from .models import *




def load_model(args):
    #load checkpoint models
    if '.ckpt' in encoder_name:
        if technique.lower() == 'simclr':
            model = SIMCLR.load_from_checkpoint(encoder_name, **args.__dict__)
        elif technique.lower() == 'simsiam:
            model = SIMSIAM.load_from_checkpoint(encoder_name, **args.__dict__)
        elif technique.lower() == 'classifier':
            model = classifier.load_from_checkpoint(encoder_name, **args.__dict__)
        else:
            raise Exception('This is not a SIMCLR, SIMSIAM or classifier model built on curator. We cannot infer an architecture from a .ckpt file alone.')
        init_model = True


    #encoder specified
    elif 'minicnn' in encoder_name:
        #special case to make minicnn output variable output embedding size depending on user arg
        output_size =  int(''.join(x for x in encoder_name if x.isdigit()))
        encoder, embedding_size = encoders.miniCNN(output_size), output_size
        init_model = False  
    elif encoder_name == 'resnet18':
        encoder, embedding_size = encoders.resnet18(pretrained=False, first_conv=True, maxpool1=True, return_all_feature_maps=False), 512
        init_model = False
    else if encoder_name == 'imagenet_resnet18':
        encoder, embedding_size = encoders.resnet18(pretrained=True, first_conv=True, maxpool1=True, return_all_feature_maps=False), 512
        init_model = False
    else if encoder_name == 'resnet50':
        encoder, embedding_size = encoders.resnet50(pretrained=False, first_conv=True, maxpool1=True, return_all_feature_maps=False), 2048
        init_model = False
    else if encoder_name == 'imagenet_resnet50':
        encoder, embedding_size = encoders.resnet50(pretrained=True, first_conv=True, maxpool1=True, return_all_feature_maps=False), 2048
        init_model = False
    
    #try loading just the encoder
    else:
        print('Trying to initialize just the encoder from a pytorch model file (.pt)')
        try:
          model = torch.load(encoder_name)
        except:
          raise Exception('Encoder could not be loaded from path')
        try:
          embedding_size = model.embedding_size
        except:
          raise Exception('Your model specified needs to tell me its embedding size. I cannot infer output size yet. Do this by specifying a model.embedding_size in your model instance')
        init_model = False

    if not init_model:
        if technique.lower() == 'simclr':
            model = SIMCLR(**args.__dict__)
        else if technique.lower() == 'simsiam:
            model = SIMSIAM(**args.__dict__)
        else if technique.lower() == 'classifier':
            model = classifier(**args.__dict__)

        
    print(colored('LOAD ENCODER: ', 'blue'),encoder_name)
    return model
  
