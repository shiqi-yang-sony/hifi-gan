#from __future__ import absolute_import, division, print_function, unicode_literals

'''
the script is based on https://github.com/jik876/hifi-gan, MIT license
'''
import sys,glob, os, json
import numpy as np
import torch
from scipy.io.wavfile import write
#sys.path.insert(1, os.path.join("/home/zhong/RelatedWorks/hifi-gan"))

from models import Generator

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

h = None
device = None
MAX_WAV_VALUE = 32768.0

class Hifi_GAN(object):
    def __init__(self, checkpoint_file):
        # config initialization
        config_file = os.path.join(os.path.split(checkpoint_file)[0], 'config.json')
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        self.config = AttrDict(json_config)
        # cuda environment
        if torch.cuda.is_available():
            #torch.cuda.manual_seed(h.seed)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # initialize the generator
        self.generator = Generator(self.config).to(self.device)
        state_dict_g = self.load_checkpoint(checkpoint_file, self.device)
        self.generator.load_state_dict(state_dict_g['generator'])
        self.generator.eval()
        self.generator.remove_weight_norm()
        
        
    def pass_grad(self, x):
        y_g_hat = self.generator(x)
        audio = y_g_hat.squeeze()
        return audio

    def predict(self, x):
        with torch.no_grad():
            y_g_hat = self.generator(x)
            audio = y_g_hat.squeeze()
            return audio
    
    def load_checkpoint(self, filepath, device):
        assert os.path.isfile(filepath)
        print("Loading '{}'".format(filepath))
        checkpoint_dict = torch.load(filepath, map_location=device)
        print("Complete.")
        return checkpoint_dict

def debug_hifigan_class():
    checkpoint_file = "/home/zhong/RelatedWorks/hifi-gan/pretrained/VCTK_V1/generator_v1"
    mymodel = Hifi_GAN(checkpoint_file = checkpoint_file)
    input_mels_dir = '/mnt/sata1/models/FY22_SoundTagging/2_highgan-mae/VoBa_mk1/2023-02-08_07-21-51/audio/100/mels_for_tfgan'
    output_dir = "/mnt/sata1/models/FY22_SoundTagging/2_highgan-mae/VoBa_mk1/2023-02-08_07-21-51/audio/100/wave_hifigan_check" #_torchaudio
    filelist = os.listdir(input_mels_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i, filname in enumerate(filelist):
            x = np.load(os.path.join(input_mels_dir, filname))
            x = torch.FloatTensor(x).to(mymodel.device)
            audio = mymodel.predict(x)
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            output_file = os.path.join(output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
            write(output_file, mymodel.config.sampling_rate, audio)
            print(output_file)

if __name__ == '__main__':
    #main()
    debug_hifigan_class()