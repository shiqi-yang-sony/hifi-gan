#from __future__ import absolute_import, division, print_function, unicode_literals

'''
the script is based on https://github.com/jik876/hifi-gan, MIT license
'''
import sys,glob, os, json
import numpy as np
import torch
from scipy.io.wavfile import write
#from data_prep.data_prep_util import get_track
from tqdm import tqdm
sys.path.insert(1, os.path.join("/home/zhong/RelatedWorks/hifi-gan"))

from models import Generator


def get_track(data_dir, format_ext, flag_head = False):
# get the relative file path, e.g., '/7976/105575/7976-105575-0010.flac'
    list_speaker_doc_track = glob.glob(data_dir + "/**/*" + format_ext, recursive=True)
# since the data_dir is known, only the last half of the paths are stored
    if flag_head == False:
        return [f[len(data_dir):] for f in list_speaker_doc_track]
    else:
        return [f for f in list_speaker_doc_track]


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

class Scale(object):
    def __init__(self, source_min, source_max, target_min, target_max, inverse=False):
        self.source_min = source_min
        self.source_max = source_max
        self.target_min = target_min
        self.target_max = target_max
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return (x - self.target_min) / (self.target_max - self.target_min) * \
                (self.source_max - self.source_min) + self.source_min
        else:
            return (x - self.source_min) / (self.source_max - self.source_min) * \
                (self.target_max - self.target_min) + self.target_min

def debug_shiqi():
    import torchvision
    checkpoint_file = "./weights/g_01300000"# "/home/zhong/Project/FY23_Audio_MaskGIT/1_quick_start/vocoder/hifigan/g_01000000"
    mymodel = Hifi_GAN(checkpoint_file = checkpoint_file)

    #input_mels_dir = './mels/'#'/home/dataset_share/AVMAGE/train800_3e-4_DA_nocls/'
    input_mels_dir = '/home/acf15978rs/projects/weights_exp/maskP0.55_0.2-1/'
    filelist = get_track(input_mels_dir, format_ext=".pth")

    #checkpoint_file = "./weights/g_01300000"
    #mymodel = Hifi_GAN(checkpoint_file = checkpoint_file)
    #input_mels_dir = 
    output_dir = "./outputs" #_torchaudio
    #filelist = os.listdir(input_mels_dir)

    #output_dir = "/home/dataset_share/AVMAGE/train800_3e-4_DA_nocls_hifigan" #_torchaudio
    os.makedirs(output_dir, exist_ok=True)

    transforms_mel_denorm_hifigan = torchvision.transforms.Compose([
            Scale(np.log(1e-4), np.log(10), -1.0, 1.0, inverse=True),
        ])

    for i, filname in tqdm(enumerate(filelist)):
            x = torch.load(os.path.join(input_mels_dir, filname))
            print("The value range of the input: ", torch.max(x), torch.min(x))
            print("the data itself: ", x)
            #break
            x = transforms_mel_denorm_hifigan(x)
            print("The value range of the input: ", torch.max(x), torch.min(x))
            print("the data itself: ", x)
            #break
            x = torch.FloatTensor(x).to(mymodel.device)
            audio = mymodel.predict(x)
            audio = audio * MAX_WAV_VALUE
            audio = audio.cpu().numpy().astype('int16')
            output_file = os.path.join(output_dir, os.path.splitext(filname)[0] + '_generated_e2e.wav')
            write(output_file, mymodel.config.sampling_rate, audio)
            print(output_file)

def debug_hifigan_class():
    '''checkpoint_file = "/home/zhong/RelatedWorks/hifi-gan/pretrained/VCTK_V1/generator_v1"
    mymodel = Hifi_GAN(checkpoint_file = checkpoint_file)
    input_mels_dir = '/mnt/sata1/models/FY22_SoundTagging/2_highgan-mae/VoBa_mk1/2023-02-08_07-21-51/audio/100/mels_for_tfgan'
    output_dir = "/mnt/sata1/models/FY22_SoundTagging/2_highgan-mae/VoBa_mk1/2023-02-08_07-21-51/audio/100/wave_hifigan_check" #_torchaudio'''
    checkpoint_file = "./weights/g_01300000"
    mymodel = Hifi_GAN(checkpoint_file = checkpoint_file)
    input_mels_dir = './mels'
    output_dir = "./outputs" #_torchaudio
    filelist = os.listdir(input_mels_dir)
    os.makedirs(output_dir, exist_ok=True)

    for i, filname in tqdm(enumerate(filelist)):
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
    #debug_hifigan_class()
    debug_shiqi()










