
import librosa as lib
import numpy as np
from PIL import Image
import torch
import emoji as em
from pretrained import inception, preprocess
from text_emojize import EMOJIS

if __name__ == "__main__":
    # file = '/Users/amanshukla/Downloads/171372540-easy-guitars-30-seconds.wav'
    file = '/Users/amanshukla/Downloads/219124885-rainy-day-30-seconds-edit.wav'

    # file = '/Users/amanshukla/Downloads/Ed_Sheeran_-_Perfect.mp3'
    # file = '/Users/amanshukla/Downloads/John-Legend-All-Of-Me-(Gospeljingle.com).mp3'

    audio_buffer, sample_rate = lib.load(file, mono=True)
    D = lib.amplitude_to_db(np.abs(lib.stft(audio_buffer)), ref=np.max)

    mat = lib.display.specshow(D, sr=sample_rate)
   
    image = Image.fromarray(mat.get_array()).convert('RGB')
    
    trans_mat = preprocess(image)
    trans_mat = trans_mat[None,:,:,:]

    model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)

    checks = torch.load('model/t25.ckpt',map_location=torch.device('cpu'))
 
    net = inception(model)
    net.load_state_dict(checks)
    net.eval()
    
    emojis = net(trans_mat)

    _, pred_ind = torch.topk(emojis,5)

    gif_emojis = map(lambda x: EMOJIS[x], pred_ind[0])
    
    print("Emojis: {pred}".format(pred=em.emojize("{}".format(' '.join(gif_emojis)),language = 'alias')))