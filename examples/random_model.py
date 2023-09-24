# from label_data import check_image_exists
import ast
from typing import Counter
from pandas import DataFrame, read_csv, concat
import glob
import matplotlib.pyplot as plt
import numpy as np
from text_emojize import EMOJIS
import emoji
from AppKit import NSSound
import time

import seaborn as sns

def check_image_exists(unique_image_id,path):

    if glob.glob(path + f"/{unique_image_id}.png", recursive=True):
        return True
    
    return False


def return_valid_frame(dataframe, path):
        valid = DataFrame()
        for _, row in dataframe.iterrows():
                if check_image_exists(row.image_id, path):
                        valid = valid.append(row)

        # valid['2'] = 0
        valid['11'] = 0
        valid['48'] = 0

        new_cols = ['image_id', 'emoji']+[str(i) for i in range(64)]
        valid = valid[new_cols]

        return valid

# if __name__ == '__main__':
#         data = read_csv('real.csv')
#         print(len(data))
#         new_data = return_valid_frame(data)

#         print(len(new_data))
        # print(len(new_data.columns))
        # print(new_data.head())       

def info_test():

        dataframe = read_csv('testset_labels.csv')
        hashmap = {}
        # print(dataframe.shape)
        dataframe['predicted_labels'] = dataframe['predicted_labels'].apply(lambda x: list(ast.literal_eval(x)))
        dataframe['test_image_id'] = dataframe['test_image_id'].apply(lambda x: x.split('/')[-1])
        dataframe['piece'] = dataframe['test_image_id'].apply(lambda x: ast.literal_eval(x.split('/')[-1].split('_')[1][:-4]))
        for idx, row in dataframe.iterrows():
                # Manipulate name
                name_string = row['test_image_id']
                image_name = name_string.split('_')[0]

                if image_name in hashmap:
                        hashmap[image_name] += row['predicted_labels']
                else:
                        hashmap[image_name] = row['predicted_labels']
                # Manipulate pred labels
        
        top = []
        new_hash = {}
        for k in [5]:

                box_arr = []
                for key in hashmap:
                        song_df = dataframe[dataframe.test_image_id.str.startswith(key)].sort_values('piece')
                        new_hash[key] = Counter(hashmap[key]).most_common()
                        
                        word, freq = zip(*new_hash[key])
                        
                        emojis = map(lambda x: EMOJIS[x], word[:k])
                        print("Top emoji for the song {pred}".format(pred=emoji.emojize("{}".format(' '.join(emojis)), language='alias')))
                        percent_overlap = 0
                        for _, r in song_df.iterrows():
                                # play song
                                song_piece = r['test_image_id'].split('.')[0] +'.mp3'
                                sound = NSSound.alloc()
                                sound.initWithContentsOfFile_byReference_(f'/Volumes/TOSHIBA/DALI/audios/temp_hold/{song_piece}',False)
                                sound.play()
                                time.sleep(sound.duration())
                                top5vals = r['predicted_labels'][:k]
                                percent_overlap += len(np.intersect1d(word[:k],top5vals))/k
                                topemojis = map(lambda x: EMOJIS[x],top5vals)
                                print("Top labels for piece {piece} = {val}".format(piece=r['piece'],
                                val=emoji.emojize("{}".format(' '.join(topemojis)), language='alias')))

                        # print(f'Average overlap with song level emoji = {percent_overlap/len(song_df)}')
                        box_arr.append(percent_overlap/len(song_df))
                        break
                top.append(box_arr)
        sns.boxplot(top)
        sns.pointplot(top,color='black')
        plt.xlabel('Top # emojis')
        plt.ylabel('accuracy')
        plt.show()

# info_test()