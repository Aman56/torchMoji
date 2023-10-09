# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 23:52:25 2023

@author: zhouz
"""

import openai

def get_musicgen():
    '''
    Run this function only once.
    This will take about 2-3min.
    '''
    from audiocraft.models import musicgen
    
    model = musicgen.MusicGen.get_pretrained('small', device='cpu')
    model.set_generation_params(duration=10)
    
    return model

def get_completion(prompt):
    '''
    Ask Tom for the API key or you can use your own OpenAI API key.
    Use GPT to genrate sentences with a given prompt

    Parameters
        ----------
        prompt : str
    '''
    openai.api_key = "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"
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]

def get_emotion_from_emoji(emoji):
    prompt = f"""
    Task: Provide an emotional, figurative, and metaphorical sentence to describe the emotion of {emoji} in the context of music composition.

    Requirements:
    1. Around 50 words.
    2. Do not mention the emoji itself in the sentence.
    3. Focus on the emotion of {emoji} and imagine some figurative situation when people can feel this emotion.
    4. Think it step by step.
    5. Imagine 5 different poets are answering this question. All poets will write down 1 step of thinking and then share it with the group at each step. The group of poets will then vote to determine whose thinking step is the best one and will go on with that step.
    6. The final output should be the answer with the highest vote.

    Sentence: A piece of music that feels like...

    Emoji: 😇
    Sentence: A piece of music that transports the listener to a celestial realm, where ethereal melodies dance on a bed of shimmering notes. It evokes a sense of serenity and tranquility, like a gentle breeze caressing the soul. The harmonies soar with angelic grace, enveloping the heart in a warm embrace, leaving behind a trail of pure bliss and heavenly euphoria.

    Emoji: 😍
    Sentence: A piece of music that feels like a passionate love affair, where every note is a tender caress and every melody is a whispered declaration of adoration. It ignites a fire within the listener's heart, filling them with a euphoric longing and a deep sense of connection. The harmonies intertwine like lovers' bodies, creating a symphony of desire that leaves the soul enraptured and yearning for more.

    Emoji {emoji}
    Sentence:
    """

    emotion = get_completion(prompt)
    return emotion

def get_genre_from_emotion(emotion):
    prompt = f"""
    Task: Provide 1 music subgenre that best matches the given emotion.

    Requirements:
    1. Your response only contains the name of the music subgenre.
    2. Your response should be a subgenre instead of a main genre.
    3. Be artistic and creative with what subgenre to choose.
    4. Think it in 2 steps. In the first step, find the best fit music genre. In the second step, find the best sub-genre within the genre you came up in the first step
    5. Imagine 5 different music lovers are answering this question. All music lovers will write down 1 step of thinking and then share it with the group at each step. The group of music lovers will then vote to determine whose thinking step is the best one and will go on with that step.
    6. The final output should be the answer with the highest vote.

    Emotion: A piece of music that feels like a passionate love affair, where every note is a tender caress and every melody is a whispered declaration of adoration. It ignites a fire within the listener's heart, filling them with a euphoric longing and a deep sense of connection. The harmonies intertwine like lovers' bodies, creating a symphony of desire that leaves the soul enraptured and yearning for more.
    Subgenre: Neo-Soul

    Emotion:{emotion}
    Subgenre:
    """

    subgenre = get_completion(prompt)
    return subgenre

def get_prompt(emoji):
    emotion = get_emotion_from_emoji(emoji)
    subgenre = get_genre_from_emotion(emotion)

    prompt = f"""
    Task: Write 1 response with professional music terms to describe a piece of music that best matches the given genre and emotion.

    Requirements:
    1. Around 50 words
    2. Only include 5 parameters in your response. [Instrument: , Timbre: , Chord: , Drumbeat: , Tempo:]
    3. Use 3 to 5 professional music theory, music composition, and music production terms in each of the parameters in your response.
    4. Format the response in this template: Instrument: , Timbre: , Chord: , Drumbeat: , Tempo:
    5. Think it step by step.
    6. Imagine 5 different professional musicians, composers, and music producers are answering this question. All people will write down 1 step of thinking and then share it with the group at each step. The group of people will then vote to determine whose thinking step is the best one and will go on with that step.
    7. The final output should be the answer with the highest vote.

    Genre: Funk-Rock
    Emotion: A piece of music that unleashes a tempest of electrifying sensations, like a thunderstorm of sonic brilliance, where the harmonies crash and collide with a mind-bending force, leaving you in a state of awe and bewilderment. It's a kaleidoscope of euphoria and chaos, a symphony that shatters the boundaries of your imagination.
    Prompt: Instrument: Electric guitar, bass, drums, and brass section, Timbre: Rich and vibrant, Chords: Groovy and funky, Drumbeat: Lively and syncopated, Tempo: 100 to 120.

    Genre: Ambient-Chillout
    Emotion: A piece of music that transports the listener to a celestial realm, where ethereal melodies dance on a bed of shimmering notes. It evokes a sense of serenity and tranquility, like a gentle breeze caressing the soul. The harmonies soar with angelic grace, enveloping the heart in a warm embrace, leaving behind a trail of pure bliss and heavenly euphoria.
    Prompt: Instrument: Synthesizers, pads, and atmospheric sounds. Timbre: Ethereal and dreamy. Chords: Simple and spacious, Drumbeat: Subtle and minimal, with soft and gentle rhythms. Tempo: around 60 to 80.

    Genre: Neo-Soul
    Emotion: A piece of music that feels like a passionate love affair, where every note is a tender caress and every melody is a whispered declaration of adoration. It ignites a fire within the listener's heart, filling them with a euphoric longing and a deep sense of connection. The harmonies intertwine like lovers' bodies, creating a symphony of desire that leaves the soul enraptured and yearning for more.
    Prompt: Instrument: Electric piano, bass guitar, drums, and soulful vocals. Timbre: Warm and soulful. Chords: Smooth and jazzy, with rich and colorful voicings. Drumbeat: Groove-based and laid-back, with syncopated rhythms. Tempo: Slow to medium, around 70 to 90.

    Genre: {subgenre}
    Emotion: {emotion}
    Prompt:
    """

    response = get_completion(prompt)
    return (subgenre, emotion, response)

def gen_and_save(emoji):
    '''
    Use music Gen to generate a piece of music from a given emoji
    '''
    from flask import current_app
    import os
    import torchaudio
    import torch   
    import datetime

    timestamp = str(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    model = get_musicgen()
    subgenre, emotion, prompt = get_prompt(emoji)
    res = model.generate([f"""Produce {emotion} Genre: {subgenre}, {prompt}"""], progress=True)
    out_f = emoji + "_" + timestamp + 'emonew.wav'
    file_path = os.path.join(current_app.config['MUSIC_FOLDER'], out_f)  
    print(file_path) 
    torchaudio.save(file_path, res[0].to("cpu"), sample_rate = 32000)
    temp_file_path = os.path.join(current_app.config['retrieve_MUSIC_FOLDER'], out_f) 
