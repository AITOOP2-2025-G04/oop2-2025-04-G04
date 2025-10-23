# 音声データを音声ファイルから読み取る
audio_data.append(AudioSegment.from_file("audio-output-before.wav", format="wav"))
audio_data.append(AudioSegment.from_file("audio-output-after.wav", format="wav"))

for data in audio_data:
    sound = preprocess_audio(data)
    # Metal(GPU)が扱えるNumpy Array形式に変換
    arr = np.array(sound.get_array_of_samples()).astype(np.float32) / 32768.0
    result = mlx_whisper.transcribe(
        arr, path_or_hf_repo="whisper-base-mlx"
    )
    print(result)

