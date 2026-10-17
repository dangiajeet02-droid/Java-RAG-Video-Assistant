import os
import json
import whisper

# Absolute path generation so path resolution never fails
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(BASE_DIR, "jsons")
os.makedirs(output_dir, exist_ok=True)

model = whisper.load_model("medium")
audios = os.listdir("audios")

for audio in audios:
    if (" " in audio):
        number = audio.split(" ")[0]
        title = " ".join(audio.split(" ")[1:]).rsplit(".", 1)[0]
        print(number, title)
        
        result = model.transcribe(
            audio=f"audios/{audio}",
            language="hi",
            task="translate",
            word_timestamps=False
        )
        
        chunks = []
        for segment in result["segments"]:
            chunks.append({
                "number": number,
                "title": title,
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"]
            })

        chunks_with_metadata = {"chunks": chunks, "text": result["text"]}
        
        # Absolute file write path
        file_name = f"{audio.rsplit('.', 1)[0]}.json"
        output_file = os.path.join(output_dir, file_name)
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)
        