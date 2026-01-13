import json
import re
import os

input_file = "data/grouped_training_data.jsonl"
output_file = "data/clean_training_data.jsonl"
reaction_pattern = r"^(Loved|Emphasized|Disliked|Laughed at|Questioned|Reacted|Liked) .*"

def clean_text_content(text):
    if not text: return ""
    text = text.replace('“', '"').replace('”', '"').replace("’", "'").replace("‘", "'")
    text = text.replace('\ufffc', '')
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'\s+', ' ', text).strip()
    return text

if not os.path.exists(input_file): exit()

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        try:
            data = json.loads(line)
            if re.match(reaction_pattern, data["output"]): continue
            
            clean_instr = clean_text_content(data["instruction"])
            clean_out = clean_text_content(data["output"])

            if clean_out and clean_instr:
                outfile.write(json.dumps({
                    "instruction": clean_instr,
                    "input": "",
                    "output": clean_out
                }) + "\n")
        except: continue
print("✅ Done.")