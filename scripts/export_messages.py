import sqlite3
import pandas as pd
import json
import os

# ⚠️ REPLACE WITH TARGET INFO
TARGET_HANDLE_ID = "+15550000000" 
DB_PATH = 'data/chat.db' 

def get_chat_history():
    if not os.path.exists(DB_PATH): return []
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    query = """
    SELECT message.text, message.is_from_me
    FROM message
    JOIN handle ON message.handle_id = handle.ROWID
    WHERE handle.id = ? AND message.text IS NOT NULL
    ORDER BY message.date ASC
    """
    cursor.execute(query, (TARGET_HANDLE_ID,))
    return cursor.fetchall()

raw_chats = get_chat_history()
conversation_blocks = []
current_text_buffer = []
current_is_me = None

for text, is_from_me in raw_chats:
    if current_is_me is None: current_is_me = is_from_me
    if is_from_me != current_is_me:
        full_message = "\n".join(current_text_buffer)
        conversation_blocks.append({"is_me": current_is_me, "text": full_message})
        current_text_buffer = []
        current_is_me = is_from_me
    if text:
        clean_text = text.replace('\ufffc', '').strip()
        if clean_text: current_text_buffer.append(clean_text)

if current_text_buffer:
    full_message = "\n".join(current_text_buffer)
    conversation_blocks.append({"is_me": current_is_me, "text": full_message})

training_data = []
for i in range(len(conversation_blocks) - 1):
    current_block = conversation_blocks[i]
    next_block = conversation_blocks[i+1]
    if current_block['is_me'] == 0 and next_block['is_me'] == 1:
        entry = {
            "instruction": current_block['text'],
            "input": "",
            "output": next_block['text']
        }
        training_data.append(entry)

with open('data/grouped_training_data.jsonl', 'w', encoding='utf-8') as f:
    for entry in training_data:
        json.dump(entry, f)
        f.write('\n')
print(f"✅ Exported {len(training_data)} pairs.")