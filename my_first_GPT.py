import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def chat():
    chat_history_ids = None
    max_length = 512
    print("Hi! I'm glad to see you in this handmade GPTchat!")

    while True:
        user_input = input("Human")
        if user_input.lower() in ['выход', 'exit']:
            print("Bye-Bye! let's chat later!")
            break
        context = "You are a polite and helpful assistant. Always respond kindly and respectfully."
        input_text = f"{context}\nHuman: {user_input}\nAI:"
        new_user_input_ids = tokenizer.encode(
            f"Вы{input_text}", return_tensors='pt'
        )

        if chat_history_ids is not None:
            chat_history_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1)
            # torch.cat - Функция PyTorch для объединения (конкатенации) тензоров вдоль указанной оси.
            # Аргумент dim=-1 указывает, что объединение должно происходить по последней оси
            # (в данном случае — по размерности последовательности токенов).
            chat_history_ids = chat_history_ids[:, -max_length:]
        else:
            chat_history_ids = new_user_input_ids

        generated_token_ids = model.generate(
            chat_history_ids,
            top_k=50,
            top_p=0.90,
            num_return_sequences=1,
            do_sample=True,
            no_repeat_ngram_size=2,
            temperature=0.7,
            repetition_penalty=1.2,
            length_penalty=1.0,
            eos_token_id=50257,
            max_new_tokens=40,
            pad_token_id=tokenizer.eos_token_id,
            attention_mask=torch.ones(chat_history_ids.shape, dtype=torch.long).to(chat_history_ids.device)
        )

        answer = tokenizer.decode(
            generated_token_ids[:, chat_history_ids.shape[-1]:][0], skip_special_tokens=True
        )
        print(f"AI: {answer}")

        chat_history_ids = torch.cat([chat_history_ids, generated_token_ids[:, chat_history_ids.shape[-1]:]], dim=-1)



chat()
