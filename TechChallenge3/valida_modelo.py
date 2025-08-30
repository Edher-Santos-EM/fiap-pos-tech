from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Carregar modelo
tokenizer = AutoTokenizer.from_pretrained('./qa_model_finetuned')
model = AutoModelForQuestionAnswering.from_pretrained('./qa_model_finetuned')

# Fazer predição
question = "What is The Prophet?"
context = "In a distant, timeless place, a mysterious prophet walks the sands. At the moment of his departure, he wishes to offer the people gifts but possesses nothing. The people gather round, each asks a question of the heart, and the man's wisdom is his gift. It is Gibran's gift to us, as well, for Gibran's prophet is rivaled in his wisdom only by the founders of the world's great religions. On the most basic topics--marriage, children, friendship, work, pleasure--his words have a power and lucidity that in another era would surely have provoked the description ""divinely inspired."" Free of dogma, free of power structures and metaphysics, consider these poetic, moving aphorisms a 20th-century supplement to all sacred traditions--as millions of other readers already have.--Brian Bruya--This text refers to theHardcoveredition."

inputs = tokenizer(question, context, return_tensors="pt")
outputs = model(**inputs)

answer_start = torch.argmax(outputs.start_logits)
answer_end = torch.argmax(outputs.end_logits) + 1

ids = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][answer_start:answer_end])
answer = tokenizer.convert_tokens_to_string(ids)

print(f"Answer: {answer}")