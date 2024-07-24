from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

model_name = "facebook/bart-large-cnn"
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def extract_causality(text):
    prompt = f"Identify the causality in the following text:\n\n{text}\n\nCausality:"
    inputs = tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1, early_stopping=True)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

# text = "The heavy rains caused flooding in the area, which led to the evacuation of residents."

text = """
Dr. Jonathan Hughes, a seasoned physician with over 20 years of experience, faced an unfortunate and tragic incident that would forever alter his professional reputation. It all began on a quiet Tuesday morning when Mrs. Eleanor Smith, a 60-year-old patient with a history of heart disease, came in for a routine check-up.

Dr. Hughes, known for his meticulous approach, was unusually rushed that day due to an emergency surgery scheduled in the afternoon. Consequently, he overlooked a crucial detail in Mrs. Smith’s medical history – her severe allergy to penicillin. When Mrs. Smith mentioned her recurring chest pains, Dr. Hughes quickly prescribed an antibiotic, intending to prevent any potential infections.

Unfortunately, he prescribed penicillin, which triggered a severe allergic reaction in Mrs. Smith. Within minutes of taking the medication, she began experiencing difficulty breathing and broke out in hives. The situation escalated rapidly as Mrs. Smith went into anaphylactic shock.

The nursing staff acted swiftly, administering epinephrine and calling for emergency assistance. Despite their best efforts, Mrs. Smith’s condition deteriorated, and she was rushed to the ICU. She remained in critical condition for several days, battling complications caused by the allergic reaction.

Dr. Hughes was devastated by the incident. A thorough investigation revealed that he had indeed missed the allergy warning on Mrs. Smith's chart, which was clearly marked in red. The oversight was attributed to the unusually high stress and workload he was under that day, causing him to miss a vital piece of information that could have prevented the entire ordeal.

As a result of this incident, Dr. Hughes faced a medical review board. They determined that while the mistake was not intentional, it was a clear case of negligence. The board mandated that Dr. Hughes undergo additional training in patient safety and stress management, and he was placed under a probationary period with increased supervision.

Mrs. Smith eventually recovered, but the incident left her with lingering health issues and a deep distrust of medical professionals. Dr. Hughes, once a respected and trusted physician, found himself grappling with the consequences of his mistake. He dedicated himself to improving his practice, ensuring that such an error would never occur again.

This incident serves as a poignant reminder of the critical importance of thoroughness and vigilance in the medical profession, where even the smallest oversight can lead to dire consequences.
"""

causal_relationship = extract_causality(text)
print("Extracted Causality:", causal_relationship)