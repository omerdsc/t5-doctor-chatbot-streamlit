import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch
import numpy as np

# 📦 Model ve Tokenizer'ı yükle (cache sadece model yüklemeye)
@st.cache_resource
def load_model():
    model_path = "flan_t5_qa_model"
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.eval()  # sadece tahmin moduna al
    return tokenizer, model

tokenizer, model = load_model()

# 🧠 Tahmin + Score hesaplama fonksiyonu
def generate_answer(question, context):
    input_text = f"question: {question.strip()} context: {context.strip()}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    # Cihaz uyumu (GPU varsa)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=128,
            num_beams=4,
            no_repeat_ngram_size=3,
            repetition_penalty=1.5,
            length_penalty=1.0,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=True
        )

    # Tahmin ve skor
    answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    scores = outputs.scores
    probs = [torch.nn.functional.softmax(score, dim=-1) for score in scores]
    top_probs = [prob[0][token.item()].item() for prob, token in zip(probs, outputs.sequences[0][1:])]
    confidence = np.mean(top_probs) if top_probs else 0.0

    return answer, confidence

# 🖼️ Başlık ve açıklama
st.title("💬 Medikal Soru-Cevap Asistanı")
st.markdown("Eğitilmiş generatif modelle, medikal sorularınıza doğal cevaplar alın.")

# 💡 Örnek sorular
with st.expander("💡 Örnek Sorular"):
    st.markdown("""
    - **Soru:** What should I do for my acne?  
      **Bağlam:** Hi doctor! I used to have clear skin but since I moved, I started to get lots of acne on my forehead.

    - **Soru:** Why do I feel dizzy every morning?  
      **Bağlam:** I feel dizzy when I get out of bed every morning. It only lasts a few seconds.

    - **Soru:** How to treat my constant migraines?  
      **Bağlam:** I've been having headaches almost every day and they mostly feel like migraines.
    """)

# 🧾 Girişler
question = st.text_input("❓ Soru", placeholder="E.g. What should I do for my acne?")
context = st.text_area("📄 Bağlam (Hasta açıklaması)", placeholder="E.g. I used to have clear skin but now I get lots of acne...")

# 🚀 Cevapla Butonu
if st.button("🚀 Cevapla"):
    if not question.strip() or not context.strip():
        st.warning("Lütfen hem soru hem de bağlam giriniz.")
    else:
        answer, confidence = generate_answer(question, context)
        st.success("🧠 Tahmin Edilen Cevap:")
        st.write(f"**{answer}**")
        st.markdown("---")
        st.info(f"📈 Tahmin Güveni (Confidence Score): **{confidence*100:.2f}%**")
