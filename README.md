# 🩺 Doctor-Hasta Chatbot | T5-Base ile Soru-Cevap Modeli

Bu proje, doktor-hasta etkileşimlerinden oluşan bir veri seti üzerinde eğitilen bir **Soru-Cevap (Question Answering)** sistemidir. Kullanıcıların tıbbi sorularına anlamlı yanıtlar verebilen bu yapay zeka modeli, **Google'ın T5-Base modeli** ile fine-tune edilmiştir. Modelin eğitimi sonrası, kullanıcı dostu bir arayüz geliştirmek için **Streamlit** kullanılmıştır.

> ❗️ Not: Veri seti ve eğitilmiş model bu repoya dahil edilmemiştir. Ancak veri setine aşağıdaki bağlantıdan ulaşabilirsiniz. Modeli kendi ortamınızda eğitebilirsiniz.

---

## 📊 Veri Seti

Veri seti Kaggle üzerinden temin edilmiştir:

🔗 [Doctor-Patient Chatbot Dataset (Kaggle)](https://www.kaggle.com/datasets/yousefsaeedian/ai-medical-chatbot)



## 🧠 Proje Yapısı

```bash
Doctor_ques_answer/
│
├── englishData.py              # Streamlit arayüzü (eğitilen modeli çağırır)
├── englishData_colab.ipynb     # Colab'da T5 modeli ile fine-tuning süreci
├── NLP_Sent_Anlyz.ipynb        # NLP mantığını anlatan basit duygu analizi
├── streamlit-arayüz.ipynb      # Streamlit arayüzünün notebook versiyonu

```

## Kullanılan Teknolojiler:

* T5-Base (Hugging Face Transformers)

* Pandas, NumPy

* Streamlit

* Google Colab

<img width="1919" height="919" alt="streamlit-arayüz" src="https://github.com/user-attachments/assets/e0224bb6-73e6-45ad-a00a-31e61ad8c88c" />

## Model Eğitimi:
Model: t5-base

* Ortam: Google Colab

*Eğitim dosyası: englishData_colab.ipynb

*Eğitilen model repoya dahil edilmedi.

*Modeli yeniden eğitmek isteyenler, Colab dosyasını kullanarak veri seti ile birlikte sıfırdan model oluşturabilirler.


## NLP’ye Giriş (Ekstra):
NLP_Sent_Anlyz.ipynb dosyası, doğal dil işleme (NLP) kavramını anlatan basit bir duygu analizi örneği içerir. NLP’ye yeni başlayanlar için öğretici niteliktedir.
