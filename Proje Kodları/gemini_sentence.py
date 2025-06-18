import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def kelimelerden_cumle_uret(tahmin_listesi):
    try:
        print(f"Gelen tahmin listesi: {tahmin_listesi}")
        
        if not tahmin_listesi:
            return "Lütfen önce tahmin yapın!"
            
        # Gemini modelini başlat
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        # Prompt oluştur
        prompt = (
            f"Aşağıdaki kelimeleri kullanarak Türkçe, anlamlı ve doğal bir cümle kur. "
            f"LÜTFEN DİKKAT: SADECE VERİLEN KELİMELERİ KULLAN, YENİ KELİME EKLEME! "
            f"Sadece gerekli ekler (çekim ekleri) ve bağlaçlar (ve, ile, ama, fakat, çünkü, vb.) ekleyebilirsin.\n"
            f"Kelimeler: {', '.join(tahmin_listesi)}\n"
            f"Cümle:"
        )
        
        # Cümle oluştur
        response = model.generate_content(prompt)
        
        if not response.text:
            raise Exception("Model yanıt vermedi")
            
        cumle = response.text.strip()
        print(f"Oluşturulan cümle: {cumle}")
        return cumle
            
    except Exception as e:
        print(f"Hata detayı: {str(e)}")
        return f"Cümle oluşturulurken bir hata oluştu: {str(e)}" 