"""
İşaret Dili Tanıma Modülleri
Bu paket, işaret dili tanıma için gerekli sınıf ve fonksiyonları içerir.
"""

from .sign_language_recognizer import SignLanguageRecognizer

__version__ = '1.0.0'
__author__ = 'Your Name'

# Dışa açılacak sınıf ve fonksiyonları belirt
__all__ = ['SignLanguageRecognizer']

# Bu sayede dışarıdan şöyle import edilebilir:
# from models import SignLanguageRecognizer 
