document.addEventListener('DOMContentLoaded', () => {
    const welcomePage = document.getElementById('welcomePage');
    const loginPage = document.getElementById('loginPage');
    const video = document.getElementById('video');
    const textOutput = document.getElementById('textOutput');
    const predictButton = document.getElementById('predictButton');
    const countdownElement = document.getElementById('countdown');
    const cameraToggle = document.getElementById('cameraToggle');
    const userDisplay = document.getElementById('userDisplay');
    let ws = null;
    let streamCanvas = document.createElement('canvas');
    let streamContext = streamCanvas.getContext('2d');
    let isRecording = false;
    let frameBuffer = [];
    let isCameraOn = false;
    let currentStream = null;
    let currentUser = null;
    let frameInterval = null;
    const FRAME_INTERVAL = 100; // Her 100ms'de bir frame gönder
    const RECORDING_DURATION = 5; // 5 saniye
    const FRAME_RATE = 10; // Saniyede 10 frame

    // Kullanıcı verilerini localStorage'dan yükle
    let users = JSON.parse(localStorage.getItem('users')) || {};

    // EmailJS başlatma
    emailjs.init("TKoGpZLQJr-V-7cVq");

    // Hoş geldiniz sayfasına tıklama olayını dinle
    welcomePage.addEventListener('click', () => {
        welcomePage.classList.add('slide-up');
        loginPage.style.display = 'flex';
    });
    
    // Kamerayı başlat
    async function startCamera() {
        console.log("Kamera başlatılıyor...");
        if (!isCameraOn) {
            try {
                const constraints = {
                    video: {
                        width: { ideal: 640 },
                        height: { ideal: 480 },
                        facingMode: 'user'
                    },
                    audio: false
                };

                const stream = await navigator.mediaDevices.getUserMedia(constraints);
                console.log("Kamera erişimi başarılı");
                
                video.srcObject = stream;
                currentStream = stream;
                
                // Kamera durumu yazısını güncelle
                const cameraStatus = document.getElementById('cameraStatus');
                if (cameraStatus) {
                    cameraStatus.textContent = 'Açık';
                }
                
                // Video elementinin arka plan rengini sıfırla
                video.style.backgroundColor = '';
                
                // Kamera toggle butonunu açık konuma getir
                const cameraToggle = document.getElementById('cameraToggle');
                if (cameraToggle) {
                    cameraToggle.checked = true;
                }
                
                // Video yüklendiğinde
                video.onloadedmetadata = () => {
                    console.log("Video metadata yüklendi");
                    video.play()
                        .then(() => {
                            console.log("Video oynatma başladı");
                            isCameraOn = true;
                            video.style.display = 'block';
                            
                            // Canvas boyutlarını ayarla
                            streamCanvas.width = video.videoWidth;
                            streamCanvas.height = video.videoHeight;
                            
                            // WebSocket bağlantısını başlat
                            if (!ws || ws.readyState !== WebSocket.OPEN) {
                                initializeWebSocket();
                            }
                        })
                        .catch(err => {
                            console.error("Video oynatma hatası:", err);
                            alert("Video başlatılamadı. Lütfen sayfayı yenileyip tekrar deneyin.");
                        });
                };
            } catch (err) {
                console.error('Kamera erişimi hatası:', err);
                alert('Kamera erişimi sağlanamadı. Lütfen kamera izinlerini kontrol edin ve sayfayı yenileyin.');
            }
        }
    }

    // WebSocket bağlantısını başlat
    function initializeWebSocket() {
        if (ws) {
            console.log("Mevcut WebSocket bağlantısı kapatılıyor...");
            ws.close();
        }

        console.log("WebSocket bağlantısı başlatılıyor...");
        ws = new WebSocket('ws://127.0.0.1:8767');
        
        ws.onopen = function() {
            console.log("WebSocket bağlantısı kuruldu");
            textOutput.value = 'Cümleye çevirmek için lütfen butona tıklayınız...';
            textOutput.style.display = 'block';
            textOutput.style.color = '#888';
            startFrameCapture(); // Bağlantı kurulduğunda frame yakalamayı başlat
        };

        ws.onmessage = function(event) {
            try {
                console.log("Ham veri alındı:", event.data);
                const data = JSON.parse(event.data);
                console.log("İşlenmiş veri:", data);

                if (data.prediction) {
                    const languageToggle = document.querySelector('.language-toggle');
                    const isEnglish = languageToggle.classList.contains('english');
                    
                    // Sadece kelimeyi al
                    const prediction = isEnglish ? data.en_label : data.prediction;
                    const confidence = data.confidence;
                    console.log("Yeni tahmin:", prediction, "Güven:", confidence);
                    
                    // Tahmini sadece listeye ekle
                    if (prediction && confidence > 0.35) {
                        addToPredictionsList(prediction);
                    }
                } else if (data.sentence) {
                    // GPT'den gelen cümleyi göster
                    textOutput.value = data.sentence;
                    textOutput.style.color = '#222';
                } else if (data.error) {
                    console.error('Sunucu hatası:', data.error);
                    textOutput.value = 'Hata: ' + data.error;
                }
            } catch (e) {
                console.error('Mesaj işleme hatası:', e);
                console.error('Ham veri:', event.data);
                textOutput.value = 'Mesaj işleme hatası oluştu';
            }
        };

        ws.onerror = function(error) {
            console.error('WebSocket hatası:', error);
            textOutput.value = 'Bağlantı hatası oluştu!';
            stopFrameCapture();
        };

        ws.onclose = function() {
            console.log('WebSocket bağlantısı kapandı');
            textOutput.value = 'Bağlantı kesildi, yeniden bağlanılıyor...';
            stopFrameCapture();
            
            // 3 saniye sonra yeniden bağlanmayı dene
            setTimeout(() => {
                console.log("WebSocket yeniden bağlanıyor...");
                initializeWebSocket();
            }, 3000);
        };
    }

    // Frame yakalama işlemi
    function startFrameCapture() {
        console.log("Frame yakalama başlatılıyor...");
        if (frameInterval) {
            clearInterval(frameInterval);
        }
        
        frameInterval = setInterval(() => {
            if (isCameraOn && ws && ws.readyState === WebSocket.OPEN) {
                sendFrame();
            }
        }, FRAME_INTERVAL);
    }

    // Frame yakalamayı durdur
    function stopFrameCapture() {
        if (frameInterval) {
            clearInterval(frameInterval);
            frameInterval = null;
        }
    }

    // Frame gönderme işlemi
    async function sendFrame() {
        if (!isCameraOn || !ws || ws.readyState !== WebSocket.OPEN) {
            console.log("Frame gönderilemiyor: Kamera veya WebSocket hazır değil");
            return;
        }

        return new Promise((resolve, reject) => {
            try {
                streamContext.drawImage(video, 0, 0, streamCanvas.width, streamCanvas.height);
                const imageData = streamCanvas.toDataURL('image/jpeg', 0.7);
                
                console.log("Frame yakalandı ve dönüştürüldü");
                const message = JSON.stringify({ 
                    image: imageData,
                    timestamp: Date.now(),
                    isRecording: isRecording
                });
                
                ws.send(message);
                console.log("Frame gönderildi, boyut:", message.length);
                
                resolve();
            } catch (e) {
                console.error('Frame gönderme hatası:', e);
                reject(e);
            }
        });
    }

    // Tahmin işlemi
    async function startPrediction() {
        console.log("Tahmin işlemi başlatılıyor...");
        
        if (!isCameraOn) {
            console.error("Kamera açık değil");
            alert('Lütfen önce kamerayı başlatın.');
            return;
        }

        if (!ws || ws.readyState !== WebSocket.OPEN) {
            console.error("WebSocket bağlantısı yok veya açık değil");
            alert('Sunucu bağlantısı kurulamadı. Lütfen sayfayı yenileyin.');
            return;
        }

        predictButton.disabled = true;
        isRecording = true;
        console.log("Tahmin başlatılıyor...");
        
        try {
            // Geri sayımı başlat
            countdownElement.classList.add('active');
            console.log("Geri sayım başlıyor...");
            
            // 5 saniye boyunca kayıt
            for (let i = 3; i > 0; i--) {
                countdownElement.textContent = i;
                console.log(`Geri sayım: ${i}`);
                
                // Her saniyede 3 frame gönder
                for (let j = 0; j < 3; j++) {
                    if (isRecording) {
                        try {
                            await sendFrame();
                            await new Promise(resolve => setTimeout(resolve, 333)); // 333ms bekle
                        } catch (error) {
                            console.error('Frame gönderme hatası:', error);
                        }
                    }
                }
            }

            console.log("Geri sayım tamamlandı");
            isRecording = false;
            countdownElement.textContent = 'Tahmin yapılıyor...';
            console.log("Son tahmin bekleniyor...");
            
            // Son frame'i gönder ve sonucu bekle
            try {
                await sendFrame();
                console.log("Son frame gönderildi, tahmin bekleniyor...");
            } catch (error) {
                console.error('Son frame gönderme hatası:', error);
            }
            
            // 2 saniye bekle
            await new Promise(resolve => setTimeout(resolve, 2000));
        } catch (error) {
            console.error('Tahmin işlemi hatası:', error);
        } finally {
            countdownElement.classList.remove('active');
            countdownElement.textContent = '';
            predictButton.disabled = false;
            console.log("Tahmin işlemi tamamlandı");
        }
    }

    // Sayfa geçişleri
    window.showPage = function(pageId) {
        // Eğer ana sayfa gösterilmek isteniyor ve oturum açıksa, doğrudan kamera sayfasına yönlendir
        if (pageId === 'welcomePage' && getSession()) {
            pageId = 'cameraPage';
        }
        // Önce tüm sayfaları gizle
        document.querySelectorAll('.page, .login-page').forEach(page => {
            page.style.display = 'none';
        });
        // İstenen sayfayı göster
        const targetPage = document.getElementById(pageId);
        if (targetPage) {
            targetPage.style.display = 'flex';
            if (pageId === 'cameraPage') {
                startCamera();
            }
        }
    };

    // Giriş işlemi
    window.login = function() {
        const username = document.getElementById('loginUsername').value.trim();
        const password = document.getElementById('loginPassword').value.trim();
        let users = JSON.parse(localStorage.getItem('users')) || {};
        console.log('Kullanıcılar:', users);
        console.log('Giriş:', username, password);

        if (!username || !password) {
            alert('Lütfen tüm alanları doldurun!');
            return;
        }
        if (users[username] && users[username].password === password) {
            currentUser = username;
            setSession(username);
            setUserDropdown(username);
            showPage('cameraPage');
        } else {
            alert('Kullanıcı adı veya şifre hatalı!');
        }
    };

    // Kayıt işlemi
    window.register = function() {
        const username = document.getElementById('registerUsername').value.trim();
        const email = document.getElementById('registerEmail').value.trim();
        const password = document.getElementById('registerPassword').value.trim();
        const confirmPassword = document.getElementById('confirmPassword').value.trim();
        if (!username || !email || !password || !confirmPassword) {
            alert('Lütfen tüm alanları doldurun!');
            return;
        }
        if (password.length <=8) {
            alert('Şifreniz en az 8 karakter olmalıdır!');
            return;
        }
        if (!validateEmail(email)) {
            alert('Geçerli bir e-posta adresi giriniz!');
            return;
        }
        // Şifre güvenlik kontrolü
        if (!checkPasswordStrength(password).valid) {
            passwordStrength.textContent = 'Çok zayıf';
            alert('Şifreniz çok zayıf! Lütfen daha güçlü bir şifre girin.');
            return;
        }
        if (password !== confirmPassword) {
            alert('Şifreler eşleşmiyor!');
            return;
        }
        if (users[username]) {
            alert('Bu kullanıcı adı zaten kullanılıyor!');
            return;
        }
        users[username] = { password: password, email: email };
        localStorage.setItem('users', JSON.stringify(users));
        currentUser = username;
        setSession(username);
        setUserDropdown(username);
        showPage('cameraPage');
    };

    // E-posta doğrulama fonksiyonu
    function validateEmail(email) {
        const re = /^(([^<>()\[\]\\.,;:\s@\"]+(\.[^<>()\[\]\\.,;:\s@\"]+)*)|(".+"))@(([^<>()[\]\\.,;:\s@\"]+\.)+[^<>()[\]\\.,;:\s@\"]{2,})$/i;
        return re.test(email);
    }

    // Şifre güvenlik kontrolü
    const registerPassword = document.getElementById('registerPassword');
    const passwordStrength = document.getElementById('passwordStrength');
    registerPassword.addEventListener('input', function() {
        const pwd = registerPassword.value;
        const result = checkPasswordStrength(pwd);
        if (!pwd) {
            passwordStrength.textContent = '';
            passwordStrength.style.color = '#e53935';
        } else if (!result.valid) {
            passwordStrength.textContent = 'Çok zayıf';
            passwordStrength.style.color = '#e53935';
        } else {
            passwordStrength.textContent = 'Güçlü';
            passwordStrength.style.color = '#43a047';
        }
    });

    function checkPasswordStrength(password) {
        if (password.length < 8) return { valid: false };
        // Aynı karakter tekrarı
        if (/^(.)\1{5,}$/.test(password)) return { valid: false };
        // Dinamik ardışık sayı veya harf kontrolü (artan/azalan, en az 5 karakter)
        for (let i = 0; i <= password.length - 5; i++) {
            let asc = true, desc = true;
            for (let j = 0; j < 4; j++) {
                const curr = password.charCodeAt(i + j);
                const next = password.charCodeAt(i + j + 1);
                if (next - curr !== 1) asc = false;
                if (curr - next !== 1) desc = false;
            }
            if (asc || desc) return { valid: false };
        }
        // Büyük harf, sayı, sembol kontrolü
        const hasUpper = /[A-Z]/.test(password);
        const hasNumber = /[0-9]/.test(password);
        const hasSymbol = /[^A-Za-z0-9]/.test(password);
        if (!(hasUpper && hasNumber && hasSymbol)) return { valid: false };
        return { valid: true };
    }

    // Misafir olarak devam et
    window.continueAsGuest = function() {
        currentUser = 'Misafir';
        setSession('Misafir');
        setUserDropdown('Misafir');
        showPage('cameraPage');
    };

    // Event listeners
    predictButton.addEventListener('click', startPrediction);
    
    // Dil değiştirme
    window.toggleLanguage = function() {
        const toggle = document.querySelector('.language-toggle');
        const textOutput = document.getElementById('textOutput');
        const predictionsList = document.getElementById('predictionsList');
        const items = predictionsList.getElementsByClassName('prediction-item');
        
        toggle.classList.toggle('turkish');
        toggle.classList.toggle('english');
        
        // Eğer metin kutusunda bir cümle varsa, dili değiştir
        if (textOutput.value && textOutput.value.includes(' ')) {
            const currentText = textOutput.value;
            if (toggle.classList.contains('english')) {
                // Türkçeden İngilizceye çevir
                translateText(currentText, 'tr', 'en')
                    .then(translatedText => {
                        textOutput.value = translatedText;
                    })
                    .catch(error => {
                        console.error('Çeviri hatası:', error);
                    });
            } else {
                // İngilizceden Türkçeye çevir
                translateText(currentText, 'en', 'tr')
                    .then(translatedText => {
                        textOutput.value = translatedText;
                    })
                    .catch(error => {
                        console.error('Çeviri hatası:', error);
                    });
            }
        }
    };

    // Çeviri fonksiyonu
    async function translateText(text, fromLang, toLang) {
        try {
            const response = await fetch(`https://translate.googleapis.com/translate_a/single?client=gtx&sl=${fromLang}&tl=${toLang}&dt=t&q=${encodeURIComponent(text)}`);
            const data = await response.json();
            return data[0].map(item => item[0]).join('');
        } catch (error) {
            console.error('Çeviri API hatası:', error);
            return text;
        }
    }

    // Kamera toggle switch kontrolü
    cameraToggle.addEventListener('change', function() {
        if (this.checked) {
            startCamera();
        } else {
            stopCamera();
        }
    });

    // Kamera durdurma fonksiyonu
    function stopCamera() {
        if (currentStream) {
            currentStream.getTracks().forEach(track => track.stop());
            video.srcObject = null;
            isCameraOn = false;
            currentStream = null;
            
            // Video elementini siyah yap ve "Kamera Kapalı" yazısı ekle
            video.style.backgroundColor = 'black';
            video.style.display = 'block';
            
            // Kamera durumu yazısını güncelle
            const cameraStatus = document.getElementById('cameraStatus');
            if (cameraStatus) {
                cameraStatus.textContent = 'Kapalı';
            }
            
            stopFrameCapture();
            
            // WebSocket bağlantısını kapat
            if (ws) {
                ws.close();
                ws = null;
            }
        }
    }

    // Tahmin listesi için değişkenler
    let predictions = [];

    // Tahmin listesine yeni tahmin ekleme
    function addToPredictionsList(prediction) {
        if (!prediction) {
            console.error("Tahmin değeri boş olamaz");
            return;
        }
        
        const predictionsList = document.getElementById('predictionsList');
        if (!predictionsList) {
            console.error("predictionsList elementi bulunamadı");
            return;
        }
        
        // Eğer aynı tahmin zaten varsa ekleme
        if (predictions.includes(prediction)) {
            console.log("Bu tahmin zaten listede var:", prediction);
            return;
        }
        
        const predictionItem = document.createElement('div');
        predictionItem.className = 'prediction-item';
        predictionItem.innerHTML = `
            <div class="prediction-text">${prediction}</div>
            <button class="delete-btn" onclick="removePrediction('${prediction}')">
                <i class="fas fa-times"></i>
            </button>
        `;
        predictionsList.appendChild(predictionItem);
        predictions.push(prediction);
        console.log("Tahmin listeye eklendi:", prediction);
    }

    // Tahmini listeden kaldırma
    window.removePrediction = function(prediction) {
        const predictionsList = document.getElementById('predictionsList');
        const items = predictionsList.getElementsByClassName('prediction-item');
        for (let i = 0; i < items.length; i++) {
            if (items[i].querySelector('.prediction-text').textContent === prediction) {
                items[i].remove();
                break;
            }
        }
        predictions = predictions.filter(p => p !== prediction);
        console.log("Tahmin listeden silindi:", prediction);
    };

    // Cümleye çevirme
    document.getElementById('convertToSentence').addEventListener('click', async function() {
        if (predictions.length === 0) {
            alert('Lütfen önce tahmin yapın!');
            return;
        }

        try {
            // Tahmin listesini sunucuya gönder
            const message = JSON.stringify({
                predictions: predictions
            });
            
            ws.send(message);
            
            // Sunucudan gelen cevabı bekle
            const response = await new Promise((resolve, reject) => {
                const timeout = setTimeout(() => {
                    reject(new Error('Zaman aşımı'));
                }, 10000);

                const messageHandler = (event) => {
                    try {
                        const data = JSON.parse(event.data);
                        if (data.sentence) {
                            clearTimeout(timeout);
                            ws.removeEventListener('message', messageHandler);
                            resolve(data.sentence);
                        } else if (data.error) {
                            clearTimeout(timeout);
                            ws.removeEventListener('message', messageHandler);
                            reject(new Error(data.error));
                        }
                    } catch (e) {
                        // JSON parse hatası, mesajı yoksay
                    }
                };

                ws.addEventListener('message', messageHandler);
            });

            // Oluşturulan cümleyi göster
            textOutput.value = response;
        } catch (error) {
            console.error('Cümle oluşturma hatası:', error);
            textOutput.value = 'Cümle oluşturulurken bir hata oluştu: ' + error.message;
        }
    }); 

    // Metni seslendirme
    document.getElementById('speakText').addEventListener('click', function() {
        const text = textOutput.value.trim();
        if (!text) {
            alert('Seslendirilecek metin bulunamadı!');
            return;
        }

        // Web Speech API'yi kullan
        const utterance = new SpeechSynthesisUtterance(text);
        
        // Dil ayarını belirle
        const languageToggle = document.querySelector('.language-toggle');
        utterance.lang = languageToggle.classList.contains('english') ? 'en-US' : 'tr-TR';
        
        // Ses hızını ve tonunu ayarla
        utterance.rate = 1.0; // Konuşma hızı (0.1 - 10)
        utterance.pitch = 1.0; // Ses tonu (0 - 2)
        utterance.volume = 1.0; // Ses seviyesi (0 - 1)

        // Seslendirmeyi başlat
        window.speechSynthesis.speak(utterance);

        // Buton durumunu güncelle
        const speakButton = this;
        speakButton.disabled = true;
        speakButton.innerHTML = '<i class="fas fa-volume-up"></i> Seslendiriliyor...';

        // Seslendirme bittiğinde butonu normal haline getir
        utterance.onend = function() {
            speakButton.disabled = false;
            speakButton.innerHTML = '<i class="fas fa-volume-up"></i> Seslendir';
        };

        // Hata durumunda butonu normal haline getir
        utterance.onerror = function() {
            speakButton.disabled = false;
            speakButton.innerHTML = '<i class="fas fa-volume-up"></i> Seslendir';
            alert('Seslendirme sırasında bir hata oluştu!');
        };
    });

    // --- OTURUM KONTROLÜ ve DROPDOWN MENÜ ---
    const userDropdown = document.getElementById('userDropdown');
    const userButton = document.getElementById('userButton');
    const userNameText = document.getElementById('userNameText');
    const dropdownMenu = document.getElementById('dropdownMenu');

    // Oturum bilgisini localStorage'da tut
    function setSession(user) {
        localStorage.setItem('activeUser', user);
    }
    function getSession() {
        return localStorage.getItem('activeUser');
    }
    function clearSession() {
        localStorage.removeItem('activeUser');
    }

    // Sayfa yüklendiğinde oturum kontrolü
    window.addEventListener('DOMContentLoaded', () => {
        const activeUser = getSession();
        if (activeUser) {
            // Oturum varsa direkt kamera sayfasına atla
            showPage('cameraPage');
            setUserDropdown(activeUser);
        }
    });

    // Dropdown menü aç/kapa
    userButton.addEventListener('click', function(e) {
        e.stopPropagation();
        dropdownMenu.classList.toggle('active');
        userButton.classList.toggle('open');
    });
    document.addEventListener('click', function(e) {
        if (!userDropdown.contains(e.target)) {
            dropdownMenu.classList.remove('active');
            userButton.classList.remove('open');
        }
    });

    // Dropdown içeriğini güncelle
    function setUserDropdown(user) {
        if (user === 'Misafir') {
            userNameText.textContent = 'Misafir Kullanıcı';
            dropdownMenu.innerHTML = `<button class="dropdown-btn-guest" onclick="goToLogin()">Giriş Yap / Kaydol</button>`;
        } else {
            userNameText.textContent = user;
            dropdownMenu.innerHTML = `
                <button class="dropdown-btn" onclick="goToProfile()">Profil Bilgileri</button>
                <button class="dropdown-btn" onclick="logout()">Çıkış Yap</button>
            `;
        }
    }

    // Çıkış işlemi
    window.logout = function() {
        clearSession();
        dropdownMenu.classList.remove('active');
        userButton.classList.remove('open');
        showPage('loginPage');
    };

    // Giriş/Kaydol sayfasına yönlendir
    window.goToLogin = function() {
        dropdownMenu.classList.remove('active');
        userButton.classList.remove('open');
        showPage('loginPage');
    };

    // Profil sayfasına geçiş
    window.goToProfile = function() {
        dropdownMenu.classList.remove('active');
        userButton.classList.remove('open');
        const activeUser = getSession();
        if (!activeUser || !users[activeUser]) return;
        document.getElementById('editUsername').value = activeUser;
        document.getElementById('editEmail').value = users[activeUser].email;
        document.getElementById('editPassword').value = '';
        document.getElementById('currentPassword').value = '';
        const profileSuccess = document.getElementById('profileSuccess');
        if (profileSuccess) {
            profileSuccess.style.display = 'none';
        }
        showPage('profilePage');
    };

    // Profilde yeni şifre gücü kontrolü
    const editPassword = document.getElementById('editPassword');
    const editPasswordStrength = document.getElementById('editPasswordStrength');
    if (editPassword && editPasswordStrength) {
        editPassword.addEventListener('input', function() {
            const pwd = editPassword.value;
            if (!pwd) {
                editPasswordStrength.textContent = '';
                editPasswordStrength.style.color = '#e53935';
            } else if (!checkPasswordStrength(pwd).valid) {
                editPasswordStrength.textContent = 'Çok zayıf';
                editPasswordStrength.style.color = '#e53935';
            } else {
                editPasswordStrength.textContent = 'Güçlü';
                editPasswordStrength.style.color = '#43a047';
            }
        });
    }

    // Profil kaydetme işlemi: yeni şifre girildiyse kontrol et
    const saveProfileBtn = document.getElementById('saveProfileBtn');
    const profileSuccessOverlay = document.getElementById('profileSuccessOverlay');
    saveProfileBtn.addEventListener('click', function() {
        const oldUsername = getSession();
        if (!oldUsername || !users[oldUsername]) return;
        const newUsername = document.getElementById('editUsername').value.trim();
        const newEmail = document.getElementById('editEmail').value.trim();
        const newPassword = document.getElementById('editPassword').value.trim();
        const currentPassword = document.getElementById('currentPassword').value.trim();
        if (!newUsername || !newEmail || !currentPassword) {
            alert('Lütfen tüm alanları doldurun!');
            return;
        }
        if (!validateEmail(newEmail)) {
            alert('Geçerli bir e-posta adresi giriniz!');
            return;
        }
        if (newPassword && !checkPasswordStrength(newPassword).valid) {
            editPasswordStrength.textContent = 'Çok zayıf';
            editPasswordStrength.style.color = '#e53935';
            alert('Yeni şifreniz çok zayıf! Lütfen daha güçlü bir şifre girin.');
            return;
        }
        if (users[oldUsername].password !== currentPassword) {
            alert('Güncel şifreniz yanlış!');
            return;
        }
        if (newUsername !== oldUsername && users[newUsername]) {
            alert('Bu kullanıcı adı zaten kullanılıyor!');
            return;
        }
        const updatedUser = {
            password: newPassword ? newPassword : users[oldUsername].password,
            email: newEmail
        };
        if (newUsername !== oldUsername) {
            delete users[oldUsername];
        }
        users[newUsername] = updatedUser;
        localStorage.setItem('users', JSON.stringify(users));
        setSession(newUsername);
        setUserDropdown(newUsername);
        profileSuccessOverlay.classList.add('active');
        setTimeout(() => {
            profileSuccessOverlay.classList.remove('active');
            showPage('cameraPage');
        }, 3000);
    });

    // Ana sayfa butonu fonksiyonu
    window.goHome = function() {
        window.location.reload();
    };

    

    // Kullanıcıları localStorage'dan al
    function getUsers() {
        return JSON.parse(localStorage.getItem("users")) || {};
    }

    // localStorage'a kullanıcıları kaydet
    function saveUsers(users) {
        localStorage.setItem("users", JSON.stringify(users));
    }

    // Daktilo animasyonu
    function typeWriterEffect(element, text, speed = 45) {
        element.textContent = '';
        let i = 0;
        function type() {
            if (i < text.length) {
                element.textContent += text.charAt(i);
                i++;
                setTimeout(type, speed);
            }
        }
        type();
    }

    // Hoş Geldiniz sayfası için daktilo animasyonu başlat
    const typewriterEl = document.querySelector('.typewriter-text');
    if (typewriterEl) {
        typeWriterEffect(typewriterEl, 'Sessiz Bir Dil, Evrensel Bir Anlatım', 45);
    }

    // Şifre sıfırlama için gerekli değişkenler
    let resetCode = null;
    let resetEmail = null;
    let resetUsername = null;

    // Şifre sıfırlama kodu gönderme
    window.sendResetCode = async function() {
        const email = document.getElementById('resetEmail').value.trim();
        if (!email) {
            alert('Lütfen e-posta adresinizi girin!');
            return;
        }

        // E-posta adresinin kayıtlı olup olmadığını kontrol et
        const users = getUsers();
        let foundUser = null;
        for (const [username, userData] of Object.entries(users)) {
            if (userData.email === email) {
                foundUser = username;
                break;
            }
        }

        if (!foundUser) {
            alert('Bu e-posta adresi ile kayıtlı bir hesap bulunamadı!');
            return;
        }

        // 6 haneli random kod oluştur
        resetCode = Math.floor(100000 + Math.random() * 900000).toString();
        resetEmail = email;
        resetUsername = foundUser;

        // EmailJS ile e-posta gönder
        try {
            const templateParams = {
                email: email,
                code: resetCode
            };

            await emailjs.send('service_ycn17jz', 'template_5d81ldj', templateParams);
            alert('Doğrulama kodu e-posta adresinize gönderildi!');
            showPage('verificationPage');
        } catch (error) {
            console.error('E-posta gönderme hatası:', error);
            alert('E-posta gönderilirken bir hata oluştu. Lütfen tekrar deneyin.');
        }
    };

    // Doğrulama kodunu kontrol et
    window.verifyCode = function() {
        const enteredCode = document.getElementById('verificationCode').value.trim();
        if (!enteredCode) {
            alert('Lütfen doğrulama kodunu girin!');
            return;
        }

        if (enteredCode === resetCode) {
            showPage('newPasswordPage');
        } else {
            alert('Doğrulama kodu hatalı!');
        }
    };

    // Yeni şifre belirleme
    window.resetPassword = function() {
        const newPassword = document.getElementById('newPassword').value.trim();
        const confirmNewPassword = document.getElementById('confirmNewPassword').value.trim();

        if (!newPassword || !confirmNewPassword) {
            alert('Lütfen tüm alanları doldurun!');
            return;
        }
        if (newPassword.length < 8) {
            alert('Şifreniz en az 8 karakter olmalıdır!');
            return;
        }
        if (!checkPasswordStrength(newPassword).valid) {
            alert('Şifreniz çok zayıf! Lütfen daha güçlü bir şifre girin.');
            return;
        }
        if (newPassword !== confirmNewPassword) {
            alert('Şifreler eşleşmiyor!');
            return;
        }
        // Şifreyi güncelle
        const users = getUsers();
        users[resetUsername].password = newPassword;
        saveUsers(users);
        localStorage.setItem('users', JSON.stringify(users));
        alert('Şifreniz başarıyla değiştirildi!');
        showPage('loginPage');
    };

    // Yeni şifre gücü kontrolü
    const newPassword = document.getElementById('newPassword');
    const newPasswordStrength = document.getElementById('newPasswordStrength');
    if (newPassword && newPasswordStrength) {
        newPassword.addEventListener('input', function() {
            const pwd = newPassword.value;
            if (!pwd) {
                newPasswordStrength.textContent = '';
                newPasswordStrength.style.color = '#e53935';
            } else if (!checkPasswordStrength(pwd).valid) {
                newPasswordStrength.textContent = 'Çok zayıf';
                newPasswordStrength.style.color = '#e53935';
            } else {
                newPasswordStrength.textContent = 'Güçlü';
                newPasswordStrength.style.color = '#43a047';
            }
        });
    }

    // Şifre gözünün görünüp görünmeyeceğini değiştiren fonksiyon
    window.togglePasswordVisibility = function(element) {
        const input = element.previousElementSibling;
        if (!input) return;
        if (input.type === 'password') {
            input.type = 'text';
            element.querySelector('i').classList.remove('fa-eye');
            element.querySelector('i').classList.add('fa-eye-slash');
        } else {
            input.type = 'password';
            element.querySelector('i').classList.remove('fa-eye-slash');
            element.querySelector('i').classList.add('fa-eye');
        }
    };

}); 
