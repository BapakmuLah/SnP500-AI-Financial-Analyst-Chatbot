
membangun aplikasi chatbot untuk menjawab pertanyaan dari user berdasarkan dataset financial S&P tabular yg diberikan. project ini menggunakan teknik Natural Language to SQL (NL2SQL) untuk mengubah user input menjadi Query SQL Secara instant, lalu menampilkan hasilnya. 

Features : 
1. chatbot ini dibangun menggunakan langchain sebagai workflow pipeline nya dan juga menggunaka pipeline agentic dari langchain.
2. chatbot ini menggunakan model GEMINI sebagai model bawaan chatbot ini
3. chatbot ini dilengkapi dgn guardrail pada layer pertama nya untuk mengklasifikasikan user input ke dalam beberapa kategori supaya mencegah prompt berbahaya (prompt injection).
4. Chatbot ini mempunyai beberapa agent yg masing2 agent mempunyai tugas yg berbeda, diantara :
    - Intent Agent (untuk mengklasifikasikan maksud dari user prompt dan mencegah prompt injection)
    - SQL Agent (Agent ini bertugas untuk mengubah user input menjadi SQL Query yg sesuai)
    - Analysis Agent (Agent ini bertugas untuk menganalisis lebih dalam terhadap data ini)
    - Visualization Agent (Bertugas untuk memvisualisasi hasil output data yg didapatkan sesuai keinginan user)
5. user bisa bertanya apapun tentang apapun yg ada di dataset tabular tsb (financial S&P 500) bahkan bisa digunakan untuk menganalisis EDA dan melakukan visualisasi untuk mendapatkan insight / data statistik dari dataset ini.
6. Terdapat fitur evaluasi terhadap chatbot , untuk membandingkan Hasil seberapa bagus hasil prediksi model dengan ground truth
7. chatbot ini di deploy menggunakan backend FastAPI dan di integrasikan dengan frontend lalu dihosting ke dalam Hugging Face Spaces
