# seq2seq_Attention_Translation
https://9bow.github.io/PyTorch-tutorials-kr-0.3.1/intermediate/seq2seq_translation_tutorial.html <br>
위 자료를 바탕으로 공부하였다.

# 동작
## <img src = https://user-images.githubusercontent.com/55969260/68080785-a1aa4580-fe45-11e9-91ba-1abf82e315ac.png>
hidden_size = 256, encoder1 을 초기화 할시 파라미터 input_lang.n_words, hidden_size 대입. <br>
input_lang : 데이터 파일로 부터 데이터를 쌍으로 리스트에 저장하고 MAX_LENGTH 안쪽의 길이에 해당하는 pair 만 pair에 저장 <br>
<img src = https://user-images.githubusercontent.com/55969260/68080826-62c8bf80-fe46-11e9-8d1a-89fbf2f94bd6.png> <br>
받은 pairs 를 통해 input_lang.addSentence, output_lang.addSentence 수행. 이는 문장을 단어별로 split하고 word 가 word2index 사전에 없으면<br> word2index = { new_word : 2 }, word2count = { new_word : 1 }, index2word = { 2 : new_word }, n_words++ 이를 통해 사전 형성.
이때 index2word = { 0:"SOS", 1:"EOS" } 이다.  input_lang, output_lang, pairs 반환 
EncoderRNN 에 input_lang 의 word 개수와 hidden_size를 파라미터로 대입. <br>
<img src = https://user-images.githubusercontent.com/55969260/68080885-e040ff80-fe47-11e9-8ae0-25e634fb3d79.png> <br>
EncoderRNN에  input 단어의 개수와 hidden_size를 통해 nn.Embedding, gru를 초기화. forward에서 input_lang word 개수에 해당하는 벡터들이 임베딩 되고 hidden 과 같이 gru 에 들어간다. <br>
<img src = https://user-images.githubusercontent.com/55969260/68080987-bc7eb900-fe49-11e9-9681-ab983bbe09d0.png> <br>
AttnDecoderRNN에 hidden_size 와 output_lang.n_words를 파라미터로 넣어준다. <br>
<img src = https://user-images.githubusercontent.com/55969260/68081005-11baca80-fe4a-11e9-8e72-8eafe6dd265d.png> <br>
