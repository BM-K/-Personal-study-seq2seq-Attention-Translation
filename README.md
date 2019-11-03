# seq2seq_Attention_Translation
##https://9bow.github.io/PyTorch-tutorials-kr-0.3.1/intermediate/seq2seq_translation_tutorial.html
위 자료를 바탕으로 공부하였다.

# 동작
## <img src = https://user-images.githubusercontent.com/55969260/68080785-a1aa4580-fe45-11e9-91ba-1abf82e315ac.png>
hidden_size = 256, encoder1 을 초기화 할시 파라미터 input_lang.n_words, hidden_size 대입. <br>
input_lang : 데이터 파일로 부터 데이터를 쌍으로 리스트에 저장하고 MAX_LENGTH 안쪽의 길이에 해당하는 pair 만 pair에 저장 <br>
<img src = https://user-images.githubusercontent.com/55969260/68080826-62c8bf80-fe46-11e9-8d1a-89fbf2f94bd6.png> <br>
받은 pairs 를 통해 input_lang.addSentence, output_lang.addSentence 수행. 이는 문장을 단어별로 split하고 word 가 word2index 사전에 없으면<br> word2index = { new_word : 2 }, word2count = { new_word : 1 }, index2word = { 2 : new_word }, n_words++ 이를 통해 사전 형성.
이때 index2word = { 0:"SOS", 1:"EOS" } 이다. 
