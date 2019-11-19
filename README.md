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
<img src = https://user-images.githubusercontent.com/55969260/68081021-6100fb00-fe4a-11e9-8a73-c4a7d46016c2.png> <br>
<br>
깊이 들어가기 전에 trainiters를 보자. 파라미터로 encoder1, decoder1, n_iters, print_every가 들어간다. optimizer 함수로는 SGD를 택하였고 
training_pairs에 fra, eng 쌍이 텐서로 들어가게된다. 그리고 loss를 구할 때 train에 input_variable, target_variable, encoder1, decoder1, encoder_optimizer, decoder_optimizer, critertion이 들어간다. <br>
<img src = https://user-images.githubusercontent.com/55969260/68101394-ccad9b80-ff10-11e9-8b68-31d0f4c5d833.png> <br>
train 함수로 들어가보자. <br>
<img src = https://user-images.githubusercontent.com/55969260/68101445-ff579400-ff10-11e9-93a3-2fabe5959427.png> <br>
encoder_outputs를 max_length, encoder.hidden_size로 초기화하고 for문을 돌면서 encoder1에 input_variable[ei]와 encoder_hidden을 넣어준다. input_variable[ei]는 단어 하나의 텐서 ex( [[145]]] ) 값이 들어간다. 그리고 gru를 거친뒤 아웃풋을 encoder_outputs[ei]에 넣어준다. 그리고 Teacher forcing을 할지 안 할지 결정해준다. use_teacher_forcing 부분을 보자 <br>
<img src = https://user-images.githubusercontent.com/55969260/68101605-ccfa6680-ff11-11e9-82a0-552144e0d6be.png> <br>
decoder에 decoder_input, decoder_hidden, encoder_outputs 를 넣어준다. decoder forward에서 다음 그림과 같이 수행되며 output과 target_variable 과의 loss를 구하고 다음 decoder 입력 값으로 target_variable 을 넣어준다.
<br><img src=https://user-images.githubusercontent.com/55969260/69004095-edcead00-0950-11ea-8cfd-e1c396b20dd4.png><br><br>
"Teacher forcing 없이 : 자신의 예측을 다음 입력으로 사용" ↓
<br><img src=https://user-images.githubusercontent.com/55969260/69109583-8d0fb380-0abb-11ea-8ece-3e0708b639fa.png><br>
가장 큰 값의 주소를 알기 위해서 Tesorltopk를 사용한다. ni는 디코더가 예측한 가장 큰 값을 갖고 있다. ex) tensor[245]. 다음 디코더 입력으로 Variable(torch.LongTenso([[ni]])) 를 넣어준다. loss 값을 갱신해주고 디코더가 EOS_token 을 예측하면 반복문을 중단한다. 오류 역전파를 수행하고 loss 평균을 리턴한다. 다시 train 으로 온다. 
<br><img src=https://user-images.githubusercontent.com/55969260/69116647-6ad46080-0ad0-11ea-8dbd-c38af8164459.png><br>
print 할 loss 값과 plot 할 loss 값을 갱신해준다. 다음으로 evaluate 부분을 보자 
<br><img src=https://user-images.githubusercontent.com/55969260/69116924-55ac0180-0ad1-11ea-9043-f7e7784ad280.png><br>
<br><img src=https://user-images.githubusercontent.com/55969260/69117018-bb988900-0ad1-11ea-99f7-d273683547ef.png><br>
encoder1, decoder1, sentence, max_length parameter. input variable로 sentence를 Tesor화 해서 넣는다. encoder output을 얻기위해 input 길이만큼 for 문을 반복한다. 나중에 도식화를 위해서 디코더의 어텐션 출력을 저장할 decoded_words 배열을 선언한다. 
<br><img src=https://user-images.githubusercontent.com/55969260/69117787-36fb3a00-0ad4-11ea-8d20-c1bdcf4469c4.png><br>
decoder가 예측한 값을 얻기 위한 for loop이다. 
<br> 모델 저장 & 불러오기 공부후 적용해보기 and test acc 적용해보기  <작성중>
