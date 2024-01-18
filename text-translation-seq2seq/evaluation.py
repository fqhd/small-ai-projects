import torch
from utils import tensorFromSentence
from globals import EOS_token, hidden_size
from lang import prepareData
from network import EncoderRNN, AttnDecoderRNN
import random

def evaluate(encoder, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence, 'cpu')

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden, 'cpu')

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn


if __name__ == '__main__':
	# Write this out later
	# We want to be able to enter text and see the translation in realtime
	input_lang, output_lang, pairs = prepareData()
     

	encoder = EncoderRNN(input_lang.n_words, hidden_size)
	encoder.load_state_dict(torch.load('encoder.pth'))
	encoder.eval()

	decoder = AttnDecoderRNN(hidden_size, output_lang.n_words)
	decoder.load_state_dict(torch.load('decoder.pth'))
	decoder.eval() 

	for _ in range(10):
		pair = random.choice(pairs)
		print('Input Sentence: ' + pair[0])
		output_words, _ = evaluate(encoder, decoder, pair[0], input_lang, output_lang)
		output_sentence = ' '.join(output_words)
		print('AI Translation: ' + output_sentence)
		print('Target: ' + pair[1])
		print('--------------------------------------------------------------')
			