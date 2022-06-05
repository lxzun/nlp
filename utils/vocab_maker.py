import sentencepiece as spm

corpus = "openwebtext.txt"
prefix = "../vocab/vocab_20"
vocab_size = 20000
spm.SentencePieceTrainer.Train(f"--input={corpus} --model_prefix={prefix} --vocab_size={vocab_size + 6}" +
                                " --model_type=bpe" +
                                " --max_sentence_length=100200" + # 문장 최대 길이
                                " --pad_id=0 --pad_piece=[PAD]" + # pad (0)
                                " --unk_id=1 --unk_piece=[UNK]" + # unknown (1)
                                " --bos_id=2 --bos_piece=[BOS]" + # begin of sequence (2)
                                " --eos_id=3 --eos_piece=[EOS]" + # end of sequence (3)
                                " --user_defined_symbols=[SEP],[MASK]" + # 사용자 정의 토큰
                                " --model_type=char"
                                " --input_sentence_size=2300000"
                                " --shuffle_input_sentence=true")