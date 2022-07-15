from konlpy.tag import Kkma
from konlpy.tag import Okt, Hannanum
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import os
import win32com.client as win32
import win32gui



"""
한국어 문서요약기 입니다.
<기능>
<기술소개>
추출요약입니다.
구글의 Text RANK 및 tf-idf 기반 한국어 문서요약기입니다. (구글 참고하시길)
문서입력 ==> 문장 단위 분리 => NLU 전처리 작업(불용어 처리 등) ==> TF-IDF 모델 ==> 그래프 생성 ==> TextRank 적용 ==> 문서요약 
"""



# 문장 추출
class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.twitter = Okt()
        self.hannanum = Hannanum()
        # 2019.05.09 불용어 리스트 확장. --pcj--
        self.stopwords = ['이', '있', '하', '것', '들', '그', '되', '수', '이', '보', '않', '없', '나', '사람', '주', '아니', '등', '같',
                          '우리', '때', '년', '가', '한', '지', '대하', '오', '말', '일', '그렇', '위하', '때문', '그것', '두', '말하', '알',
                          '그러나', '받', '못하', '일', '그런', '또', '문제', '더', '사회', '많', '그리고', '좋', '크', '따르', '중', '나오',
                          '가지', '씨', '시키', '만들', '지금', '생각하', '그러', '속', '하나', '집', '살', '모르', '적', '월', '데', '자신', '안',
                          '어떤', '내', '내', '경우', '명', '생각', '시간', '그녀', '다시', '이런', '앞', '보이', '번', '나', '다른', '어떻',
                          '여자', '개', '전', '들', '사실', '이렇', '점', '싶', '말', '정도', '좀', '원', '잘', '통하', '소리', '놓','상기']



    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx - 1] += (' ' + sentences[idx])
                sentences[idx] = ''
        return sentences

    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence is not '':
                nouns.append(' '.join([noun for noun in self.hannanum.nouns(str(sentence))
                                       if noun not in self.stopwords and len(noun) > 1]))
        return nouns

    def get_key_word(self, sentences):
        keywords = []
        for sentence in sentences:
            if sentence is not '':
                words = self.hannanum.pos(sentence)

                # Define a chunk grammar, or chunking rules, then chunk
                grammar = """
                    NP: {<N.*>*<Suffix>?}   # Noun phrase
                    """
                parser = nltk.RegexpParser(grammar)
                chunks = parser.parse(words)

                for np in chunks.subtrees():
                    if (np.label() == 'NP') & (len(np.leaves()) > 1):
                        keywords.append(" ".join([lf[0] for lf in np.leaves()]) )


        word_dict = {i: keywords.count(i) for i in keywords}
        word_dict = sorted(word_dict.items(), key=lambda item: item[1], reverse=True)
        word_dict = [word for word in word_dict if (word in word_dict[:9]) or (word[0].count(" ") > 3)]

        return word_dict

# TF-IDF 모델 생성 및 그래프 생성
class GraphMatrix(object):
    def __init__(self):
        self.tfidf = TfidfVectorizer()
        self.cnt_vec = CountVectorizer()
        self.graph_sentence = []

    def build_sent_graph(self, sentence):
        tfidf_mat = self.tfidf.fit_transform(sentence).toarray()
        self.graph_sentence = np.dot(tfidf_mat, tfidf_mat.T)
        return self.graph_sentence

    def build_words_graph(self, sentence):
        cnt_vec_mat = normalize(self.cnt_vec.fit_transform(sentence).toarray().astype(float), axis=0)
        vocab = self.cnt_vec.vocabulary_
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word]: word for word in vocab}


# TextRank 알고리즘 적용
class Rank(object):
    def get_ranks(self, graph, d=0.85):
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0
            link_sum = np.sum(A[:, id])
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
        B = (1 - d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B)
        return {idx: r[0] for idx, r in enumerate(ranks)}


# TextRank Class 구현
class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()

        self.sentences = text

        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
        self.keyword = self.sent_tokenize.get_key_word(self.sentences)

        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)

        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)

        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)

    def summarize(self, ratio=0.1):
        summary = []
        index = []

        sent_num = int(len(self.sorted_sent_rank_idx)*ratio)

        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        index.sort()
        # print(index)
        for idx in index:
            summary.append(self.sentences[idx])
        return summary

    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)

        keywords = []
        index = []

        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)

        # index.sort()
        for idx in index:
            keywords.append(self.idx2word[idx])

        return keywords




def getText(doc_origin):
    doc = doc_origin
    fullText = []
    for para in doc.paragraphs:
        fullText.append(para.text)
    return '\n'.join(fullText)


def hwp2docx(file):
    dest = os.path.dirname(file)

    hwp = win32.gencache.EnsureDispatch('HWPFrame.HwpObject')
    hwnd = win32gui.FindWindow(None, 'Noname 1 - HWP')
    print(hwnd)
    print(file, dest)

    hwp.Open(os.path.join(dest, file))
    pre, ext = os.path.splitext(file)

    output_file = os.path.join(dest, pre + ".docx")
    hwp.SaveAs(output_file,'OOXML')

    hwp.Quit()

    return output_file




def replace_breaks(text):

    line_breaks = {'\r', '\r\n', '\n'}
    replace_text = text

    for L_break in line_breaks:
        if L_break in replace_text:
            replace_text = replace_text.replace(L_break, '')
    return replace_text

def doc_summury2(input_file):
    folder_path = os.path.dirname(input_file)
    fileName, fileExtension = os.path.splitext(input_file)
    print(fileExtension)
    if (fileExtension =='.hwp'):
        input_file = hwp2docx(input_file)
    print(input_file)

    myWord = win32.Dispatch('Word.Application')
    myWord.Visible = 0
    myWord.Documents.Open(input_file)
    sentences = myWord.Activedocument.sentences




    my_text = []

    for para in sentences:
        filtered_characters = list(s for s in str(para) if s.isprintable())
        filtered_string = ''.join(filtered_characters)

        # print(filtered_string)
        my_text.append(filtered_string)

    input_path_split = os.path.splitext(input_file)
    # news_text = '\n'.join(my_text)
    news_text = my_text

    textrank = TextRank(news_text)

    T_Highlight = textrank.summarize(0.1)

    keywords = textrank.keyword

    print(keywords)

    keywords = keywords[:4]

    for sentence in sentences:
        if replace_breaks(sentence.text) in T_Highlight:
            sentence.HighlightColorIndex  = 7
    colorindex = [4,3,16,10]
    for key,KW in enumerate(keywords):
        myWord.Options.DefaultHighlightColorIndex = colorindex[key]
        myWord.Selection.Find.Replacement.Highlight = True
        myWord.Selection.Find.Execute(KW[0], False, False, True, False, False,True, 0, True, '^&', 2)


    ext_file = os.path.join(folder_path, input_path_split[0])
    myWord.ActiveDocument.SaveAs2(ext_file + "bold" + '.docx', FileFormat=16)
    myWord.ActiveDocument.Close()




if __name__ == '__main__':
    doc_path = r"C:\Users\요약기\발명명세서.docx"
    doc_summury2(doc_path)
