from newspaper import Article
from konlpy.tag import Kkma
from konlpy.tag import Twitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
import numpy as np

"""
한국어 문서요약기 입니다.
GUI 기반 요약기입니다.

기사,블로그,각종 사이트의 URL을 입력 시 자동으로 그곳에 있는 내용을 요약해줍니다.
요약할 텍스트를 입력해도 자동으로 그 텍스트 내용을 요약해줍니다.

<기능>
1. 파일로 자동저장 (날짜별)
2. 요약할 문장 수는 조절 가능합니다. (현재는 3문장임)
3. 파파고 번역기능도 추가했습니다. (번역 결과를 같이 저장합니다.) ==> 파파고 X-Naver-Client-Id X-Naver-Client-secreat 를 입력해주세요 !!!!!!!!!!!

<기술소개>
추출요약입니다.
구글의 Text RANK 및 tf-idf 기반 한국어 문서요약기입니다. (구글 참고하시길)
문서입력 ==> 문장 단위 분리 => NLU 전처리 작업(불용어 처리 등) ==> TF-IDF 모델 ==> 그래프 생성 ==> TextRank 적용 ==> 문서요약 
"""

num_summary=3
#파파고 X-Naver-Client-Id X-Naver-Client-secreat 를 입력해주세요 !!!!!!!!!!
PAPAGO_ID=""
PAPAGO_PW=""

# 문장 추출
class SentenceTokenizer(object):
    def __init__(self):
        self.kkma = Kkma()
        self.twitter = Twitter()
        #2019.05.09 불용어 리스트 확장. --pcj--
        self.stopwords =['이', '있', '하', '것', '들', '그', '되', '수', '이', '보', '않', '없', '나', '사람', '주', '아니', '등', '같', '우리', '때', '년', '가', '한', '지', '대하', '오', '말', '일', '그렇', '위하', '때문', '그것', '두', '말하', '알', '그러나', '받', '못하', '일', '그런', '또', '문제', '더', '사회', '많', '그리고', '좋', '크', '따르', '중', '나오', '가지', '씨', '시키', '만들', '지금', '생각하', '그러', '속', '하나', '집', '살', '모르', '적', '월', '데', '자신', '안', '어떤', '내', '내', '경우', '명', '생각', '시간', '그녀', '다시', '이런', '앞', '보이', '번', '나', '다른', '어떻', '여자', '개', '전', '들', '사실', '이렇', '점', '싶', '말', '정도', '좀', '원', '잘', '통하', '소리', '놓']
    
    def url2sentences(self, url):
        article = Article(url, language='ko')
        article.download()
        article.parse()
        sentences = self.kkma.sentences(article.text)

        #print("@@@@@@@@@@@@@@@@@@",sentences) #여기는 URL 이 들어왔을 때 텍스트 추출된 변수이며 리스트 형태이다.
        
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''        
        return sentences
  
    def text2sentences(self, text):
        sentences = self.kkma.sentences(text)      
        for idx in range(0, len(sentences)):
            if len(sentences[idx]) <= 10:
                sentences[idx-1] += (' ' + sentences[idx])
                sentences[idx] = ''
        return sentences

    def get_nouns(self, sentences):
        nouns = []
        for sentence in sentences:
            if sentence is not '':
                nouns.append(' '.join([noun for noun in self.twitter.nouns(str(sentence)) 
                                       if noun not in self.stopwords and len(noun) > 1]))
        return nouns



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
        return np.dot(cnt_vec_mat.T, cnt_vec_mat), {vocab[word] : word for word in vocab}



#TextRank 알고리즘 적용
class Rank(object):
    def get_ranks(self, graph, d=0.85): 
        A = graph
        matrix_size = A.shape[0]
        for id in range(matrix_size):
            A[id, id] = 0 
            link_sum = np.sum(A[:,id])
            if link_sum != 0:
                A[:, id] /= link_sum
            A[:, id] *= -d
            A[id, id] = 1
        B = (1-d) * np.ones((matrix_size, 1))
        ranks = np.linalg.solve(A, B) 
        return {idx: r[0] for idx, r in enumerate(ranks)}


#TextRank Class 구현
class TextRank(object):
    def __init__(self, text):
        self.sent_tokenize = SentenceTokenizer()
        
        if text[:5] in ('http:', 'https'):
            self.sentences = self.sent_tokenize.url2sentences(text)
        else:
            self.sentences = self.sent_tokenize.text2sentences(text)
        
        self.nouns = self.sent_tokenize.get_nouns(self.sentences)
        
        self.graph_matrix = GraphMatrix()
        self.sent_graph = self.graph_matrix.build_sent_graph(self.nouns)
        self.words_graph, self.idx2word = self.graph_matrix.build_words_graph(self.nouns)

        self.rank = Rank()
        self.sent_rank_idx = self.rank.get_ranks(self.sent_graph)
        self.sorted_sent_rank_idx = sorted(self.sent_rank_idx, key=lambda k: self.sent_rank_idx[k], reverse=True)

        self.word_rank_idx = self.rank.get_ranks(self.words_graph)
        self.sorted_word_rank_idx = sorted(self.word_rank_idx, key=lambda k: self.word_rank_idx[k], reverse=True)
    
    def summarize(self, sent_num=3):
        summary = []
        index=[]
        for idx in self.sorted_sent_rank_idx[:sent_num]:
            index.append(idx)
        index.sort()

        for idx in index:
            summary.append(self.sentences[idx])
        return summary

    def keywords(self, word_num=10):
        rank = Rank()
        rank_idx = rank.get_ranks(self.words_graph)
        sorted_rank_idx = sorted(rank_idx, key=lambda k: rank_idx[k], reverse=True)

        keywords = []
        index=[]
        
        for idx in sorted_rank_idx[:word_num]:
            index.append(idx)

        #index.sort()
        for idx in index:
            keywords.append(self.idx2word[idx])

        return keywords



# 결과 확인
# 파이선 GUI tkinter 이용
from tkinter import *
from tkinter.ttk import *

#파일 쓰기기능 추가(2019.05.09)
import datetime
import os

dt = datetime.datetime.now() #시간 객체 생성
directory_name=str(str(dt.year)+str(dt.month)+str(dt.day)) #디렉토리명은 년,월,일의 조합으로 생성한다. (그날 그날 디렉토리 생성)
#file_name=str(dt.hour)+"_"+str(dt.minute)+"_"+str(dt.second)+"_"+"result_summarize.txt"#파일이름은 시간,분,초 + result_summarize의 조합으로 생성한다. (중복안되게)

#디렉토리 생성하기
try:
    if not(os.path.isdir(directory_name)):
        os.makedirs(os.path.join(directory_name))
except OSError as e:
    if e.errno != errno.EEXIST:
        print("Failed to create directory!!!!!")
        raise


#이곳은 TTS 음성을 생성해주는 곳
import requests
from requests import get  # to make GET request

def download(url, file_name):
    # open in binary mode
    with open(file_name, "wb") as file:
        # get request
        response = get(url)
        # write to file
        if response.status_code == 200:
            file.write(response.content)
            print("TTS의 생성이 완료되었습니다")
        else:
            print(file_name) #오류임 !!
            print("TTS가 생성되지 않았습니다")

#tts_name=str(dt.hour)+"_"+str(dt.minute)+"_"+str(dt.second)+"_"+"result_summarize.wav"#파일이름은 시간,분,초 + result_summarize의 조합으로 생성한다. (중복안되게)

#이곳은 TTS 음성을 생성해주는 곳


class MyFrame(Frame):            
    def __init__(self, master):
        Frame.__init__(self, master)
 
        self.master = master
        self.master.title("Text Summarization")
        self.pack(fill=BOTH, expand=True)
 
        # URL 입력
        frame1 = Frame(self)
        frame1.pack(fill=X)
 
        lblURL = Label(frame1, text="URL 또는 문장을 입력", width=20)
        lblURL.pack(side=LEFT, padx=10, pady=10)
 
        self.inputURL = StringVar()
        entryURL = Entry(frame1, textvariable = self.inputURL)
        entryURL.pack(fill=X, padx=10, expand=True)
        # 결과 보기 버튼
        frame2 = Frame(self)
        frame2.pack(fill=X)
        btnResult = Button(frame2, text="Summarize", command=self.onClick)
        btnResult.pack(side=RIGHT, padx=10, pady=10)        
    
        # 요약 결과 출력
        frame3 = Frame(self)
        frame3.pack(fill=BOTH, expand=True)
 
        lblComment = Label(frame3, text="특징", width=10)
        lblComment.pack(side=LEFT, anchor=N, padx=10, pady=10)
 
        global txtComment
        txtComment = Text(frame3)
        txtComment.pack(fill=X, pady=10, padx=10)

    # 버튼 클릭 이벤트 핸들러
    def onClick(self):
        txtComment.delete('1.0', END)
        url = self.inputURL.get()
        textrank = TextRank(url)
        #print("@@@@@",url) #여기가 원본 임 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        for row in textrank.summarize(num_summary):  # 몇 줄로 요약할꺼야
            txtComment.insert(END, row)
            txtComment.insert(END, '\n')
            txtComment.insert(END, '\n')
        txtComment.insert(END, '\n')
        txtComment.insert(END, 'keywords :')
        txtComment.insert(END, textrank.keywords())

        #파일에 저장하기 기능
        file_name=str(dt.hour)+"_"+str(dt.minute)+"_"+str(dt.second)+"_"+"result_summarize.txt"#파일이름은 시간,분,초 + result_summarize의 조합으로 생성한다. (중복안되게)
        f_w=open("./"+directory_name+"/"+file_name,'a',encoding='utf-8')
        
        f_w.write("요약일: "+str(dt.year)+"년"+str(dt.month)+"월"+str(dt.day)+"일"+str(dt.hour)+"시"+str(dt.minute)+"분"+"\n\n")
        f_w.write("==========================================================================================================="+"\n")
        f_w.write("입력: "+"\n"+url.strip()+"\n\n")
        f_w.write("==========================================================================================================="+"\n")
        f_w.write("키워드: ")
        for key in textrank.keywords():
            #print(key)
            f_w.write(key+"\t")

        f_w.write("\n\n")
        f_w.write("==========================================================================================================="+"\n")
        f_w.write("요약 결과: "+"\n")   
        for row in textrank.summarize(num_summary):
            #print(row)
            f_w.write(row+"\n")
        f_w.write("\n")
        f_w.write("==========================================================================================================="+"\n")
        f_w.close()



        #여기는 파파고 번역 파트
        import requests
        import json

        file_trans_name=str(dt.hour)+"_"+str(dt.minute)+"_"+str(dt.second)+"_"+"result_translation_summarize.txt"#파일이름은 시간,분,초 + result_summarize의 조합으로 생성한다. (중복안되게)
        f_w_t=open("./"+directory_name+"/"+file_trans_name,'a',encoding='utf-8')
        f_w_t.write("Summarization Date: "+str(dt.year)+"year"+str(dt.month)+"month"+str(dt.day)+"day"+str(dt.hour)+"hour"+str(dt.minute)+"minute"+"\n")
        f_w_t.write("==========================================================================================================="+"\n")
        f_w_t.write("Input: "+"\n"+url.strip()+"\n\n")
        f_w_t.write("==========================================================================================================="+"\n")
        f_w_t.write("Keywords: ") 


        #파파고 API 준비
        url="https://openapi.naver.com/v1/papago/n2mt?source=en&target=ko&text="
        request_url = "https://openapi.naver.com/v1/papago/n2mt"
        headers= {"X-Naver-Client-Id": PAPAGO_ID, "X-Naver-Client-Secret":PAPAGO_PW}  ##파파고 X-Naver-Client-Id X-Naver-Client-secreat 를 입력해주세요 !!!!!!!!!!!


        #키워드 번역 단계
        for text in textrank.keywords():
            params = {"source": "ko", "target": "en", "text": text} #여기서 언어 설정 가능  번역
            response = requests.post(request_url, headers=headers, data=params)
            result = response.json()
            parse1=result['message']
            parse2=parse1['result']
            result=parse2['translatedText']
            #print("Source: ",text)
            #print("Target: ",result)
            f_w_t.write(result+"\t")
            
        f_w_t.write("\n\n")
        f_w_t.write("==========================================================================================================="+"\n")
        f_w_t.write("Summarization Result: "+"\n") 

        #문장 번역 단계
        for text in textrank.summarize(3):
            params = {"source": "ko", "target": "en", "text": text} #여기서 언어 설정 가능  번역
            response = requests.post(request_url, headers=headers, data=params)
            result = response.json()
            parse1=result['message']
            parse2=parse1['result']
            result=parse2['translatedText']
            #print("Source: ",text)
            #print("Target: ",result)
            f_w_t.write(result+"\n")
        f_w_t.write("\n")
        f_w_t.write("==========================================================================================================="+"\n")
        f_w_t.close()
        print("PAPAGO 번역 완료")
        print("Finish")
        
        #대상 언어 코드. 
        #1.ko : 한국어
        #2.en : 영어
        #3.zh-CN : 중국어 간체
        #4.zh-TW : 중국어 번체
        #5.es : 스페인어
        #6.fr : 프랑스어
        #7.vi : 베트남어
        #8.th : 태국어
        #9.id : 인도네시아어
        #ko<->en, ko<->zh-CN, ko<->zh-TW, ko<->es, ko<->fr, ko<->vi, ko<->th, ko<->id, en<->ja, en<->fr 조합만 가능

        #번역 결과 언어 코드. 
        #1.ko : 한국어
        #2.en : 영어
        #3.zh-CN : 중국어 간체
        #4.zh-TW : 중국어 번체
        #5.es : 스페인어
        #6.fr : 프랑스어
        #7.vi : 베트남어
        #8.th : 태국어
        #9.id : 인도네시아어

from tkinter import ttk #디자인 작업

def main():
    root = Tk()
    root.geometry("1200x550+100+100")

    label = ttk.Label(root)
    label.pack()

    #디자인 작업 2019.05.10
    label.img= PhotoImage(file = 'logo.PNG')
    label.config(image = label.img,compound = 'bottom')


    app = MyFrame(root)
    root.mainloop()

if __name__ == '__main__':
    main()

