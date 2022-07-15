
## Introduction
한국어 docx/hwp 파일 요약기 및 하이라이터 입니다.<br>
https://github.com/Parkchanjun<br>
에 커밋된 코드에 기초하여 특허 명세서에 맞춰 변형하였습니다.<br>
명세서용 불용어등 추가<br>
아래하 한글, Words 파일을 변환<br>

doc_path = r"C:\Users\요약기\발명명세서.docx"<br> 
doc_summury2(doc_path)<br>
=> output = r"C:\Users\요약기\발명명세서bold.docx"<br>

DOC_SUM.bat(batch 파일)을 통해 dragNdrop으로 구현<br>

# Korea_Text_Summarization_GUI
'참조: https://github.com/Parkchanjun'<br>
Korea Text_Summarization GUI Software (Extractive , Textrank)<br>


## 기능
1. textrank 기반으로 중요문장을 찾고 하이라이터 수행 
2. 명사구. 합성 명사등을 고려하여 상위 4개 키워드 같이 하이라이트



## Install
- sklearn
- konlpy
- numpy


