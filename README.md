# Korea_Text_Summarization_GUI
Korea Text_Summarization GUI Software (Extractive , Textrank)

## Introduction
한국어 문서요약기 입니다.
GUI 기반 요약기입니다.

기사,블로그,각종 사이트의 URL을 입력 시 자동으로 그곳에 있는 내용을 요약해줍니다.
요약할 텍스트를 입력해도 자동으로 그 텍스트 내용을 요약해줍니다.

윈도우 환경에서 실행하시는 것을 추천드립니다.

## 기능
1. 파일로 자동저장 (날짜별)
2. 요약할 문장 수는 조절 가능합니다.
3. 파파고 번역기능도 추가했습니다. (번역 결과를 같이 저장합니다.)

## 기술 소개
추출요약입니다. (Extractive 방식) <br>
구글의 Text RANK 및 tf-idf 기반 한국어 문서요약기입니다.  <br>
문서입력 ==> 문장 단위 분리 => NLU 전처리 작업(불용어 처리 등) ==> TF-IDF 모델 ==> 그래프 생성 ==> TextRank 적용 ==> 문서요약 <br>
딥러닝 기반 문서요약기와 비교하여 성능은 떨어질지 모르나 학습데이터가 필요하지 않다는 장점이 있습니다.

## Install
- Tkinter를 설치해주세요.  
ubuntu: sudo apt-get install python-tk
- newspaper3k
- sklearn
- konlpy
- numpy
- requests (번역 기능 사용시)

## Precautions
실행시 주의사항: 번역기능은 PAPAGO API를 사용하므로 API키를 발급받으셔야 합니다.

- 번역기능 이용 X: TextRank_Korea_summarization_GUI.py
- 번역기능 이용: TextRank_Korea_summarization_GUI_Papago.py

num_summary 변수를 통하여 요약문장 개수 조절가능합니다.
