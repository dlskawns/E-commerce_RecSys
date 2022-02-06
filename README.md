# 리뷰기반 고객 집단 분류 및 딥러닝 기반 추천 알고리즘 
  
이커머스 데이터의 리뷰 분석을 통해 상품평 관련 키워드를 추출합니다.  
고객의 리뷰에 따라 작은 집단 분류를 진행하고 해당 정보를 바탕으로 추천시스템 모델링을 목표합니다.

<br>

## 프로젝트 주제 선정 이유:

* 디지털 산업 수요증가 - 이커머스, OTT플랫폼, SNS마케팅 
* 디지털 산업 경쟁 심화 - 지그재그인수(카카오), 배민라이브쇼핑, 디즈니플로스, 애플TV+ 등 
* 고객 유지, 이탈률 최소화를 위한 차별적 시스템, 서비스 도입 필요
* 개인적인 디지털 산업에 대한 커리어 목표 및 관심

<br>

## 데이터 셋 Features 설명

출처: Kaggle  

데이터셋 명/파일명/용량: Amazon Fine Food Reviews / Review.csv / 300.9 mb  

#### Columns

Id(int): 샘플별 고유값  
ProductId(str): 상품Id로 해싱처리 되어있음  
UserId(str): 고객Id로 이역시 해싱처리 되어있음  
ProfileName(str): 고객이 설정한 계정 이름  
HelpfulnessNumerator(int): 해당 리뷰가 도움이 된다고 추천한 고객 수  
HelpfulnessDenominator(int): 추천을 동의하거나 하지 않은   
Score(int): 리뷰를 쓴 고객이 해당 상품에 매긴 평점  
Time(timestamp): 리뷰가 작성된 날짜 데이터. 타임스탬프 형식  
Summary(str): 고객 리뷰 데이터의 제목  
Text(str): 고객 리뷰 데이터 전문  

<br>

## 프로젝트 데이터 셋 & 선정 이유:

* 고객 행동 기반 데이터를 통해 자사(플랫폼 및 쇼핑몰 운영사) 고객의 이탈 원인 파악이 필요  
* 날짜 데이터(timestamp)를 통해 언제 구매가 줄었는지 파악
* 리뷰(Text) context를 통해 문장 내 주요 키워드 추출 가능
* 평점(Score)를 통해 추천 모델 학습을 위한 label처리 가능

<br>

## 분석 및 모델링 진행 방법

### 1. EDA & Feature Engineering  
각 Features의 특징을 면밀히 체크하고, Feature 간의 관계를 파악  

* 퍼널(Funnel) 분석을 위한 최종적으로 재구매 주기 및 재구매가 줄어든 시점을 파악해봅니다.  
* 평점 또는 작성 날짜 데이터와 같이 고객 구매행동 feature와 다른 feature 간의 관계를 파악해봅니다.  

<br>

<br>

### 2. Text 분석

* 리뷰(Text) 분석을 통해 키워드 및 고객 집단(cohort)분류를 위한 인사이트를 얻습니다.
* nltk 모듈의 함수가 다양하므로 nltk로 토큰화. (ex: most_common, FreqDist 등)
* TOP50 내의 단어 중 무의미한 단어들을 갖고 있다면 불용어로 처리. 최종 명사, 형용사로 전체 데이터 주요 키워드 파악

<br>

<br>

### 3. 키워드 추출 모델
Sentence Transformer - Bert(distilbert-base-nli-mean-tokens)를 이용  

#### * 선정 이유:
NLI(Natural Language Inference) 작업에 적합한 모델로, 문장(doc)단위 임베딩이 가능하다.  

#### * 가설:
문장단위 임베딩을 진행 후, 단어 임베딩의 유사도를 파악해 해당 문장 내 중요 키워드를 찾아낼 수 있다.  

#### * 진행 방법:  
* 빈도 수 기반 벡터화를 진행 (Counter Vectorizer) - n gram(3,3)으로 단어 묶음 설정  
* sentence transformer로 샘플 별 문장 임베딩 생성  
* sentence transformer로 샘플 별 단어 임베딩 생성  
* Cosine 유사도를 통해 특정 중요 단어 파악  
* 동일한 방법으로 n gram(1,1)로 진행하여 최종 키워드 top 5~8개 추출    

<br>

<br>

### 4. 문맥을 통한 고객 집단 분류 모델
Transformer - Bert(bert-base-uncased)를 이용  
반려동물용 상품을 판매하는 것이 파악되어 반려동물 유무(0 - 없음, 1 - 강아지, 2 - 고양이, 3 - 강아지와 고양이)로 고객 세분화 진행  


#### * 선정 이유:
label(반려동물 유무 여부)을 설정한 뒤, 전체 문맥에 따른 특징적인 패턴을 찾아 전체 데이터셋에 자동으로 label을 지정하기 위함.
ㄴ> 이를 통해 궁극적으로 고객의 집단은 분류하는 feature를 생성.

#### * 가설:
BERT를 이용해 문맥데이터와 label을 학습하면 문장들의 패턴을 파악해 반려동물 유무에 대한 고객 세분화가 가능할 것이다.

#### * 방법:
* 리뷰 내에서 dog, cat을 파악할 수 있는 키워드를 통해 일차적으로 label 생성
* label 분포 파악 후 imbalanced 할 경우, under sampling 및 SMOTE 진행 후 성능 파악
* 샘플링 방법 선정 후(최종-under sampling) BERT 다중분류 모델 학습 진행
* 학습 완료된 모델을 통해 자동 라벨링 진행

<br>

<br>

### 5. 특성을 활용한 딥러닝 기반 추천 시스템 모델
DCN - Deep Cross NET 이용
CTR(클릭률)에 활용되는 모델로 다양한 특성들의 조합을 고려할 수 있는 모델


#### * 선정 이유:
유저특성, 상품특성을 인풋데이터로 넣어 Embedding, Cross, DNN Layer로 이어지는 Stacked 구조의 모델을 거쳐
각 feature의 조합에 따라 라벨을 예측하는 지도학습을 진행하기 위함.

#### * 가설:
Deep Cross Net의 구조를 이용해 모델을 설계하면, 인풋 데이터의 모든 Feature들에 대한 특성을 고려해 학습 및 예측이 가능할 것이다.

#### * 방법:
* 인풋 데이터
  * Score를 바탕으로 학습 및 예측(추천)하기 위한 label 생성
    * 전체 평균 Score(4.18)에 따라 labelling -> 4점 이하 = 0, 5점 = 1
  * UserId, ItemId뿐 아니라 유저특성, 상품특성을 Embedding Layer의 인풋데이터로 넣는다.
  * 추천 시스템을 고려해서 학습 특성 선택
* 모델 아키텍쳐
  * 인풋 데이터의 각 특성 별로 Embedding Layer를 작성해넣고, 이를 concat하여 합쳐준다.
  * Cross Layer를 작성한 뒤, Embedding Layer의 아웃풋을 인풋으로 통과한다.
  * Deep Layer를 작성한 뒤, Cross Layer의 아웃풋을 인풋으로 넣어 최종 분류기(binary-sigmoid)를 통과시킨다.
  * metric = BinaryAccuracy
* 추천 모듈(recommendation)
  * UserId 입력시 users(유저 룩업테이블)에서 DCN 모델 predict 진행을 위한 feature 정보를 가져온다. (UserId, 리뷰 수, 반려동물 유무)
  * items(상품 룩업테이블)의 상품에 대한 label을 예측한다.
  * 예측 완료 후 높은 순위의 확률을 가진 10개 상품을 노출시킨다.









