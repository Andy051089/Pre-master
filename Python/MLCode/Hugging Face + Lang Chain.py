# from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
#%% Hugging face + LangChain
'''
1.
Lang Chain : 可以簡單使用預訓練模型，做自己特定的任務
Hugging Face : 有很多預訓練模型可以讓我們用
模型常見可以做文本分類、問答任務、命名識別
    EX:我一直頭痛，有一點發燒，我自己有吃一顆SCANOL，為什麼會這樣?
    文本分類 : 1.內科 2.流感 3.留觀....
    問答任務 : 可能身體出現免疫反應，所以才會發燒...
    命名識別 : 頭痛、發燒:症狀。藥 : SCANOL。...
2.
chunk_size : 每一文字塊字數(提高效率、模型輸入限制)
chunk_overlap : 每小塊重複性(保持每塊的連續性)


'''
# 讀檔案
loader = TextLoader('C:\研究所\自學\各模型\DATA\衛教.txt', encoding = 'utf-8')
documents = loader.load()
# 把檔案轉成文字
text = ' '.join([doc.page_content for doc in documents])
# 把全文切割成一小塊一小塊
# 如何執行文字切割
text_splitter = CharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
# 照上面設定進行切割
texts = text_splitter.split_text(text)
# 把文字切割Tokenized及轉換成向量功能
embeddings = HuggingFaceEmbeddings()
# 把每段文字執行Embedding，並從FAISS搜尋每塊向量的相似向量
db = FAISS.from_texts(texts, embeddings)
# 儲存搜尋的相似值，K(找3個)
retriever = db.as_retriever(search_kwargs = {'k' : 3})
# 加載模型
model_name = 'google/flan-t5-base'
# 將輸入問題轉換成模型看得懂得格式、輸出結果轉換成文字
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
# 告訴模型如何回答、給予回答格式
prompt_template = """
Answer according to the context, but don't copy paste. Using your own words. If there's no according context, Answer 'I don't have enough information about your question.'
text ： {context}
Question ： {question}
Ans ： 
"""
PROMPT = PromptTemplate(
    template = prompt_template, 
    input_variables = ['context', 'question'])
'''
text2text-generation : 模型做甚麼任務
model = model : 使用之前加載的model
tokenizer = tokenizer : 使用之前加載的tokenizer
max_length = 200 : 生成最大長度
min_length = 50 : 生成最小長度
no_repeat_ngram_size = 3 : 避免重複生成連續3個詞
num_beams = 5 : 每一步都考慮最優的5個，並從中選一個
temperature = 0.7 : 回答隨機性及創造性

'''
pipe = pipeline(
    'text2text-generation',
    model = model, 
    tokenizer = tokenizer, 
    max_length = 200,
    min_length = 50,
    no_repeat_ngram_size = 3,
    num_beams = 5,
    temperature = 0.5)
# 建立根據我們設定模型
llm = HuggingFacePipeline(pipeline = pipe)
# 建立問答系統
'''
llm = llm : 使用上面見的llm
chain_type = "stuff" : 將問題相關的內容一次全部塞給模型
retriever = retriever : 使用前面設定的retriever搜尋結果
return_source_documents = True : 同时給答案和用于生成答案的檔案
chain_type_kwargs = {'prompt' : PROMPT}) : 使用剛剛的PROMPT
'''
qa_chain = RetrievalQA.from_chain_type(
    llm = llm,
    chain_type = 'stuff',
    retriever = retriever,
    return_source_documents = True,
    chain_type_kwargs = {'prompt' : PROMPT})
query = 'What is democracy?'
result = qa_chain({'query' : query})
print(result['result'])