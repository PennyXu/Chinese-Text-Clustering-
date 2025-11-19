from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pandas as pd
import re
import string
import jieba
import ast
import xp_tool

def preprocess_chinese_text(text) -> list:
    if pd.isna(text):
        return []
    text = re.sub(r"[^\u4e00-\u9fa5\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    words = jieba.lcut(text, cut_all=False)
    stop_words = { "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都", "一",
    "一个", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着", "没有",
    "看", "好", "自己", "这", "那", "对于", "进行", "表示", "提出", "出来", "还是",
    "时候", "什么", "怎样", "哪里", "多少", "一下", "一下下", "有些", "有的", "应该",
    "可能", "可以", "让", "把", "给", "跟", "对", "向", "与", "同", "被", "由", "从",
    "比", "为", "之", "以", "而", "则", "却", "虽", "然", "但", "是", "或", "且", "并",
    "也", "又", "还", "再", "更", "最", "太", "极", "很", "挺", "非常", "十分", "格外",
    "稍微", "略", "几", "多", "少", "只", "仅", "才", "就", "都", "全", "总", "共", "统",
    "凡", "每", "各", "别", "另", "某", "本", "该", "此", "彼", "哪", "何", "孰", "焉",
    "矣", "也", "哉", "乎", "者", "也", "之", "兮", "耳", "矣", "耶", "欤", "嗯", "哦", "啊", "呀", "吧", "吗", "呢", "啦", "呗", "咯", "哼", "哈", "呵", "嗨", "啧",
    "用户","反映","导致","司机","平台","客服"
    }
    return [word for word in words if word not in stop_words and len(word) > 1]

def kmeans(df,num) -> tuple:
  df["question_clean"] = df["result"].apply(preprocess_chinese_text)
  df["清洗后文本"] = df["question_clean"].apply(lambda x: " ".join(x))
  tfidf = TfidfVectorizer(ngram_range=(1, 2)) 
  X = tfidf.fit_transform(df["清洗后文本"])
  n_clusters = num
  kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
  df["label"] = kmeans.fit_predict(X)
  label_count = df["label"].value_counts().sort_index()
  max_count = label_count.max()
  max_rate = max_count / len(df)
  terms = tfidf.get_feature_names_out()
  order_centroids = kmeans.cluster_centers_.argsort()[:, ::-1] 
  label_id = label_count.idxmax()
  key_words = [terms[ind] for ind in order_centroids[label_id, :10]]
  return f"{', '.join(key_words)}", str(df[df["label"] == label_id]["result"].head(5).tolist())

def pinpoint_the_cause() -> tuple:
  obj1 = xp_tool.CallAi(API_KEY,BASE_URL)
  obj1.prompt = """
    你是用户反馈聚类结果的业务校验助手，仅需基于以下业务规则，判断聚类结果是否 “理想”，最终仅输出 “是” 或 “否”，无需额外文字。
    业务判断规则（必须严格遵循）
    簇内所有反馈需围绕1 个核心问题类型（如 “费用异常”“数据错误”“审核缓慢”“功能故障”），无跨类型反馈混入；
    不同簇的核心问题类型需完全不同，无重复归类；
    单个簇样本数≥2 条（极特殊新问题除外，若存在单样本簇则直接判定为不理想）。
    请你仅根据上述规则和输入内容，判断该聚类结果是否理想，最终仅输出 “是” 或 “否”。
  """
  for i in range(1,21):
    key_words,result = kmeans(i)
    if obj.chat(result) == '否':
        continue
    else:
        break
  return key_words,result
