import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.metrics import roc_auc_score
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device( "cpu")
path = "../model\second_stage\mymodel_epoch16.pth"
model = torch.load(path)
# 加载BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',Truncation=True)

model.to(device)
test_text = [


"Microsoft has delayed the rollout of XP Service Pack Two  SP2  for home users, and some customers will not be able to download it until the end of the month. The new code was supposed to go out on Microsoft #39;s Automatic Update service on "


]

#分类中心
c = [0.9563664793968201, -2.8526737689971924, 2.3940420150756836, 0.3691411316394806, 1.7371948957443237,
     1.812051773071289, -2.300105333328247, -4.0499773025512695, -5.0134124755859375, 3.460848331451416,
     -3.297334909439087, -4.956443786621094, -2.1571218967437744, -0.5635120272636414, 4.161524772644043,
     1.923145055770874, -4.488897800445557, 2.709360122680664, 0.9623228311538696, -3.6044492721557617]


#分类半径
r= 0.1427691271561678



def  text_to_chat(text):

    # text = "This is a sample text for BERT feature extraction."

    # 使用tokenizer将文本转换为token ID和attention mask
    tokens = tokenizer.encode_plus(text, max_length=300,truncation=True,padding=True, add_special_tokens=True, return_tensors='pt')
    tokens.to(device)
    input_ids = tokens['input_ids']
    attention_mask = tokens['attention_mask']

    # 使用BERT模型处理输入，得到特征向量
    with torch.no_grad():
        output = model(input_ids, attention_mask)
        feature_vector = output[0].cpu().numpy()[0].tolist()
    feature = []
    for i in feature_vector:
        feature.append(i*10)
    # print(feature)
    return feature


#求列表间的欧式距离
def ou(a,b):
    a = np.array(a)
    b = np.array(b)

    dist = np.linalg.norm(a-b)
    return  dist


for i in test_text:


    if ou(c, text_to_chat(i)) > r:
        print("所检测文本更可能是人类所写")
    else:
        print("所检测文本更可能是chatgpt所生成")

