import operator
from collections import defaultdict

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

chapterByNo = {
    "01": "Infections",
    "02": "Neoplasms",
    "03": "Blood",
    "04": "Immune system",
    "05": "Endocrine, nutritional, metabolic",
    "06": "Mental and behavioural",
    "07": "Sleep-wake",
    "08": "Nervous system",
    "09": "Eye and adnexa",
    "10": "Ear and mastoid",
    "11": "Circulatory system",
    "12": "Respiratory system",
    "13": "Digestive system",
    "14": "Skin",
    "15": "Musculoskeletal system ...",
    "16": "Genitourinary System",
    "17": "Sexual health",
    "18": "Pregnancy, childbirth ...",
    "19": "Perinatal and neonatal",
    "20": "Developmental anomalies",
    "21": "Symptoms, signs, findings ...",
    "22": "Injury, poisoning, ...",
    "23": "External causes",
    "24": "Factors influencing health ...",
    "25": "Codes for special purposes",
    "26": "Traditional Medicine",
    "V": "Functioning",
    "X": "Extension Codes"
}

token_endpoint = 'https://icdaccessmanagement.who.int/connect/token'
client_id = '109fd0b9-a8a3-4eb9-a9d7-02b301d52e78_458726a5-9a50-47b6-a9c5-c532f506e563'
client_secret = 'yAjJcwsi9iS3kmQry4VINsB4OtEjl3FztT67kwe/Z3w='
scope = 'icdapi_access'
grant_type = 'client_credentials'

payload = {'client_id': client_id,
           'client_secret': client_secret,
           'scope': scope,
           'grant_type': grant_type}

stop_words = ['abnormal', 'upper', 'acute', 'allergic', 'decreased', 'increased', 'aching', 'acute', 'adrenal',
              'consumption', 'altered', 'pressure', 'nos', 'blurred', 'body', 'transplant', 'pain', 'murmur', 'late',
              'early', 'drainage', 'abstains from', 'at home', 'at work', 'abdominal']


def search(query=None):
    r = requests.post(token_endpoint, data=payload, verify=False).json()
    token = r['access_token']
    uri = 'https://id.who.int/icd/release/11/2018/mms/search?q={}'.format(query)
    headers = {'Authorization': 'Bearer ' + token,
               'Accept': 'application/json',
               'Accept-Language': 'en'}
    r = requests.get(uri, headers=headers, verify=False)
    temp = defaultdict(list)
    for ch in r.json()['DestinationEntities']:
        print(ch)
        temp[chapterByNo[ch['Chapter']]].append(ch['Score'])
    res = {ch: max(scores) for ch, scores in temp.items()}
    typ = max(res.items(), key=operator.itemgetter(1))[0] if res else None
    typ = None if typ in ["Codes for special purposes", "Traditional Medicine", "Functioning", "Extension Codes",
                          "Symptoms, signs, findings ..."] else typ

    print(query, "=", typ)
    return typ
