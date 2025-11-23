import os
import numpy as np
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
 
# 確保 static 資料夾存在
os.makedirs("static", exist_ok=True)

MODEL_PATH = os.path.join("model", "efficientnetv2s_best_tf210.h5")
IMG_SIZE = (300, 300)

# 英文學名對應中文
CLASS_NAMES = [
    "Acridotheres javanicus", "Acridotheres tristis", "Actitis hypoleucos",
    "Alcedo atthis", "Amaurornis phoenicurus", "Anarhynchus alexandrinus",
    "Anas crecca", "Anas zonorhyncha", "Aplonis panayensis", "Ardea alba",
    "Ardea cinerea", "Ardea coromanda", "Ardea intermedia", "Bambusicola sonorivox",
    "Chlidonias hybrida", "Columba livia", "Copsychus malabaricus", "Copsychus saularis",
    "Dendrocitta formosae", "Dicrurus macrocercus", "Egretta garzetta", "Elanus caeruleus",
    "Fulica atra", "Gallinula chloropus", "Gorsachius melanolophus", "Gracupica nigricollis",
    "Heterophasia auricularis", "Himantopus himantopus", "Hirundo javanica", "Hirundo rustica",
    "Hydrophasianus chirurgus", "Hypsipetes leucocephalus", "Lanius cristatus", "Lanius schach",
    "Liocichla steerii", "Lonchura punctulata", "Lophospiza trivirgata", "Lophura swinhoii",
    "Milvus migrans", "Motacilla alba", "Motacilla cinerea", "Motacilla tschutschensis",
    "Myophonus insularis", "Nycticorax nycticorax", "Passer montanus", "Pericrocotus solaris",
    "Phasianus colchicus", "Phoenicurus auroreus", "Phoenicurus fuliginosus", "Pica serica",
    "Platalea minor", "Pluvialis fulva", "Pomatorhinus musicus", "Prinia inornata",
    "Psilopogon nuchalis", "Pycnonotus sinensis", "Pycnonotus taivanus", "Recurvirostra avosetta",
    "Spilopelia chinensis", "Spilornis cheela", "Streptopelia orientalis", "Streptopelia tranquebarica",
    "Tachybaptus ruficollis", "Thinornis dubius", "Tringa glareola", "Tringa nebularia",
    "Trochalopteron morrisonianum", "Urocissa caerulea", "Yuhina brunneiceps", "Zosterops simplex"
]

BIRD_NAME_MAP = {
   "Acridotheres javanicus": "白尾八哥",
   "Acridotheres tristis": "家八哥",
   "Actitis hypoleucos": "磯鷸",
   "Alcedo atthis": "翠鳥",
   "Amaurornis phoenicurus": "白腹秧雞",
   "Anarhynchus alexandrinus": "東方環頸鴴",
   "Anas crecca": "小水鴨",
   "Anas zonorhyncha": "花嘴鴨",
   "Aplonis panayensis": "亞洲輝椋鳥",
   "Ardea alba": "大白鷺",
   "Ardea cinerea": "蒼鷺",
   "Ardea coromanda": "黃頭鷺",
   "Ardea intermedia": "中白鷺",
   "Bambusicola sonorivox": "台灣竹雞",
   "Chlidonias hybrida": "黑腹燕鷗",
   "Columba livia": "野鴿",
   "Copsychus malabaricus": "白腰鵲鴝",
   "Copsychus saularis": "鵲鴝",
   "Dendrocitta formosae": "樹鵲",
   "Dicrurus macrocercus": "大卷尾",
   "Egretta garzetta": "小白鷺",
   "Elanus caeruleus": "黑翅鳶",
   "Fulica atra": "白冠雞",
   "Gallinula chloropus": "紅冠水雞",
   "Gorsachius melanolophus": "黑冠麻鷺",
   "Gracupica nigricollis": "黑領椋鳥",
   "Heterophasia auricularis": "白耳畫眉",
   "Himantopus himantopus": "高蹺鴴",
   "Hirundo javanica": "洋燕",
   "Hirundo rustica": "家燕",
   "Hydrophasianus chirurgus": "水雉",
   "Hypsipetes leucocephalus": "紅嘴黑鵯",
   "Lanius cristatus": "紅尾伯勞",
   "Lanius schach": "棕背伯勞",
   "Liocichla steerii": "黃胸藪眉",
   "Lonchura punctulata": "斑文鳥",
   "Lophospiza trivirgata": "鳳頭蒼鷹",
   "Lophura swinhoii": "藍腹鴴",
   "Milvus migrans": "黑鳶",
   "Motacilla alba": "白鶺鴒",
   "Motacilla cinerea": "灰鶺鴒",
   "Motacilla tschutschensis": "東方黃鶺鴒",
   "Myophonus insularis": "台灣紫嘯鶇",
   "Nycticorax nycticorax": "夜鷺",
   "Passer montanus": "麻雀",
   "Pericrocotus solaris": "灰喉山椒鳥",
   "Phasianus colchicus": "環頸雉",
   "Phoenicurus auroreus": "黃尾鴝",
   "Phoenicurus fuliginosus": "鉛色水鶇",
   "Pica serica": "普通喜鵲",
   "Platalea minor": "黑面琵鷺",
   "Pluvialis fulva": "金斑鴴",
   "Pomatorhinus musicus": "小彎嘴",
   "Prinia inornata": "褐頭鷦鶯",
   "Psilopogon nuchalis": "五色鳥",
   "Pycnonotus sinensis": "白頭翁",
   "Pycnonotus taivanus": "烏頭翁",
   "Recurvirostra avosetta": "反嘴鷸",
   "Spilopelia chinensis": "珠頸斑鳩",
   "Spilornis cheela": "大冠鷲",
   "Streptopelia orientalis": "金背鳩",
   "Streptopelia tranquebarica": "紅鳩",
   "Tachybaptus ruficollis": "小鸊鷉",
   "Thinornis dubius": "小環頸鴴",
   "Tringa glareola": "鷹斑鷸",
   "Tringa nebularia": "青足鷸",
   "Trochalopteron morrisonianum": "金翼白眉",
   "Urocissa caerulea": "台灣藍鵲",
   "Yuhina brunneiceps": "冠羽畫眉",
   "Zosterops simplex": "斯氏繡眼"
}

app = Flask(__name__)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

@app.route("/", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        file = request.files.get("file")
        if file:
            file_path = os.path.join("static", file.filename)
            file.save(file_path)

            img = load_img(file_path, target_size=IMG_SIZE)
            img = img_to_array(img)
            img = np.expand_dims(img, axis=0)
            img = tf.keras.applications.efficientnet_v2.preprocess_input(img)

            preds = model.predict(img)[0]
            top_idx = np.argmax(preds)
            bird_en = CLASS_NAMES[top_idx]
            bird_cn = BIRD_NAME_MAP.get(bird_en, bird_en)
            top_conf = preds[top_idx] * 100

            top3 = [(BIRD_NAME_MAP.get(CLASS_NAMES[i], CLASS_NAMES[i]), preds[i] * 100)
                    for i in preds.argsort()[-3:][::-1]]

            return render_template(
                "index.html",
                prediction=bird_cn,
                confidence=f"{top_conf:.2f}%",
                top3=top3,
                img_path=file_path,
            )
    return render_template("index.html")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)